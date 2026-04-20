# [修复] LangGraph 非线性状态机与节点流转逻辑 (引入贝叶斯双重校验与总结闭环)
from dotenv import load_dotenv
load_dotenv()  # 自动读取并加载同级目录下的 .env 文件
import logging
import sys
from typing import Dict, Any
import os

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 导入图状态定义
from state.graph_state import GraphState

# 导入所有节点执行函数
from agents.student import student_node_step
from algorithms.llmkt_bayesian import llmkt_bayesian_update_step
from agents.verifier import verifier_evaluate_step
from agents.teacher import teacher_node_step
from agents.consultant import consultant_node_step

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SocratMCTS")

# ==========================================
# 辅助节点 (Helpers)
# ==========================================
def turn_manager_step(state: GraphState) -> Dict[str, Any]:
    """防死循环回合管理器"""
    current_turns = state.get("turn_count", 0)
    return {"turn_count": current_turns + 1}

def summary_node_step(state: GraphState) -> Dict[str, Any]:
    """
    [重构] 概念收敛性总结节点
    为支撑 SPR (总结合格率) 闭环，调用大模型生成真实的收敛性总结。
    """
    logger.info("👩‍🏫 [Teacher]: 触发概念收敛性总结，正在生成复盘文本...")
    
    # 1. 提取达标的知识点 (KCs)
    student_kcs = state.get("student_kcs", {})
    mastered_kcs = [kc_id for kc_id, kc_state in student_kcs.items() if kc_state.posterior_prob >= 0.90]
    mastered_kcs_str = ", ".join(mastered_kcs) if mastered_kcs else "核心编程概念"
    
    # 2. 实例化总结专属的 LLM
    llm = ChatOpenAI(
        model_name="socrat-teacher-glm4", 
        temperature=0.4,
        api_key="EMPTY", 
        base_url="http://192.168.123.8:8002/v1"  # 指向你的本地 vLLM 或其他服务商地址
    )
    
    # 3. 构建 Prompt 链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深且温和的编程导师。当前学生已经成功修复了代码Bug，并且你判断他们已经掌握了以下核心概念：【{mastered_kcs}】。\n"
                   "请你用不超过80个字，生成一段鼓励性的总结反思，确认他们的学习成果，并彻底结束本次辅导。绝不要再抛出新的问题！"),
        MessagesPlaceholder(variable_name="chat_history")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "mastered_kcs": mastered_kcs_str,
            "chat_history": state.get("messages", [])
        })
        summary_text = response.content
    except Exception as e:
        logger.error(f"总结节点 LLM 生成失败，启用兜底总结: {e}")
        summary_text = f"恭喜你完成了代码修复！从刚才的探讨可以看出，你对 {mastered_kcs_str} 的理解已经非常到位了。编程就是这样一步步排查的过程，继续保持！"
        
    # 4. 封装成 AIMessage 并追加到图状态中
    summary_message = AIMessage(content=summary_text)
    
    # 返回增量 message 和 策略结束标志
    return {
        "messages": [summary_message],
        "current_strategy": {"strategy_type": "Concept_Summary_Complete"}
    }


# ==========================================
# 核心条件路由 (Conditional Edges)
# ==========================================
def should_continue_teaching(state: GraphState) -> str:
    """
    条件路由判断：结合 Bug 修复率与贝叶斯知识追踪后验概率的双重检验。
    """
    scores = state.get("verifier_scores", {})
    bug_resolved = scores.get("bug_resolved", 0.0)
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 8)

    # 1. 提取所有 KC 的后验概率，评估认知底盘
    student_kcs = state.get("student_kcs", {})
    kc_threshold_met = False
    lowest_prob = 1.0

    if student_kcs:
        for kc_id, kc_state in student_kcs.items():
            if kc_state.posterior_prob < lowest_prob:
                lowest_prob = kc_state.posterior_prob
        # 严格执行论文标准：核心概念掌握度必须跨越 0.90 极高阈值
        if lowest_prob >= 0.90:
            kc_threshold_met = True
    else:
        # 冷启动时防误判
        lowest_prob = 0.5

    logger.info(f"【图流转判定】轮次: {turn_count}/{max_turns} | Bug解决率: {bug_resolved:.2f} | 最低KC掌握度: {lowest_prob:.2f}")

    # 终止条件 1：双重校验通过 (代码跑通 + 认知达标)
    if bug_resolved >= 0.85 and kc_threshold_met:
        logger.info("🎉 双重达标：Bug 已被成功修复，且核心概念(KC)已内化，进入 SPR 总结反思环节。")
        return "summary_node"
    
    # 异常流转拦截：非理解性修复 (猜对代码，但概念没懂)
    if bug_resolved >= 0.85 and not kc_threshold_met:
        logger.warning("⚠️ 侦测到非理解性修复 (Bug解决但KC未达标)。强行拦截，回流 Consultant 追问底层逻辑。")
        return "consultant_node"
    
    # 终止条件 2：达到最大防死循环限制，强制终止
    if turn_count >= max_turns:
        logger.warning("⚠️ 满足终止条件：达到最大对话轮次限制，遗憾退出并强行终止会话。")
        return END
        
    # 继续教学：交由 Consultant (大脑) 进行 MCTS 规划
    logger.info("-> 局势尚未明朗，进入 Consultant 节点进行 MCTS 策略规划...")
    return "consultant_node"


def build_socrat_mcts_graph():
    """
    构建并编译 SocratMCTS 非线性状态机
    """
    logger.info("正在组装 SocratMCTS LangGraph 工作流...")
    
    workflow = StateGraph(GraphState)

    # 1. 注册所有的节点 (Nodes)
    workflow.add_node("student_node", student_node_step)
    workflow.add_node("llmkt_node", llmkt_bayesian_update_step)
    workflow.add_node("verifier_node", verifier_evaluate_step)
    workflow.add_node("consultant_node", consultant_node_step) 
    workflow.add_node("teacher_node", teacher_node_step)
    workflow.add_node("turn_manager", turn_manager_step)
    workflow.add_node("summary_node", summary_node_step)

    # 2. 定义静态流转边 (Edges)
    workflow.set_entry_point("student_node")
    workflow.add_edge("student_node", "llmkt_node")
    workflow.add_edge("llmkt_node", "verifier_node")
    
    workflow.add_edge("consultant_node", "teacher_node")
    workflow.add_edge("teacher_node", "turn_manager")
    workflow.add_edge("turn_manager", "student_node")
    workflow.add_edge("summary_node", END)

    # 3. 注册条件路由 (Conditional Edges)
    workflow.add_conditional_edges(
        "verifier_node",
        should_continue_teaching,
        {
            "summary_node": "summary_node", 
            END: END,
            "consultant_node": "consultant_node"
        }
    )

    # 4. 编译返回可执行的 App
    app = workflow.compile()
    logger.info("SocratMCTS 工作流编译成功！")
    return app


# ==========================================
# 本地测试与运行入口
# ==========================================
if __name__ == "__main__":
    socrat_app = build_socrat_mcts_graph()
    
    initial_state = {
        "messages": [],
        "student_kcs": {},
        "global_kl_shift": 0.0,
        "current_strategy": None,
        "verifier_scores": {},
        "is_simulation": False,
        "student_persona": "normal",
        "turn_count": 0,
        "max_turns": 5
    }
    
    print("\n" + "="*50)
    print("🚀 开始运行 SocratMCTS 教学模拟器 🚀")
    print(f"当前注入的对抗学生画像: {initial_state['student_persona']}")
    print("="*50 + "\n")
    
    try:
        for output in socrat_app.stream(initial_state, config={"recursion_limit": 50}):
            node_name = list(output.keys())[0]
            node_state = output[node_name]
            
            if node_name in ["student_node", "teacher_node", "summary_node"]:
                latest_msg = node_state["messages"][-1]
                if isinstance(latest_msg, HumanMessage):
                    print(f"\n👨‍🎓 [Student]: {latest_msg.content}")
                elif isinstance(latest_msg, AIMessage):
                    print(f"\n👩‍🏫 [Teacher]: {latest_msg.content}")
                    
    except Exception as e:
        logger.error(f"框架运行过程中发生严重异常: {e}")
        
    print("\n" + "="*50)
    print("🛑 教学模拟流转结束 🛑")
    print("="*50 + "\n")