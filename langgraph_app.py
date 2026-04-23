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
from agents.verifier import verifier_evaluate_step, global_evaluate_step
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

def vanilla_agent_step(state: GraphState) -> Dict[str, Any]:
    """
    [修改] 朴素大模型基线节点
    升级为使用 70B 大模型作为 Strong Baseline。
    """
    logger.info("=== Vanilla Agent 节点开始生成朴素回复 (基于 70B 强基线) ===")
    
    # 【修改点】：替换为与 Consultant 相同的 70B 大模型配置
    llm = ChatOpenAI(
            model_name="/home/xyc/qwen-72b-awq",  # 【关键修改】必须与你启动 vLLM 时的模型路径完全一致
            temperature=0.4,
            api_key="EMPTY",                      # vLLM 本地服务默认不需要秘钥，填 "EMPTY" 或随便填都可以
            max_tokens=800,                       # 保留你的物理刹车，这对于控制输出长度非常有用
            model_kwargs={"response_format": {"type": "json_object"}}, # 强制 JSON 输出，非常适合智能体通信
            base_url="http://192.168.123.2:8000/v1"   # 【关键修改】请根据代码运行的位置决定 IP
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位编程导师。请根据学生的回复，用自然语言提供启发。绝对不要直接给出完整的修复代码块。"),
        MessagesPlaceholder(variable_name="chat_history")
    ])
    
    try:
        response = prompt | llm
        content = response.invoke({"chat_history": state.get("messages", [])}).content
    except Exception as e:
        logger.error(f"Vanilla Agent (70B) 生成失败: {e}")
        content = "我明白你的困惑。让我们再仔细看看这行代码，你觉得问题出在哪里？"
        
    return {"messages": [AIMessage(content=content)]}

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
def route_after_student(state: GraphState) -> str:
    """
    [新增] 根据实验模式分流：
    如果是朴素基线，直接走向 Vanilla Agent；否则走向 LLMKT 开始苏格拉底框架。
    """
    mode = state.get("experiment_mode", "Socrat_Full")
    if mode == "Vanilla_Prompting":
        return "vanilla_agent_node"
    return "llmkt_node"

def should_continue_teaching(state: GraphState) -> str:
    """
    条件路由判断：结合 Bug 修复率与贝叶斯知识追踪后验概率的双重检验。
    """
    mode = state.get("experiment_mode", "Socrat_Full")
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

    # 【新增】如果是 Vanilla_Prompting 模式，因跳过了 LLMKT 节点，无 KC 状态，强制豁免阈值
    if mode == "Vanilla_Prompting":
        kc_threshold_met = True

    logger.info(f"【图流转判定】模式: {mode} | 轮次: {turn_count}/{max_turns} | Bug解决率: {bug_resolved:.2f} | 最低KC掌握度: {lowest_prob:.2f}")

    # 终止条件 2：达到最大防死循环限制，强制终止
    if turn_count >= max_turns:
        logger.warning("⚠️ 满足终止条件：达到最大对话轮次限制，遗憾退出并强行终止会话。")
        # 【修改点 1】：拦截原来的 END，转交给全局评估节点
        return "global_evaluator_node"

    # 终止条件 1：双重校验通过 (代码跑通 + 认知达标)
    if bug_resolved >= 0.85 and kc_threshold_met:
        logger.info("🎉 双重达标：Bug 已被成功修复，且核心概念(KC)已内化，进入 SPR 总结反思环节。")
        return "summary_node"
    
    # 异常流转拦截：非理解性修复 (猜对代码，但概念没懂)
    if bug_resolved >= 0.85 and not kc_threshold_met:
        logger.warning("⚠️ 侦测到非理解性修复 (Bug解决但KC未达标)。强行拦截，回流 Consultant 追问底层逻辑。")
        return "consultant_node"
    
    # 继续教学：交由 Consultant (大脑) 进行 MCTS 规划
    if mode == "Vanilla_Prompting":
        logger.info("-> Vanilla 模式：跳过 Consultant，直接进入下一轮...")
        return "turn_manager"
    else:
        logger.info("-> 局势尚未明朗，进入 Consultant 节点进行策略规划...")
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
    workflow.add_node("global_evaluator_node", global_evaluate_step)
    workflow.add_node("vanilla_agent_node", vanilla_agent_step) # 【新增节点】

    # 2. 定义静态流转边 (Edges)
    workflow.set_entry_point("student_node")
    
    # 【修改点】移除直接相连，改为条件路由分流
    # workflow.add_edge("student_node", "llmkt_node")
    workflow.add_conditional_edges(
        "student_node",
        route_after_student,
        {
            "vanilla_agent_node": "vanilla_agent_node",
            "llmkt_node": "llmkt_node"
        }
    )
    
    workflow.add_edge("llmkt_node", "verifier_node")
    workflow.add_edge("vanilla_agent_node", "verifier_node") # 【新增边】朴素节点生成的回复也必须接受红线巡检
    
    workflow.add_edge("consultant_node", "teacher_node")
    workflow.add_edge("teacher_node", "turn_manager")
    workflow.add_edge("turn_manager", "student_node")
    # 【修改点 3】：改变 summary_node 的出口，并让 global_evaluator_node 成为唯一通向 END 的节点
    workflow.add_edge("summary_node", "global_evaluator_node")
    workflow.add_edge("global_evaluator_node", END)

    # 3. 注册条件路由 (Conditional Edges)
    workflow.add_conditional_edges(
        "verifier_node",
        should_continue_teaching,
        {
            "summary_node": "summary_node", 
            "global_evaluator_node": "global_evaluator_node",
            "consultant_node": "consultant_node",
            "turn_manager": "turn_manager"  # 【新增路由出口】专供 Vanilla 模式跳过 Consultant
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