# [修复] LangGraph 非线性状态机与节点流转逻辑
from dotenv import load_dotenv
load_dotenv()  # 自动读取并加载同级目录下的 .env 文件
import logging
import sys
from typing import Dict, Any
import os
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# 导入图状态定义
from state.graph_state import GraphState

# 导入所有节点执行函数
from agents.student import student_node_step
from algorithms.llmkt_bayesian import llmkt_bayesian_update_step
from agents.verifier import verifier_evaluate_step
from agents.teacher import teacher_node_step

# 【关键修复 1】纠正导入路径：导入真正封装了大模型推理与 MCTS 调用的 Consultant Agent 节点，
# 而不是仅仅导入底层无状态的 MCTS 纯算法步骤。这样才能保证 tactical_draft 的生成。
from agents.consultant import consultant_node_step

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SocratMCTS")

# ==========================================
# 新增的辅助节点 (Helpers)
# ==========================================
def turn_manager_step(state: GraphState) -> Dict[str, Any]:
    """
    防死循环回合管理器
    在教师发言完毕进入下一轮学生提问前，显式递增 turn_count。
    保证主干状态机的生命周期受到绝对控制。
    """
    current_turns = state.get("turn_count", 0)
    return {"turn_count": current_turns + 1}

def summary_node_step(state: GraphState) -> Dict[str, Any]:
    """
    概念收敛性总结节点
    为支撑论文中 SPR (总结合格率) 这一红线指标而添加。
    当检测到 Bug 修复后，系统不应直接终止，而应在此节点进行复盘。
    """
    logger.info("👩‍🏫 [Teacher]: (系统总结) 恭喜你完成了代码修复！让我们来回顾一下刚才解决的核心概念瓶颈...")
    # 真实应用中，这可以调用大模型基于此前的状态图生成总结文本。
    # 这里直接抛出固定的标识，表示系统已成功进入总结状态并完结教学。
    return {"current_strategy": {"strategy_type": "Concept_Summary_Complete"}}


# ==========================================
# 核心条件路由 (Conditional Edges)
# ==========================================
def should_continue_teaching(state: GraphState) -> str:
    """
    条件路由判断：决定是进入下一轮教学，还是结束当前会话。
    """
    scores = state.get("verifier_scores", {})
    bug_resolved = scores.get("bug_resolved", 0.0)
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 8)

    logger.info(f"【图流转判定】当前轮次: {turn_count}/{max_turns} | Bug 解决概率: {bug_resolved:.2f}")

    # 终止条件 1：代码 Bug 已被明确修复，路由至总结反思节点
    if bug_resolved >= 0.85:
        logger.info("🎉 满足终止条件：Bug 已被成功修复，进入 SPR 总结反思环节。")
        return "summary_node"
    
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
    
    # 【关键修复 1 延续】将状态图的 consultant_node 正确绑定至 consultant_node_step
    workflow.add_node("consultant_node", consultant_node_step) 
    
    workflow.add_node("teacher_node", teacher_node_step)
    
    # 注册新引入的关键节点
    workflow.add_node("turn_manager", turn_manager_step)
    workflow.add_node("summary_node", summary_node_step)

    # 2. 定义静态流转边 (Edges)
    workflow.set_entry_point("student_node")
    workflow.add_edge("student_node", "llmkt_node")
    workflow.add_edge("llmkt_node", "verifier_node")
    
    # 路由节点的决策出口
    # verifier_node 的下一步由 add_conditional_edges 动态决定
    
    workflow.add_edge("consultant_node", "teacher_node")
    # 教师发言完毕后，经过回合管理器拦截递增，再重回学生节点构成闭环
    workflow.add_edge("teacher_node", "turn_manager")
    workflow.add_edge("turn_manager", "student_node")
    
    # 总结完毕后，状态机安全终结
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
        "student_persona": "stubborn",
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
            
            if node_name in ["student_node", "teacher_node"]:
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