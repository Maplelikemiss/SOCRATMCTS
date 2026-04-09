# [重构] 替代原 consultant_teacher_socratic_teaching_system.py，定义 LangGraph 非线性状态机与节点流转逻辑
from dotenv import load_dotenv
load_dotenv()  # 自动读取并加载同级目录下的 .env 文件
import logging
import sys
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# 导入图状态定义
from state.graph_state import GraphState

# 导入所有节点执行函数
from agents.student import student_node_step
from algorithms.llmkt_bayesian import llmkt_bayesian_update_step
from agents.verifier import verifier_evaluate_step
from agents.consultant import consultant_node_step
from agents.teacher import teacher_node_step

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SocratMCTS")

def should_continue_teaching(state: GraphState) -> str:
    """
    条件路由判断：决定是进入下一轮教学，还是结束当前会话。
    """
    scores = state.get("verifier_scores", {})
    bug_resolved = scores.get("bug_resolved", 0.0)
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 8)  # 默认最大 8 轮，防止死循环

    logger.info(f"【图流转判定】当前轮次: {turn_count}/{max_turns} | Bug 解决概率: {bug_resolved:.2f}")

    # 终止条件 1：代码 Bug 已被明确修复
    if bug_resolved >= 0.85:
        logger.info("🎉 满足终止条件：Bug 已被成功修复，教学目标达成！")
        return END
    
    # 终止条件 2：达到最大防死循环限制
    if turn_count >= max_turns:
        logger.warning("⚠️ 满足终止条件：达到最大对话轮次限制，强行终止会话。")
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

    # 2. 定义静态流转边 (Edges)
    workflow.set_entry_point("student_node")
    workflow.add_edge("student_node", "llmkt_node")
    workflow.add_edge("llmkt_node", "verifier_node")
    # 注意：verifier_node 的出口由下面的 add_conditional_edges 动态决定
    workflow.add_edge("consultant_node", "teacher_node")
    workflow.add_edge("teacher_node", "student_node")

    # 3. 注册条件路由 (Conditional Edges)
    workflow.add_conditional_edges(
        "verifier_node",
        should_continue_teaching,
        {
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
    # 构建图应用
    socrat_app = build_socrat_mcts_graph()
    
    # 初始化启动状态 (冷启动)
    initial_state = {
        "messages": [],
        "student_kcs": {},
        "global_kl_shift": 0.0,
        "current_strategy": None,
        "verifier_scores": {},
        "is_simulation": False,
        "student_persona": "stubborn", # 注入固执的对抗性学生画像进行压力测试
        "turn_count": 0,
        "max_turns": 5
    }
    
    print("\n" + "="*50)
    print("🚀 开始运行 SocratMCTS 教学模拟器 🚀")
    print(f"当前注入的对抗学生画像: {initial_state['student_persona']}")
    print("="*50 + "\n")
    
    # 启动图流转引擎
    try:
        # stream() 允许我们观察到图中每一个节点流转的增量状态
        for output in socrat_app.stream(initial_state, config={"recursion_limit": 50}):
            # 获取当前刚执行完的节点名称
            node_name = list(output.keys())[0]
            node_state = output[node_name]
            
            # 仅在学生和老师节点输出对话，以便于观察
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