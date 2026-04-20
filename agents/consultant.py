# [重构] 拆分自原 instructor.py，负责后台 MCTS 策略生成与 JSON 输出
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import dotenv
import os
dotenv.load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
# 引用之前定义的模块
from state.graph_state import GraphState
from algorithms.mcts_planner import MCTSPlanner

logger = logging.getLogger(__name__)

# ==========================================
# 1. 定义 Consultant 强制输出的结构化 JSON 格式
# ==========================================
class ConsultantStrategyPayload(BaseModel):
    """
    顾问输出的结构化策略载体，用于精确控制下游的 Teacher 智能体
    """
    strategy_type: str = Field(
        ..., 
        description="MCTS选定的核心动作类型 (如: Elicit_Questioning, Provide_Hint, Explain_Concept, Direct_Correction)"
    )
    focus_kc_id: str = Field(
        ..., 
        description="当前需要重点关注的知识组件(KC)或 Bug 标识"
    )
    internal_reasoning: str = Field(
        ..., 
        description="后台推演逻辑：向 Teacher 解释为什么现在要采用这个策略"
    )
    tactical_draft: str = Field(
        ..., 
        description="具体的战术草案或核心线索，Teacher 将依据此草案生成最终的自然语言回复"
    )

class ConsultantAgent:
    """
    顾问智能体 (大脑)
    职责：结合 MCTS 规划引擎的宏观决策与 LLM 的微观生成能力，制定苏格拉底教学策略。
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        # 初始化结构化输出的 LLM (要求环境配置好 OPENAI_API_KEY)
        #self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        self.llm = ChatOpenAI(
            model_name="llama-3.3", 
            temperature=0.4,
            api_key="vllm-local-service", 
            #max_tokens=800,          # 【核心修复】加上这个！强制最多只准生成800个字
            base_url="http://192.168.123.8:8000/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        '''
        self.llm = ChatOpenAI(
            model_name="qwen3.5-plus", 
            temperature=0.4,
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        '''
        self.structured_llm = self.llm.with_structured_output(ConsultantStrategyPayload)
        
        # Consultant 的系统提示词：核心在于“幕后指挥”
        self.system_prompt = """
        你是一个精通苏格拉底式教学法与教育心理学的幕后教学顾问 (Consultant)。
        你的任务是基于 MCTS (蒙特卡洛树搜索) 引擎提供的【宏观动作指令】，结合当前学生的【认知状态】与【对话历史】，
        生成一份详细的教学策略草案。
        
        注意：你不是直接和学生对话的人！你的输出将被送给前台的 Teacher 智能体。
        你必须提供清晰的战术线索 (tactical_draft)，告诉 Teacher 下一步该反问什么，或者提示什么。
        绝对不要生成直接修复代码的代码块，除非 MCTS 明确指令为 'Direct_Correction'。

        【极其严格的红线要求】
        请只输出合法的 JSON 格式对象！绝对不要包含任何前导语、结束语或 ```json 这样的 Markdown 标记！
        你的 JSON 必须且只能包含以下 4 个完全一致的 Key，不能修改任何一个字母（以下为格式示例，注意转义）：
        {{
            "strategy_type": "必须填入 MCTS 规划的动作名称",
            "focus_kc_id": "填入相关的知识组件 ID",
            "internal_reasoning": "填写后台推演逻辑",
            "tactical_draft": "填写具体的战术草案"
        }}
        
        绝对禁止输出类似 {{"action_mode": "...", "reason": "..."}} 这样篡改键名的格式！

        【新增严格约束】：
        在编写 "tactical_draft" 时，【绝对不允许】包含任何具体的 Python 代码！
        你的草案只能是自然语言的指导策略（例如：“请反问学生列表长度和最大索引之间的数学关系”）。
        """

    def generate_strategy(self, state: GraphState, mcts_action: str) -> Dict[str, Any]:
        """
        调用 LLM 将 MCTS 的宏观动作细化为具体的 JSON 策略。
        """
        # 1. 提取当前认知画像 (寻找薄弱点)
        student_kcs = state.get("student_kcs", {})
        weakest_kc_id = "general_understanding"
        weakest_prob = 1.0
        
        for kc_id, kc_state in student_kcs.items():
            if kc_state.posterior_prob < weakest_prob:
                weakest_prob = kc_state.posterior_prob
                weakest_kc_id = kc_id
                
        # 2. 组装给 Consultant LLM 的上下文提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "【系统环境指令】\n"
                     "MCTS 引擎已规划当前的最佳宏观动作为: {mcts_action}\n"
                     "系统检测到学生当前最薄弱的知识点/Bug为: {weakest_kc} (掌握概率: {prob:.2f})\n"
                     "请基于上述信息，输出给 Teacher 的指导策略。")
        ])
        
        chain = prompt | self.structured_llm
        
        try:
            # 3. 生成结构化 JSON 策略
            strategy_obj = chain.invoke({
                "chat_history": state.get("messages", []),
                "mcts_action": mcts_action,
                "weakest_kc": weakest_kc_id,
                "prob": weakest_prob
            })
            return strategy_obj.model_dump()
            
        except Exception as e:
            logger.error(f"Consultant 策略生成失败，使用降级兜底策略: {e}")
            # 兜底容错机制，防止 LangGraph 崩溃
            return {
                "strategy_type": mcts_action,
                "focus_kc_id": weakest_kc_id,
                "internal_reasoning": "大模型生成异常，启用系统默认兜底推理。",
                "tactical_draft": "请继续使用苏格拉底式提问引导学生思考当前的代码逻辑。"
            }

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def consultant_node_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 顾问节点：
    1. 调用 MCTSPlanner 获取宏观最优动作
    2. 调用 ConsultantAgent 生成详细 JSON 策略草案
    注：替代了原来 mcts_planner.py 里的粗糙包装。
    """
    logger.info("=== Consultant 节点开始运作 ===")
    
    # 步骤 1: 蒙特卡洛树搜索 (纯算法推演)
    planner = MCTSPlanner(num_simulations=6, max_depth=3)
    mcts_result = planner.search(state)
    best_action = mcts_result.get("strategy_type", "Elicit_Questioning")
    
    # 步骤 2: 大模型策略细化 (业务逻辑具象化)
    agent = ConsultantAgent()
    detailed_strategy = agent.generate_strategy(state, best_action)
    
    logger.debug(f"生成的详细策略: {detailed_strategy}")
    
    # 将包含详细 draft 的策略注入图中，等待 Teacher 读取
    return {
        "current_strategy": detailed_strategy
    }