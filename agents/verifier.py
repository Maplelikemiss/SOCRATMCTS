# [重构] 改造原 verifier.py，对接全新的 9 维评估体系计算
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 引用全局状态定义
from state.graph_state import GraphState

logger = logging.getLogger(__name__)

# ==========================================
# 1. 定义 9 维评估体系的 Pydantic 结构
# ==========================================
class NineDimEvaluation(BaseModel):
    """
    9 维立体评估体系打分结构 (取值范围均为 0.0 - 1.0)
    这些维度将共同支撑 MCTS 的 Default Policy (Rollout) 与图的终止判断
    """
    bug_resolved: float = Field(
        default=0.0, 
        description="核心指标：Bug 是否已被学生自行修复或代码已经正确？(0=未修复, 1=已完全修复)"
    )
    student_understanding_score: float = Field(
        default=0.5, 
        description="学生当前对核心知识点的理解程度 (这可以反哺 LLMKT 的观测更新)"
    )
    socratic_guidance: float = Field(
        default=0.5, 
        description="教师是否坚持了苏格拉底式启发？(0=直接给答案, 1=完美的提问与启发)"
    )
    tone_and_empathy: float = Field(
        default=0.5, 
        description="教师回复的语气亲和力与同理心，是否缓解了学生的焦虑"
    )
    clarity: float = Field(
        default=0.5, 
        description="教师表达的清晰度与准确性"
    )
    relevance: float = Field(
        default=0.5, 
        description="对话内容与当前 Bug / 知识点的相关性，是否跑题"
    )
    engagement: float = Field(
        default=0.5, 
        description="学生的参与度与积极性"
    )
    cognitive_struggle: float = Field(
        default=0.5, 
        description="学生是否正在经历良性的认知挣扎 (直接给答案得0分，一直循环提问导致烦躁也得低分)"
    )
    independence: float = Field(
        default=0.0, 
        description="学生独立解决问题的倾向性与实际表现"
    )

class VerifierAgent:
    """
    验证器智能体 (裁判)
    职责：独立观察全局对话，进行 9 维度的客观打分评估。
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        # 裁判需要绝对客观，temperature 必须为 0.0
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        '''
        self.llm = ChatOpenAI(
            model_name="qwen-2.5-72b-instruct", 
            temperature=0.4,
            api_key="你的开源模型API_KEY或者随便填", 
            base_url="http://localhost:8000/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        '''
        self.structured_llm = self.llm.with_structured_output(NineDimEvaluation)
        
        self.system_prompt = """
        你是一个严苛且绝对客观的教育学评估专家（Verifier）。
        请阅读以下教师（Teacher）与学生（Student）的对话历史，对当前的教学状态进行 9 个维度的打分。
        
        【打分红线规则 - 极其重要】
        1. 所有的打分必须严格介于 0.0 到 1.0 之间。
        2. 如果发现 Teacher 越权，直接给出了带有正确修复结果的代码块，`socratic_guidance` 和 `cognitive_struggle` 必须打极低分（< 0.2）。
        3. 只有当 Student 在最新的回复中明确给出了正确的思路或正确的代码时，`bug_resolved` 才能打高分（>= 0.85），否则即使 Teacher 给了答案，只要学生没反馈学会，bug 就不算被解决！
        """

    def evaluate(self, state: GraphState) -> Dict[str, float]:
        """
        根据当前对话历史计算 9 维度得分。
        """
        messages = state.get("messages", [])
        
        # 冷启动处理：如果还没有对话，返回默认安全底分
        if not messages:
            return NineDimEvaluation().model_dump()
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "请根据上述最新对话状态，输出 9 维评估打分。务必保持客观！")
        ])
        
        chain = prompt | self.structured_llm
        
        try:
            # 调用大模型执行结构化评估
            eval_result = chain.invoke({"chat_history": messages})
            return eval_result.model_dump()
        except Exception as e:
            logger.error(f"Verifier 结构化打分失败，启用中性底分容错: {e}")
            # 容错机制：大模型 API 异常时，返回默认底盘数据
            return NineDimEvaluation().model_dump()

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def verifier_evaluate_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 验证器节点：
    基于完整的对话历史，调用 VerifierAgent 进行打分，并将结果存入状态。
    """
    logger.info("=== Verifier 节点开始执行 9 维评估 ===")
    
    agent = VerifierAgent()
    scores = agent.evaluate(state)
    
    # 打印核心判断指标，方便调试
    logger.debug(f"当前 Bug 解决概率: {scores.get('bug_resolved', 0.0):.2f} | 苏格拉底指数: {scores.get('socratic_guidance', 0.0):.2f}")
    
    # 状态更新：直接覆盖/更新 GraphState 中的 verifier_scores 字典
    return {
        "verifier_scores": scores
    }