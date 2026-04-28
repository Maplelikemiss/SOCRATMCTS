# [修复与解耦] 拆分单轮评估与全局评估，彻底解决兜底污染和计算冗余问题
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import dotenv
import os

dotenv.load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

# 引用全局状态定义
from state.graph_state import GraphState

logger = logging.getLogger(__name__)

# ==========================================
# 1. 解耦的 Pydantic 数据结构定义
# ==========================================
class SingleTurnEvaluation(BaseModel):
    """第一层：单轮对话即时评估 (严格二进制 0.0 或 1.0)"""
    bug_resolved: float = Field(
        ..., description="系统状态判定：Bug 是否已被学生自行修复？(0.0=未修复, 1.0=已完全修复)"
    )
    ndar: float = Field(
        ..., description="[核心红线] 无直接答案率 (No Direct Answer Rate)。只要包含一次直接代码修复或直白答案，必须打 0.0。未泄露打 1.0。"
    )
    prr: float = Field(
        ..., description="问题相关率 (Problem Relevance Rate)。最新提问是否精准锁定认知断层且无幻觉？(0.0 或 1.0)"
    )
    spr: float = Field(
        ..., description="总结合格率 (Summary Pass Rate)。当攻克知识点后，系统是否进行了无幻觉的概念收敛性总结？(0.0 或 1.0)"
    )
    iar: float = Field(
        ..., description="指令遵循率 (Instruction Adherence Rate)。Teacher 是否严格遵守了 Consultant 的策略指令？(0.0 或 1.0)"
    )

class GlobalEvaluation(BaseModel):
    """第二层：多轮教学过程全局评估 (李克特 1-5 分制，映射为 0.2 - 1.0 浮点数)"""
    logicality: float = Field(
        ..., description="逻辑性 (Logicality)。多轮对话是否遵循先概念后语法的严密递进？(0.2, 0.4, 0.6, 0.8, 1.0)"
    )
    repetitiveness: float = Field(
        ..., description="重复性 (Repetitiveness)。【逆向指标】提问是否陷入死循环？得分越高代表重复率越低、表现越好。(0.2 到 1.0)"
    )
    guidance: float = Field(
        ..., description="引导能力 (Guidance)。全局战略性隐喻与代码追踪的整体引导效能。(0.2 到 1.0)"
    )
    flexibility: float = Field(
        ..., description="灵活性/ZPD自适应 (Flexibility)。遇到认知负荷时是否主动降级难度或调用角色互换？(0.2 到 1.0)"
    )
    clarity: float = Field(
        ..., description="清晰度 (Clarity)。启发式问题在自然语言表达上是否语义通透？(0.2 到 1.0)"
    )

# ==========================================
# 2. 单轮验证智能体 (巡检员)
# ==========================================
class SingleTurnVerifierAgent:
    """负责每一轮的即时状态与红线巡检"""
    def __init__(self, model_name: str = "qwen3.5-plus", temperature: float = 0.0):
        '''
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature,
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        '''
        self.llm = ChatOpenAI(
            model_name="/home/xyc/qwen-72b-awq", 
            temperature=temperature,
            api_key="EMPTY",
            base_url="http://192.168.123.2:8000/v1"
        )
        self.structured_llm = self.llm.with_structured_output(SingleTurnEvaluation)
        
        self.system_prompt = """
        你是一个极其严苛且绝对客观的教育学评估专家。
        请阅读以下对话，仅对**当前最新一轮**的教学状态进行单轮即时评估。
        
        【单轮打分红线规则】
        1. bug_resolved: 必须看到学生在最新回复中输出了正确的代码或清晰的修复逻辑，才能给 1.0，否则一律 0.0。
        2. ndar（无直接答案率）: 生死红线！只要 Teacher 越权给出了带修复结果的代码块或直接指出了修复点，NDAR 必须为 0.0。未泄露打 1.0。如果 Teacher 在自然语言中直接指出了极其具体的修改方案（例如：“你应该把 range(1, len(nums) + 1) 改成 range(1, len(nums))”、“你应该去掉 +1”），也必须打 0.0！
        3. prr（问题相关率）: 最新提问精准锁定问题且无幻觉打 1.0，否则 0.0。
        4. spr（总结合格率）: 若触发知识点总结且无幻觉打 1.0，否则 0.0（若未到总结阶段默认 1.0）。
        5. iar（指令遵循率）: Teacher 严格遵守了 Consultant 的策略打 1.0，否则 0.0。

        【极其严格的格式要求 - 必读】
        你必须且只能输出一个合法的 JSON 对象。
        绝对禁止输出任何前言、后语、解释性文字，也绝对禁止使用 ```json 等 Markdown 标记！
        你的输出必须完全匹配以下 JSON 结构（键名严格小写，值为 0.0 或 1.0 的浮点数）：
        {{
            "bug_resolved": 0.0,
            "ndar": 1.0,
            "prr": 1.0,
            "spr": 1.0,
            "iar": 1.0
        }}
        """

    def _get_default_scores(self) -> Dict[str, float]:
        return SingleTurnEvaluation(
            bug_resolved=0.0, ndar=1.0, prr=1.0, spr=1.0, iar=1.0
        ).model_dump()

    def evaluate(self, state: GraphState) -> Dict[str, float]:
        messages = state.get("messages", [])
        if not messages:
            return self._get_default_scores()
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "请根据最新对话状态输出单轮评估。必须且只能输出包含 bug_resolved, ndar, prr, spr, iar 5个键的 JSON。")
        ])
        
        chain = prompt | self.structured_llm
        
        for attempt in range(3):
            try:
                return chain.invoke({"chat_history": messages}).model_dump()
            except Exception as e:
                logger.warning(f"⚠️ SingleTurnVerifier 结构化打分失败 (尝试 {attempt+1}/3): {e}")
                
        logger.error("❌ SingleTurnVerifier 连续打分失败，启用安全底分")
        return self._get_default_scores()

# ==========================================
# 3. 全局评估智能体 (终局定调员)
# ==========================================
class GlobalEvaluatorAgent:
    """负责对话结束时的整体宏观复盘评估"""
    def __init__(self, model_name: str = "qwen3.5-plus", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model_name="/home/xyc/qwen-72b-awq", 
            temperature=temperature,
            api_key="EMPTY",
            base_url="http://192.168.123.189:8000/v1"
        )
        self.structured_llm = self.llm.with_structured_output(GlobalEvaluation)
        
        self.system_prompt = """
        你是一个资深的教育评估专家。
        本次代码辅导对话已经结束，请你基于完整的对话历史，进行事后复盘（Post-mortem Review），并对教学过程的 5 个宏观全局指标进行打分。
        
        【全局评估规则】(采用 0.2, 0.4, 0.6, 0.8, 1.0 的五档浮点数)
        1. logicality（逻辑性）: 多轮对话是否遵循先概念后语法的严密递进？
        2. repetitiveness（重复性）: 【逆向指标】提问是否陷入死循环？得分越高代表重复率越低、表现越好（视角开阔不绕弯子打 0.8或1.0，死循环打 0.2或0.4）。
        3. guidance（引导能力）: 全局战略性隐喻与代码追踪的整体引导效能。
        4. flexibility（灵活性）: 遇到认知负荷时是否主动降级难度或调用角色互换？
        5. clarity（清晰度）: 启发式问题在自然语言表达上是否语义通透？

        【极其严格的格式要求 - 必读】
        你必须且只能输出一个合法的 JSON 对象。
        绝对禁止输出任何前言、后语、解释性文字，也绝对禁止使用 ```json 等 Markdown 标记！
        你的输出必须完全匹配以下 JSON 结构（键名严格小写，值为 0.2 到 1.0 之间的浮点数）：
        {{
            "logicality": 0.6,
            "repetitiveness": 0.8,
            "guidance": 0.6,
            "flexibility": 0.8,
            "clarity": 0.6
        }}
        """

    def _get_default_scores(self) -> Dict[str, float]:
        return GlobalEvaluation(
            logicality=0.6, repetitiveness=0.6, guidance=0.6, flexibility=0.6, clarity=0.6
        ).model_dump()

    def evaluate(self, state: GraphState) -> Dict[str, float]:
        messages = state.get("messages", [])
        if not messages:
            return self._get_default_scores()
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "辅导已结束。请输出全局维度的打分。必须包含 logicality, repetitiveness, guidance, flexibility, clarity 5个键的 JSON。")
        ])
        
        chain = prompt | self.structured_llm
        
        for attempt in range(3):
            try:
                return chain.invoke({"chat_history": messages}).model_dump()
            except Exception as e:
                logger.warning(f"⚠️ GlobalEvaluator 结构化打分失败 (尝试 {attempt+1}/3): {e}")
                
        logger.error("❌ GlobalEvaluator 连续打分失败，启用中等底分")
        return self._get_default_scores()


# ==========================================
# 4. 接入 LangGraph 的 Node 执行函数
# ==========================================
def verifier_evaluate_step(state: GraphState) -> Dict[str, Any]:
    """原 Verifier 节点：现在仅做单轮即时巡检"""
    logger.info("=== Verifier 节点执行单轮红线巡检 ===")
    agent = SingleTurnVerifierAgent()
    scores = agent.evaluate(state)
    logger.debug(f"当前 Bug 解决状态: {scores.get('bug_resolved', 0.0):.2f} | 核心红线(NDAR): {scores.get('ndar', 1.0):.2f}")
    
    return {
        "verifier_scores": scores,
        "verifier_history": [scores] # 仅追加单轮红线分数
    }

def global_evaluate_step(state: GraphState) -> Dict[str, Any]:
    """新增图节点：在流程即将结束时，进行唯一一次的全局宏观打分"""
    logger.info("=== Global Evaluator 节点进行终局宏观定调 ===")
    agent = GlobalEvaluatorAgent()
    global_scores = agent.evaluate(state)
    logger.debug(f"全局打分完成: {global_scores}")
    
    return {
        "global_evaluation_scores": global_scores
    }