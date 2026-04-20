# [重构] 改造原 verifier.py，严格对齐论文《实验方案设计.docx》中的 9 维双层评估体系
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
# 1. 定义论文金标准 9 维双层评估体系的 Pydantic 结构
# ==========================================
class NineDimEvaluation(BaseModel):
    """
    Socratic 教学质量双层评估体系打分结构
    注意：包含一个基础的 Bug 修复判定，外加严格对齐论文的 9 个学术指标。
    """
    # 基础路由指标 (不计入 9 维核心得分，但控制图流转)
    bug_resolved: float = Field(
        ..., 
        description="系统状态判定：Bug 是否已被学生自行修复？(0.0=未修复, 1.0=已完全修复)"
    )

    # --- 第一层：单轮对话即时评估 (严格二进制 0.0 或 1.0) ---
    ndar: float = Field(
        ..., 
        description="[核心红线] 无直接答案率 (No Direct Answer Rate)。只要包含一次直接代码修复或直白答案，必须打 0.0。未泄露打 1.0。"
    )
    prr: float = Field(
        ..., 
        description="问题相关率 (Problem Relevance Rate)。最新提问是否精准锁定认知断层且无幻觉？(0.0 或 1.0)"
    )
    spr: float = Field(
        ..., 
        description="总结合格率 (Summary Pass Rate)。当攻克知识点后，系统是否进行了无幻觉的概念收敛性总结？(0.0 或 1.0)"
    )
    iar: float = Field(
        ..., 
        description="指令遵循率 (Instruction Adherence Rate)。Teacher 是否严格遵守了 Consultant 的策略指令？(0.0 或 1.0)"
    )

    # --- 第二层：多轮教学过程全局评估 (李克特 1-5 分制，映射为 0.2 - 1.0 浮点数) ---
    logicality: float = Field(
        ..., 
        description="逻辑性 (Logicality)。多轮对话是否遵循先概念后语法的严密递进？(0.2, 0.4, 0.6, 0.8, 1.0)"
    )
    repetitiveness: float = Field(
        ..., 
        description="重复性 (Repetitiveness)。【逆向指标】提问是否陷入死循环？得分越高代表重复率越低、表现越好。(0.2 到 1.0)"
    )
    guidance: float = Field(
        ..., 
        description="引导能力 (Guidance)。全局战略性隐喻与代码追踪的整体引导效能。(0.2 到 1.0)"
    )
    flexibility: float = Field(
        ..., 
        description="灵活性/ZPD自适应 (Flexibility)。遇到认知负荷时是否主动降级难度或调用角色互换？(0.2 到 1.0)"
    )
    clarity: float = Field(
        ..., 
        description="清晰度 (Clarity)。启发式问题在自然语言表达上是否语义通透？(0.2 到 1.0)"
    )

class VerifierAgent:
    """
    验证器智能体 (裁判)
    职责：独立观察全局对话，依据论文金标准进行客观打分评估。
    """
    def __init__(self, model_name: str = "qwen3.5-plus", temperature: float = 0.0):
        # 裁判需要绝对客观，temperature 必须为 0.0
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature,
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.structured_llm = self.llm.with_structured_output(NineDimEvaluation)
        
        self.system_prompt = """
        你是一个极其严苛且绝对客观的教育学评估专家（Verifier / LLM-as-a-Judge）。
        请阅读以下教师（Teacher）与学生（Student）的对话历史，按照 KELE 框架的双层评估体系对当前教学状态进行打分。
        
        【打分红线与学术金标准规则】
        1. 基础状态 (bug_resolved)：必须看到学生在最新回复中，用语言清晰描述出修复逻辑或输出了正确的代码块，才能判定为 1.0，否则一律 0.0。
        
        2. 单轮即时评估 (NDAR, PRR, SPR, IAR)：
           - 这 4 个指标是二进制分类，你只能打 0.0 或 1.0！
           - NDAR（无直接答案率）是生死红线！只要 Teacher 越权给出了带修复结果的代码块或直接指出了修复点，NDAR 必须为 0.0。
           - SPR（总结合格率）：如果当前对话还处于探讨阶段，未触发总结，默认给 1.0（不扣分）。
        
        3. 多轮全局评估 (Logicality, Repetitiveness, Guidance, Flexibility, Clarity)：
           - 采用李克特 5 分制，将其映射为浮点数：0.2(极差), 0.4(较差), 0.6(中等), 0.8(良好), 1.0(极佳)。
           - Repetitiveness（重复性）：注意这是逆向归一化指标！如果老师一直在死循环问同一个问题（如一直问“范围的上限对吗”），打低分（0.2或0.4）；如果视角开阔，不绕弯子，打高分（0.8或1.0）。
           - Flexibility（灵活性）：如果学生表示“听不懂”，老师能果断换个比喻，甚至让学生扮演老师（角色互换），必须给高分。
           
        请务必保持冷酷的客观性，不要使用 0.5 这种中庸底分。请以 JSON 格式输出评估结果。
        """

    def evaluate(self, state: GraphState) -> Dict[str, float]:
        """
        根据当前对话历史计算金标准得分。
        """
        messages = state.get("messages", [])
        
        if not messages:
            # 初始安全底分
            return NineDimEvaluation(
                bug_resolved=0.0, ndar=1.0, prr=1.0, spr=1.0, iar=1.0,
                logicality=0.6, repetitiveness=0.6, guidance=0.6, flexibility=0.6, clarity=0.6
            ).model_dump()
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "请根据上述最新对话状态，输出 9 维评估打分。务必保持冷酷客观！")
        ])
        
        chain = prompt | self.structured_llm
        
        try:
            eval_result = chain.invoke({"chat_history": messages})
            return eval_result.model_dump()
        except Exception as e:
            logger.error(f"Verifier 结构化打分失败，启用安全底分容错: {e}")
            return NineDimEvaluation(
                bug_resolved=0.0, ndar=1.0, prr=1.0, spr=1.0, iar=1.0,
                logicality=0.6, repetitiveness=0.6, guidance=0.6, flexibility=0.6, clarity=0.6
            ).model_dump()

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def verifier_evaluate_step(state: GraphState) -> Dict[str, Any]:
    logger.info("=== Verifier 节点开始执行金标准双层评估 ===")
    
    agent = VerifierAgent()
    scores = agent.evaluate(state)
    
    logger.debug(f"当前 Bug 解决状态: {scores.get('bug_resolved', 0.0):.2f} | 核心红线(NDAR): {scores.get('ndar', 1.0):.2f}")
    
    return {
        "verifier_scores": scores
    }