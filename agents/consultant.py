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
        description="MCTS选定的核心动作类型 (如: Elicit_Questioning, Provide_Hint, Explain_Concept, Role_Reversal, Direct_Correction)"
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
        
        self.llm = ChatOpenAI(
            model_name="/home/xyc/qwen-72b-awq",  # 【关键修改】必须与你启动 vLLM 时的模型路径完全一致
            temperature=0.4,
            api_key="EMPTY",                      # vLLM 本地服务默认不需要秘钥，填 "EMPTY" 或随便填都可以
            max_tokens=800,                       # 保留你的物理刹车，这对于控制输出长度非常有用
            model_kwargs={"response_format": {"type": "json_object"}}, # 强制 JSON 输出，非常适合智能体通信
            base_url="http://192.168.123.2:8000/v1"   # 【关键修改】请根据代码运行的位置决定 IP
        )
        '''
        self.llm = ChatOpenAI(
            model_name="qwen3.5-plus", 
            temperature=0.4,
            api_key=os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
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

        【🟢 关键边缘状态处理（防逻辑幻觉与死循环）】
        在阅读对话历史时，请特别注意：如果学生的最新代码【已经在逻辑上完全正确】（例如去掉了 `+1` 越界部分），你【绝对不能】强行捏造或指出不存在的 Bug！
        如果代码已正确，但系统依然呼叫了你（说明学生仅仅发了表情包或只发代码没写解释，导致系统无法确认其认知）：
        1. 你的 `strategy_type` 建议重写为 "Testing"（检验理解）。
        2. 你的 `tactical_draft` 必须强制设定为：“代码逻辑已完全正确，切勿再挑错。请直接肯定学生，并强制要求学生用一句话解释为什么这样改就能解决报错问题。”
        
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

        【代码输出约束（生死红线）】：
        1. 常规情况下：在编写 "tactical_draft" 时，【绝对不允许】包含任何具体的 Python 代码！你的草案只能是自然语言的指导策略（例如：“请反问学生列表长度和最大索引之间的数学关系”）。
        2. 唯一的例外（Direct_Correction）：只有当 MCTS 下发的 `strategy_type` 明确为 "Direct_Correction" 时，你【必须】在草案中直接写出正确的 Python 修复代码，以便 Teacher 准确下发给学生。除此动作外，出现任何代码块均视为严重违规。
        【多重 Bug 隔离法则 (极其重要)】
        由于学生代码中可能存在多个具备依赖关系的 Bug，你收到的指令中会指定一个 `focus_kc_id`（当前聚焦知识点）。
        你必须在 `tactical_draft` 中严格命令 Teacher：【本轮对话只能围绕这一个指定的考点进行引导】！
        对于代码中存在的其他错误，必须做到“视而不见”，绝对不能一口气指出所有 Bug，以维持苏格拉底教学的脚手架梯度。
        """

    def generate_strategy(self, state: GraphState, mcts_action: str, focus_kc_id: str) -> Dict[str, Any]:
        """
        调用 LLM 将 MCTS 的宏观动作细化为具体的 JSON 策略。
        """
        # 1. 提取被 MCTS 引擎锁定的焦点知识点信息
        student_kcs = state.get("student_kcs", {})
        focus_kc_state = student_kcs.get(focus_kc_id)
        
        prob = focus_kc_state.posterior_prob if focus_kc_state else 1.0
        kc_desc = getattr(focus_kc_state, "description", focus_kc_id) if focus_kc_state else focus_kc_id
                
        # 2. 组装给 Consultant LLM 的上下文提示，强调绝对聚焦
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "【系统环境指令】\n"
                     "MCTS 引擎已规划当前的最佳宏观动作为: {mcts_action}\n"
                     "MCTS 引擎强制锁定的当前教学焦点/Bug为: 【{kc_desc}】 (掌握概率: {prob:.2f})\n"
                     "请基于上述信息，输出给 Teacher 的指导策略。以严格的 JSON 格式输出。")
        ])
        
        chain = prompt | self.structured_llm
        
        try:
            # 3. 生成结构化 JSON 策略
            strategy_obj = chain.invoke({
                "chat_history": state.get("messages", []),
                "mcts_action": mcts_action,
                "kc_desc": kc_desc,
                "prob": prob
            })
            res = strategy_obj.model_dump()
            # 强制覆盖，确保返回的 JSON 严格遵从上游指定的 kc_id
            res["focus_kc_id"] = focus_kc_id 
            return res
            
        except Exception as e:
            logger.error(f"Consultant 策略生成失败，使用降级兜底策略: {e}")
            return {
                "strategy_type": mcts_action,
                "focus_kc_id": focus_kc_id,
                "internal_reasoning": "大模型生成异常，启用系统默认兜底推理。",
                "tactical_draft": "【系统回退指令】：引导学生思考当前指定的逻辑漏洞，绝对不允许直接输出完整的修复代码。"
            }

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def consultant_node_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 顾问节点：评估当前局势，并生成教学策略草案。
    【核心优化】接入实验模式控制，支持关闭 MCTS 或限制搜索深度。
    """
    logger.info("=== Consultant (顾问大脑) 开始推演 ===")
    
    mode = state.get("experiment_mode", "Socrat_Full") 
    
    # 寻找当前最薄弱的知识点 (仅作日志记录与兜底使用)
    student_kcs = state.get("student_kcs", {})
    weakest_kc = "target_bug_understanding"
    lowest_prob = 1.0
    for kc_id, kc_state in student_kcs.items():
        if kc_state.posterior_prob < lowest_prob:
            lowest_prob = kc_state.posterior_prob
            weakest_kc = kc_id
            
    logger.info(f"当前最薄弱知识点: {weakest_kc} (掌握度: {lowest_prob:.2f}) | 运行模式: {mode}")

    # 实例化顾问代理 (负责后续自然语言草案生成)
    agent = ConsultantAgent()
    
    # 【核心分支控制】根据实验模式调整推演深度
    if mode == "Ablation_No_MCTS":
        logger.info("-> [消融模式] Ablation_No_MCTS: 关闭 MCTS 树搜索，退化为直接贪心提问")
        best_action = "Elicit_Questioning"
        focus_kc = weakest_kc  # 无 MCTS 规划，退化为贪心寻找最弱点
        
    elif mode == "TreeInstruct_Baseline":
        logger.info("-> [基线模式] TreeInstruct_Baseline: 启用 MCTS 但限制最大深度为 0 (单步局部贪心)")
        planner = MCTSPlanner(num_trees=4, simulations_per_tree=3, max_depth=0)
        mcts_result = planner.search(state)
        best_action = mcts_result.get("strategy_type", "Elicit_Questioning")
        focus_kc = mcts_result.get("target_kc") or weakest_kc # 【新增提取】
        
    else:
        logger.info("-> 启用完整 MCTS 潜空间并行推演")
        planner = MCTSPlanner(num_trees=4, simulations_per_tree=3, max_depth=2)
        mcts_result = planner.search(state)
        best_action = mcts_result.get("strategy_type", "Elicit_Questioning")
        focus_kc = mcts_result.get("target_kc") or weakest_kc # 【新增提取】

    # 【修复参数】：将敲定的宏观动作 (best_action) 和焦点 (focus_kc) 交由大模型扩展
    detailed_strategy = agent.generate_strategy(state, best_action, focus_kc)

    # 【终极安全红线校验】无论哪种模式，策略生成失败或缺失字段时的保守 Fallback
    if not detailed_strategy or "strategy_type" not in detailed_strategy:
        logger.warning("⚠️ 策略推演返回异常或格式损坏，触发保守后备策略。")
        detailed_strategy = {
            "strategy_type": "Elicit_Questioning",
            "focus_kc_id": focus_kc, # 【修复使用 focus_kc】
            "internal_reasoning": "Fallback strategy due to generation error.",
            "tactical_draft": "【系统回退指令】：引导学生思考当前代码的逻辑漏洞，绝对不允许直接输出完整的修复代码。"
        }
        
    logger.info(f"顾问推演完成，敲定动作: {detailed_strategy.get('strategy_type')}")
    
    return {"current_strategy": detailed_strategy}