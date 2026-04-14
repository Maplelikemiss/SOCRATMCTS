# [重构] 拆分自原 instructor.py，负责接收 Consultant 策略并转化为自然语言教学
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

# 引用全局状态定义
from state.graph_state import GraphState

logger = logging.getLogger(__name__)

class TeacherAgent:
    """
    教师智能体 (嘴巴)
    职责：严格执行 Consultant 的 JSON 策略草案，将其转化为自然语言与学生交互。
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.4):
        # Teacher 需要更好的语言表现力，temperature 可以稍微调高一点，增加亲和力
        #self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        self.llm = ChatOpenAI(
            model_name="socrat-teacher-glm4", 
            temperature=0.4,
            api_key="EMPTY", 
            base_url="http://127.0.0.1:8000/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        
        # 强约束系统提示词，坚决防止越权提供答案
        self.system_prompt = """
        你是一位经验丰富、充满耐心的资深编程导师（Teacher）。
        你目前的任务是：严格执行幕后教学顾问（Consultant）为你制定的【教学策略草案】，将其转化为自然、亲和、具有启发性的口语回复。
        
        【核心绝对原则】
        1. 绝对服从：你的回复必须围绕顾问提供的战术线索 (tactical_draft) 展开。
        2. 拒绝包办：除非顾问明确指定策略为 'Direct_Correction'，否则你【绝对不能】直接给出最终的代码答案或完整的修复后代码块！
        3. 启发为主：采用苏格拉底式提问（Socratic Questioning），抛出问题，引导学生自己思考。
        4. 语气亲和：多用鼓励性的语言（如“你观察得很仔细”、“方向是对的”），减轻学生的认知焦虑。
        5. 简明扼要：每次只聚焦一个核心疑点，不要长篇大论。
        """

    def generate_response(self, state: GraphState) -> str:
        """
        基于当前对话历史和 Consultant 的策略生成最终回复。
        """
        # 1. 安全获取策略 (如果发生异常流转，提供兜底策略)
        strategy = state.get("current_strategy")
        if not strategy:
            logger.warning("Teacher 未能在状态中找到 current_strategy，启动防御性兜底。")
            strategy = {
                "strategy_type": "Elicit_Questioning",
                "focus_kc_id": "unknown",
                "internal_reasoning": "状态缺失兜底",
                "tactical_draft": "温和地询问学生目前对这段代码逻辑的具体困惑在哪里。"
            }
            
        # 2. 组装下发给 LLM 的指令
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "【顾问(Consultant) 下发的内部指令】\n"
                     "核心动作类型: {strategy_type}\n"
                     "关注的知识点: {focus_kc_id}\n"
                     "顾问推演逻辑: {reasoning}\n"
                     "====================\n"
                     "【你的执行草案 (必须遵守)】\n"
                     "{tactical_draft}\n"
                     "====================\n"
                     "请立即根据以上草案，直接输出你要对学生说的话（无需包含任何内心独白或前缀，直接输出对话内容）。")
        ])
        
        chain = prompt | self.llm
        
        try:
            # 3. 触发 LLM 生成回复
            response_msg = chain.invoke({
                "chat_history": state.get("messages", []),
                "strategy_type": strategy.get("strategy_type"),
                "focus_kc_id": strategy.get("focus_kc_id"),
                "reasoning": strategy.get("internal_reasoning"),
                "tactical_draft": strategy.get("tactical_draft")
            })
            return response_msg.content
            
        except Exception as e:
            logger.error(f"Teacher 文本生成失败: {e}")
            return "我明白你的困惑了。咱们一步步来，你能先跟我说说你是怎么理解当前这段代码的逻辑的吗？"

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def teacher_node_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 教师节点：
    读取状态中的 current_strategy，生成自然语言，并追加到消息历史中。
    """
    logger.info("=== Teacher 节点开始生成自然语言回复 ===")
    
    agent = TeacherAgent()
    response_text = agent.generate_response(state)
    
    # 封装为 AIMessage
    ai_message = AIMessage(content=response_text)
    
    logger.debug(f"Teacher 回复: {response_text}")
    
    # LangGraph 中对 messages 字段使用的是 add_messages (append 模式)
    # 所以我们只需返回增量的列表，图状态会自动拼接
    return {
        "messages": [ai_message]
    }