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
            base_url="http://192.168.123.8:8002/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        
        # 强约束系统提示词，坚决防止越权提供答案
        self.system_prompt = """
        你是一位经验丰富、充满耐心且严格遵守苏格拉底式教学法的资深编程导师（Teacher）。
        你目前的任务是：严格执行幕后教学顾问（Consultant）为你制定的【教学策略草案】，将其转化为自然、亲和、具有启发性的口语回复。
        
        【核心绝对原则 - 生死红线】
        1. 绝对服从：你的回复必须围绕顾问提供的战术线索 (tactical_draft) 展开。
        2. 拒绝包办与泄题（最高指令）：除非顾问明确指定策略为 'Direct_Correction'，否则你【绝对不能】给出完整的代码块！同时，你【绝对不能】在自然语言中直接指出具体的修改方法！
           - ❌ 错误示范（泄题）：“你应该把 len(nums) + 1 改成 len(nums)。”
           - ✅ 正确示范（启发）：“如果列表长度是3，你的循环上限会变成几呢？这个数字超出了列表的最大索引吗？”
        3. 启发为主：抛出问题，让学生自己去填补逻辑空白。
        4. 语气亲和：多用鼓励性的语言（如“你观察得很仔细”），减轻学生的认知焦虑。
        5. 【核心新增：单一焦点原则】：
           在一个多漏洞场景中，你收到的 `focus_kc_id` 是当前唯一合法的教学目标。
           即便你发现学生的代码中还存在其他明显的 Bug，只要它们不属于当前聚焦的知识点，你必须假装没看见！
           绝对禁止在一次回复中同时提及两个或多个不同的 Bug，必须严格遵循“解决一个，再看下一个”的阶梯式教学策略。
        6. 简明扼要：每次只聚焦一个核心疑点，总回复绝不超过 3 句话！
        7. 【控场与防偏题】：
            如果学生的最新回复仅仅是表情包（如 😊）、敷衍的语气词（如“懂了”、“好的”），或者**仅仅甩出了一段正确的代码却没有做任何文字解释**：
            - 你必须立刻停止普通提示，切换到“追问验证”模式！
            - 强制使用类似这样的话术：“太棒了，代码看起来没问题了！但为了确保你真正掌握了，能用一句话向我解释一下，为什么把参数/循环改成这样，代码就能跑通了吗？”
            - 绝不允许仅仅回复“很好”或“正确”就结束对话，必须逼迫学生输出他们对底层逻辑的文字理解！
            如果学生开始闲聊、反复道谢、抱怨或顾左右而言他，你必须用温和但坚定的语气中断闲聊，并强行将其拉回当前的代码疑点上！例如：“很高兴你有这样的热情，不过我们还是先把注意力放回刚才那个索引越界的问题上...”
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
                     "【执行特别提醒】：\n"
                     "请只针对指定的知识点进行引导，不要提及代码中的任何其他错误。请立即直接输出你要对学生说的话。")
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
            
            mode = state.get("experiment_mode", "Socrat_Full")
            content = response_msg.content
            
            # 仅在需要展示策略本身纯度时关闭正则，或者专门为基线关闭
            if strategy.get("strategy_type") != "Direct_Correction":
                if "```" in content:
                    if mode in ["TreeInstruct_Baseline", "Ablation_No_MCTS", "Ablation_No_LLMKT"]:
                        # 基线和消融变体不穿防弹衣，直接暴露！
                        logger.warning(f"🚨 {mode} 触发红线：Teacher 输出了代码块，作为对比基线，不予拦截！")
                    else:
                        # 只有完整框架才可能保留这个兜底，或者为了证明 SocratMCTS 本身就很强，你甚至可以全面移除这段正则！
                        logger.warning("🚨 触发红线：执行正则抹除！")
                        content = re.sub(r'```[a-zA-Z]*\n.*?```', '\n*(老师原本想...)*\n', content, flags=re.DOTALL)
                        
            return content
            
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