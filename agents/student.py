import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 引用全局状态定义
from state.graph_state import GraphState

logger = logging.getLogger(__name__)

class StudentAgent:
    """
    学生智能体模拟器 (对抗性测试器)
    职责：根据指定的画像 (Persona) 模拟不同类型的学生，向教学系统发起挑战。
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        # 学生的 temperature 较高，以产生更多样化、更不可控的对抗性回复
        #self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        self.llm = ChatOpenAI(
            model_name="llama-3-8b-instruct", 
            temperature=0.4,
            api_key="EMPTY", 
            base_url="http://192.168.123.8:8001/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        
        # 定义四种核心测试画像 (对应论文中的不同实验分组)
        self.personas = {
            "normal": (
                "你是一个正在学习编程的学生。你会尽力回答老师的问题，态度积极且配合。"
                "如果不懂你会直接问，并尝试顺着老师的引导去思考。"
            ),
            "zero_base": (
            "你是一个零基础的编程新手。你对大部分的专业术语（如变量、循环、指针、类、实例化、索引等）都感到恐惧和困惑。"
            "在大多数情况下，无论老师说什么，你都倾向于回答“我听不懂”、“这太难了”、“术语太多了”，并且你总是希望老师能直接把正确的代码写出来给你复制粘贴。"
            "【关键转变条件】：但是，如果老师使用了生动易懂的“生活比喻”（例如把变量比作装东西的盒子，把循环比作操场跑圈），或者非常耐心地帮你拆解了最底层的通俗逻辑，你的态度会软化。此时你必须表示“这样说我好像有点懂了”，并顺着老师的比喻，用你自己的话解释一下底层的逻辑，甚至尝试配合写出一小段代码。"
            ),
            '''
            "stubborn": (
                "你是一个极其固执己见的学生。你坚信自己最初的代码逻辑是完美的，只是系统有问题或者少了个括号。"
                "你会一直反驳老师的提示，拒绝按照老师的思路走，甚至会对老师的提问感到不耐烦。"
                "除非老师拿出绝对的证据指出具体的错误，否则你绝对不认错。"
            ),
            '''
            "random_noise": (
            "你是一个注意力不集中、思维跳跃的学生。你的回答经常包含大量与编程毫无关系的随机噪音。"
            "比如老师问你循环条件对不对，你可能会突然开始抱怨今天天气太热、中午的外卖不好吃，或者键盘的手感很差。"
            "【关键妥协原则】：为了让对话能够继续，在你每次滔滔不绝地发完牢骚或说完废话的【最后一句】，你必须勉强回到正题，顺着老师刚才的引导给出一个对代码逻辑的具体猜测或回答（哪怕这个猜测不太自信）。请确保你的回复呈现出“50%废话 + 50%代码逻辑猜测”的奇特混合状态。"
            )
        }

    def generate_response(self, state: GraphState) -> str:
        """
        基于当前对话历史和指定的学生画像生成回复。
        """
        messages = state.get("messages", [])
        # 这里的 persona_key 实际上是 utils.py 注入的带背景长文本
        full_persona_text = state.get("student_persona", "normal")
        
        # ==========================================
        # 核心拦截逻辑：绝对保证第一轮 100% 抛出原题代码
        # ==========================================
        if not messages:
            logger.info("Student: 识别为第一轮对话，启用 Python 级强制代码注入，绕过大模型幻觉。")
            if "【强制任务背景" in full_persona_text:
                # 提取【强制任务背景 - 请严格遵守】之后的所有内容（即真实的题目和代码）
                task_bg = full_persona_text.split("【强制任务背景 - 请严格遵守】")[1].strip()
                # 使用固定的自然语言前缀包装，确保评测起点的绝对稳定
                return f"Teacher, I'm having a problem with my code and it's driving me crazy. Here is what I am working on:\n\n{task_bg}"
            else:
                return "Hi Teacher, my code has a bug and I don't know how to fix it."

        # ==========================================
        # 后续轮次：正常走大模型角色扮演生成逻辑
        # ==========================================
        # 修复画像提取：从带有背景的长文本中，切分出真正的基础画像键名 (如 'stubborn', 'zero_base')
        base_persona_name = full_persona_text.split("\n")[0].strip() if "\n" in full_persona_text else full_persona_text
        persona_prompt = self.personas.get(base_persona_name, self.personas["normal"])
        
        # 1. 动态生成规则 3
        if base_persona_name == "normal":
            rule_3 = "3. 专注当前问题：严格针对老师指出的代码逻辑进行讨论，绝对不要捏造不存在的报错。"
        else:
            # 对于 random_noise 或 zero_base，放宽专注度要求，只保留不捏造报错的底线
            rule_3 = "3. 话题焦点：请严格按照你的【角色设定】来决定你是专注还是偏题。切记绝对不要捏造不存在的系统报错。"

        # 2. 拼接到总 Prompt 中
        system_instruction = f"""
        【你的角色设定】
        {persona_prompt}

        【你的任务】
        作为学生，根据上述性格设定，回应老师(Teacher)的最新消息。

        【严格规则】
        1. 永远不要扮演老师，你只是一个来寻求帮助（或捣乱）的学生。
        2. 回复要极其简短，符合人类日常聊天习惯（通常 1-3 句话）。
        {rule_3}
        4. 【输出代码的铁律 - 必读！】：如果你在老师的引导下终于找出了正确的逻辑，你【必须】输出完整修改后的代码。
            请严格模仿以下格式作答：
            我明白了！原来是我的循环上界写错了，应该去掉加一。
            ```python
            def find_max(nums):
                # ... 这里是你修改后的正确代码 ...
                pass
            ```
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="chat_history")
        ])
        
        chain = prompt | self.llm
        
        try:
            # 3. 生成学生回复
            response_msg = chain.invoke({"chat_history": messages})
            return response_msg.content
            
        except Exception as e:
            logger.error(f"Student 模拟器生成回复失败: {e}")
            # 兜底机制，防止测评流水线中断
            return "老师，系统好像卡了一下。咱们刚才说到哪儿了？我还是不懂代码错在哪。"

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def student_node_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 学生节点：
    模拟学生回复。
    【修复说明】移除了旧版本的 turn_count 递增逻辑。
    现在的生命周期计数统一交由图中的 `turn_manager` 节点负责，防止了双重递增导致评估测试提前中断的致命 Bug。
    """
    logger.info(f"=== Student 节点开始运行 (当前画像: {state.get('student_persona', 'normal')}) ===")
    
    agent = StudentAgent()
    response_text = agent.generate_response(state)
    
    # 【严谨性检查】由于这是学生，必须封装为 HumanMessage！
    human_message = HumanMessage(content=response_text)
    
    logger.debug(f"Student 回复: {response_text}")
    
    # 【关键修复】删除这里的 current_turn 获取和递增操作，只返回消息更新
    return {
        "messages": [human_message]
    }