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
                "在大多数情况下，无论老师说什么，只要他还在干巴巴地谈论代码语法，你都倾向于回答“我听不懂”、“这太难了”、“术语太多了”，并且你总是希望老师能直接把正确的代码写出来给你复制粘贴。"
                "【关键转变条件 - 必须严格遵守】：除非老师在最新回复中明确使用了生动易懂的“生活比喻”（例如把变量比作装东西的盒子，把循环比作操场跑圈），或者主动降维使用了“角色互换”（让你来当小老师），你的态度才会软化。只有当老师满足上述条件之一时，你才能表示“这样说我好像有点懂了”，并尝试配合思考。否则，只要老师还是在直接问代码逻辑，请继续保持困惑、拒绝思考并抱怨！"
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
                task_bg = full_persona_text.split("【强制任务背景 - 请严格遵守】")[1].strip()
                # 【修改这里】：将英文开场白改为中文
                return f"老师，我的代码一直报错，快把我逼疯了。这是我目前的题目和代码：\n\n{task_bg}"
            else:
                return "老师你好，我的代码有Bug，我不知道怎么修。"

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
            请严格按照以下结构作答，但【警告：必须用你自己的话解释当前这道题的具体错误，绝对禁止抄袭下方示例中的文字】：
            
            [用一句话解释你刚弄懂的错误原因，例如：我明白了！原来是我把 xx 写成了 xx...]
            ```python
            # ... 这里是你修改后的正确代码 ...
            ```
        5. 严禁只回复表情。
        6. 你必须且只能使用**简体中文**进行回复！绝对禁止在日常交流中使用英文（除了不可避免的 Python 代码和变量名）。
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