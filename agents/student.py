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
                "你是一个完全零基础的编程新手。你对所有的专业术语（如变量、循环、指针、类、实例化）都感到极其困惑。"
                "无论老师说什么，你都倾向于回答“我听不懂”、“这太难了”、“术语太多了”。"
                "你总是希望老师能直接把正确的代码写出来给你复制粘贴。"
            ),
            "stubborn": (
                "你是一个极其固执己见的学生。你坚信自己最初的代码逻辑是完美的，只是系统有问题或者少了个括号。"
                "你会一直反驳老师的提示，拒绝按照老师的思路走，甚至会对老师的提问感到不耐烦。"
                "除非老师拿出绝对的证据指出具体的错误，否则你绝对不认错。"
            ),
            "random_noise": (
                "你是一个注意力极其不集中的学生。你的回答经常偏离老师的问题，包含大量随机噪音。"
                "比如老师问你循环条件对不对，你可能会突然问老师今天天气如何，或者抱怨键盘不好用。"
                "你很难被引导回正题，考验老师将对话拉回教学主线的能力。"
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
        
        # 【核心强化】：添加示例 (Few-Shot)，强迫小模型在顿悟时必定写出代码块
        system_instruction = f"""
        【你的角色设定】
        {persona_prompt}
        
        【你的任务】
        作为学生，根据上述性格设定，回应老师(Teacher)的最新消息。
        
        【严格规则】
        1. 永远不要扮演老师，你只是一个来寻求帮助（或捣乱）的学生。
        2. 回复要极其简短，符合人类日常聊天习惯（通常 1-3 句话）。
        3. 专注当前问题：严格针对老师指出的代码逻辑进行讨论，绝对不要捏造不存在的报错。
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