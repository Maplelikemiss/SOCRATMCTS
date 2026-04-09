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
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        '''
        self.llm = ChatOpenAI(
            model_name="qwen-2.5-72b-instruct", 
            temperature=0.4,
            api_key="你的开源模型API_KEY或者随便填", 
            base_url="http://localhost:8000/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )
        '''
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
        # 1. 提取画像配置 (默认是正常学生)
        persona_key = state.get("student_persona", "normal")
        persona_prompt = self.personas.get(persona_key, self.personas["normal"])
        messages = state.get("messages", [])

        # 2. 动态构建系统提示词
        system_instruction = f"""
        【你的角色设定】
        {persona_prompt}
        
        【你的任务】
        作为学生，根据上述性格设定，回应老师(Teacher)的最新消息。
        
        【严格规则】
        1. 永远不要扮演老师，你只是一个来寻求帮助（或捣乱）的学生。
        2. 回复要极其简短，符合人类日常聊天习惯（通常 1-3 句话）。
        3. 如果这是第一轮对话（没有历史消息），请主动给出一个有 Bug 的代码片段（例如 Python 列表越界或死循环），并向老师求助。
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
    模拟学生回复，并维护全局的轮次计数器 (turn_count)。
    """
    logger.info(f"=== Student 节点开始运行 (当前画像: {state.get('student_persona', 'normal')}) ===")
    
    agent = StudentAgent()
    response_text = agent.generate_response(state)
    
    # 【严谨性检查】由于这是学生，必须封装为 HumanMessage！
    human_message = HumanMessage(content=response_text)
    
    logger.debug(f"Student 回复: {response_text}")
    
    # 更新轮次计数器 (防死循环机制的核心)
    current_turn = state.get("turn_count", 0)
    
    # 增量更新返回
    return {
        "messages": [human_message],
        "turn_count": current_turn + 1
    }