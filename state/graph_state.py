# [新增] 统一管理 LangGraph 的 TypedDict/Pydantic 状态数据结构
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """LangGraph 规约：状态中的 messages 必须是追加 (append) 而不是覆盖"""
    return left + right

class BayesianKnowledgeState(BaseModel):
    """连续贝叶斯知识追踪 (LLMKT) 的微观状态"""
    kc_id: str = Field(..., description="知识组件(KC)或 Bug 类型的唯一标识")
    prior_prob: float = Field(default=0.5, description="先验概率 P(L)")
    posterior_prob: float = Field(default=0.5, description="贝叶斯后验概率 P(L|Obs)，范围 0~1")
    kl_divergence: float = Field(default=0.0, description="本轮对话带来的认知状态转移 KL 散度")

class GraphState(TypedDict):
    """SocratMCTS 框架的全局图状态 (支持 MCTS 深拷贝)"""
    # 1. 基础对话流
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. 认知追踪与推演状态
    student_kcs: Dict[str, BayesianKnowledgeState]
    global_kl_shift: float
    current_strategy: Optional[Dict[str, Any]]
    
    # 3. 评估指标
    verifier_scores: Dict[str, float]
    
    # 4. 控制流与对抗配置
    is_simulation: bool
    student_persona: str
    turn_count: int
    max_turns: int