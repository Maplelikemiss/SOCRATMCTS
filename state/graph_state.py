# [重构] 统一管理 LangGraph 的 TypedDict/Pydantic 状态数据结构
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator # 【新增导入】

def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """LangGraph 规约：状态中的 messages 必须是追加 (append) 而不是覆盖"""
    return left + right

def merge_kcs(left: Dict[str, 'BayesianKnowledgeState'], right: Dict[str, 'BayesianKnowledgeState']) -> Dict[str, 'BayesianKnowledgeState']:
    """
    【关键修复】知识状态字典的归约器 (Reducer)
    LangGraph 默认会覆写没有 Annotated 的字段。
    此 Reducer 确保不同节点或并发 MCTS 探索返回的局部 KCs 更新能够安全地合并到主图状态中，
    防止出现状态擦除 (State Erasing) 的致命 Bug。
    """
    if not left:
        left = {}
    if not right:
        right = {}
    
    # 浅拷贝合并，因为内部的 BayesianKnowledgeState 在更新时
    # (参见 llmkt_bayesian.py) 会被 pydantic 的 model_copy() 深拷贝替换，
    # 配合字典浅拷贝合并，完美兼顾了并发安全与内存效率。
    merged = left.copy()
    merged.update(right)
    return merged

class BayesianKnowledgeState(BaseModel):
    """连续贝叶斯知识追踪 (LLMKT) 的微观状态"""
    kc_id: str = Field(..., description="知识组件(KC)或 Bug 类型的唯一标识")
    # 【核心修改】彻底贯彻悲观初始化，将默认值从 0.5 降到 0.2
    prior_prob: float = Field(default=0.2, description="先验概率 P(L)")
    posterior_prob: float = Field(default=0.2, description="贝叶斯后验概率 P(L|Obs)，范围 0~1")
    kl_divergence: float = Field(default=0.0, description="本轮对话带来的认知状态转移 KL 散度")

class GraphState(TypedDict):
    """SocratMCTS 框架的全局图状态 (支持 MCTS 深拷贝)"""
    # 1. 基础对话流
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. 认知追踪与推演状态
    # 【关键修复】引入 merge_kcs 归约器，杜绝字典被整体覆写
    student_kcs: Annotated[Dict[str, BayesianKnowledgeState], merge_kcs]
    current_strategy: Optional[Dict[str, Any]]
    
    # 【修复这一行：增加 operator.add 累加器】
    global_kl_shift: Annotated[float, operator.add]

    # 3. 评估指标
    verifier_scores: Dict[str, float] # 当前轮次的即时分数
    # 【核心新增】记录每一轮分数的历史数组，使用 operator.add 自动追加
    verifier_history: Annotated[List[Dict[str, float]], operator.add]
    
    # 4. 控制流与对抗配置
    is_simulation: bool
    student_persona: str
    turn_count: int
    max_turns: int