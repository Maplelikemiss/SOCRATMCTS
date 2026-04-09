# [重构] 连续知识追踪模块：结合 Reducer 优化防污染逻辑与局部增量状态更新
import math
import logging
from typing import Dict, Any, Tuple, List
from langchain_core.messages import BaseMessage

# 引用已设计的状态结构
from state.graph_state import GraphState, BayesianKnowledgeState

logger = logging.getLogger(__name__)

class BayesianKnowledgeTracer:
    """
    连续贝叶斯知识追踪引擎 (LLMKT)
    采用软证据更新(Soft Evidence Update)结合 BKT 标准转移方程。
    """
    def __init__(self, slip_prob: float = 0.1, guess_prob: float = 0.2, transition_prob: float = 0.15):
        # P(S): 学生已掌握但因粗心失误的概率
        self.P_S = slip_prob
        # P(G): 学生未掌握但侥幸猜对/蒙对的概率
        self.P_G = guess_prob
        # P(T): 知识点在一次交互中被学习并掌握的传递概率
        self.P_T = transition_prob

    def calculate_kl_divergence(self, prior: float, posterior: float) -> float:
        """
        计算认知状态转移的 KL 散度 D_KL(Posterior || Prior)
        用于量化 MCTS 推演过程中的“认知信息增益”
        """
        epsilon = 1e-9
        prior = max(epsilon, min(1.0 - epsilon, prior))
        posterior = max(epsilon, min(1.0 - epsilon, posterior))
        
        kl = posterior * math.log(posterior / prior) + \
             (1 - posterior) * math.log((1 - posterior) / (1 - prior))
        return kl

    def update_kc_state(self, prior_prob: float, observation_score: float) -> Tuple[float, float]:
        """
        基于连续的观测得分更新后验概率。
        
        :param prior_prob: P(L_{t-1})
        :param observation_score: 介于 0.0 到 1.0 之间的观测得分 (通过 LLM 对话提取)
        :return: (posterior_prob, kl_divergence)
        """
        # 1. 计算观测边缘概率 P(Correct) 和 P(Incorrect)
        p_correct = prior_prob * (1 - self.P_S) + (1 - prior_prob) * self.P_G
        p_incorrect = 1.0 - p_correct

        # 2. 纯证据条件下的后验概率 (标准贝叶斯定理)
        p_l_given_correct = (prior_prob * (1 - self.P_S)) / max(p_correct, 1e-9)
        p_l_given_incorrect = (prior_prob * self.P_S) / max(p_incorrect, 1e-9)
        
        # 3. 软证据更新 (Soft Update) - 处理 LLM 输出的连续置信度
        p_l_given_obs = observation_score * p_l_given_correct + (1.0 - observation_score) * p_l_given_incorrect
        
        # 4. 考虑学习传递律 P(T)
        posterior_prob = p_l_given_obs + (1.0 - p_l_given_obs) * self.P_T
        posterior_prob = max(0.01, min(0.99, posterior_prob)) # 数值截断保护
        
        # 5. 计算增益
        kl_div = self.calculate_kl_divergence(prior_prob, posterior_prob)
        
        return posterior_prob, kl_div

def _extract_llm_observation(messages: List[BaseMessage], kc_id: str) -> float:
    """
    [内部辅助函数] LLMKT 独立特征提取：
    实际工程中此处应调用轻量级模型 (如 Llama-3 8B) 获取学生对该 kc_id 掌握的 Soft Score。
    """
    if not messages:
        return 0.5
    
    last_message = messages[-1].content.lower()
    
    # 启发式模拟：根据学生语言中的关键词映射概率
    if "不知道" in last_message or "不懂" in last_message or "报错" in last_message:
        return 0.1
    elif "原来如此" in last_message or "明白了" in last_message or "修复" in last_message:
        return 0.85
    
    return 0.5  # 默认中性观测，表示当前对话未涉及该知识点或信息模糊

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def llmkt_bayesian_update_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 认知追踪节点：基于最新对话，计算连续后验概率与 KL 散度。
    【核心优化】完美配合 GraphState 中的 merge_kcs 归约器，仅返回发生实质性偏移的 KC。
    """
    tracer = BayesianKnowledgeTracer()
    student_kcs = state.get("student_kcs", {})
    messages = state.get("messages", [])
    
    # 局部增量更新字典，避免全量拷贝造成的内存浪费
    partial_updated_kcs = {}
    total_kl_shift = 0.0

    # 冷启动处理：如果当前没有任何 KC，动态注入一个基础 KC
    if not student_kcs:
        default_kc_id = "target_bug_understanding"
        # 初始化新的 Pydantic 状态对象
        new_kc = BayesianKnowledgeState(kc_id=default_kc_id, prior_prob=0.5, posterior_prob=0.5)
        partial_updated_kcs[default_kc_id] = new_kc
        # 为了让下方的循环在冷启动时也能执行评估，临时引用一下
        student_kcs = partial_updated_kcs 
        logger.info(f"LLMKT: 初始化默认知识组件 {default_kc_id}")

    # 遍历更新学生的各项知识组件
    for kc_id, kc_state in student_kcs.items():
        prior = kc_state.posterior_prob 
        obs_score = _extract_llm_observation(messages, kc_id)
        
        # 【关键优化 1】过滤噪声：如果观测得分在中立区 (0.5) 徘徊，视为无效观测，跳过计算
        if abs(obs_score - 0.5) < 0.05:
            continue
            
        new_posterior, kl_div = tracer.update_kc_state(prior, obs_score)
        
        # 【关键优化 2】KL 阈值截断：仅当认知发生了实质性转移时，才生成新的深拷贝对象
        if kl_div > 1e-4:
            new_kc_state = kc_state.model_copy(update={
                "prior_prob": prior,
                "posterior_prob": new_posterior,
                "kl_divergence": kl_div
            })
            partial_updated_kcs[kc_id] = new_kc_state
            total_kl_shift += kl_div
        
    if total_kl_shift > 0:
        logger.debug(f"LLMKT 局部状态更新完成, 整体认知增益(KL): {total_kl_shift:.4f}")
    
    # 返回 diff 字典：LangGraph 底层的 merge_kcs 会将 partial_updated_kcs 安全地合并回主树
    return {
        "student_kcs": partial_updated_kcs,
        "global_kl_shift": total_kl_shift
    }