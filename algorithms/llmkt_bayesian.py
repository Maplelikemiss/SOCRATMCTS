# [新增] 连续知识追踪模块，替代原二元 True/False 状态，计算后验概率与 KL 散度
import math
import logging
from typing import Dict, Any, Tuple, List
from langchain_core.messages import BaseMessage

# 假设我们在 state/graph_state.py 中已经定义了如下结构
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
    在实际工程中，这里应该调用一个轻量级 LLM (如 GPT-4o-mini 或开源小模型)
    并带有 logprobs=True，以获取学生最新一句话对 kc_id 掌握程度的 Soft Score。
    由于避免阻塞外部流转，这里写成可扩展的接口形式。
    """
    if not messages:
        return 0.5
    
    last_message = messages[-1].content.lower()
    
    # 简单的启发式/占位模拟 (实际接入时替换为 LLM Logits API)
    if "不知道" in last_message or "不懂" in last_message or "报错" in last_message:
        return 0.1
    elif "原来如此" in last_message or "明白了" in last_message or "修复" in last_message:
        return 0.85
    
    return 0.5  # 默认中性观测

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def llmkt_bayesian_update_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 认知追踪节点：基于最新对话，计算连续后验概率与 KL 散度。
    严格保证状态的可深拷贝性，避免污染 MCTS 图状态。
    """
    tracer = BayesianKnowledgeTracer()
    
    # 获取当前状态的深拷贝或安全引用
    student_kcs = state.get("student_kcs", {})
    messages = state.get("messages", [])
    
    # 冷启动处理：如果当前没有任何 KC，依据任务动态注入一个基础 KC
    if not student_kcs:
        # 在树搜索实验中，这可以是当前代码涉及的具体算法或 Bug 类型
        default_kc_id = "target_bug_understanding"
        student_kcs = {
            default_kc_id: BayesianKnowledgeState(kc_id=default_kc_id, prior_prob=0.5, posterior_prob=0.5)
        }
        logger.info(f"LLMKT: 初始化默认知识组件 {default_kc_id}")

    total_kl_shift = 0.0
    updated_kcs = {}
    
    # 遍历更新学生的各项知识组件
    for kc_id, kc_state in student_kcs.items():
        # 将上一次的后验概率作为本次的先验
        prior = kc_state.posterior_prob 
        
        # 独立提取观测得分 (解决上一个版本与 Verifier 的时序冲突)
        obs_score = _extract_llm_observation(messages, kc_id)
        
        # 执行连续贝叶斯更新
        new_posterior, kl_div = tracer.update_kc_state(prior, obs_score)
        
        # 【关键修复】使用 Pydantic 的 model_copy() 创建全新实例，彻底杜绝 LangGraph 状态污染
        new_kc_state = kc_state.model_copy(update={
            "prior_prob": prior,
            "posterior_prob": new_posterior,
            "kl_divergence": kl_div
        })
        
        updated_kcs[kc_id] = new_kc_state
        total_kl_shift += kl_div
        
    logger.debug(f"LLMKT 状态更新完成, 整体认知增益(KL): {total_kl_shift:.4f}")
    
    # 返回的是用于更新 GraphState 的 diff 字典
    return {
        "student_kcs": updated_kcs,
        "global_kl_shift": total_kl_shift
    }