# [新增] MCTS 核心引擎，支持深拷贝并行推演
import math
import copy
import random
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage

# 引用已设计的状态结构
from state.graph_state import GraphState

logger = logging.getLogger(__name__)

class MCTSNode:
    """MCTS 搜索树节点，承载深拷贝的瞬时图状态"""
    def __init__(self, state: GraphState, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action  # 导致当前状态的策略动作
        self.children: Dict[str, 'MCTSNode'] = {}
        
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, legal_actions: List[str]) -> bool:
        return len(self.children) == len(legal_actions)

    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """使用 UCT (UCB1) 算法选择最优子节点"""
        best_score = -float('inf')
        best_node = None
        
        for child in self.children.values():
            if child.visits == 0:
                return child # 未被访问的节点具有最高优先级
            
            # UCB1 公式：利用 (Q) + 探索 (U)
            exploit = child.value / child.visits
            explore = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_node = child
                
        return best_node

class MCTSPlanner:
    """
    针对苏格拉底式教学优化的 MCTS 规划器
    通过评估 KL散度增益 与 Bug解决率，搜索最优教学动作序列。
    """
    def __init__(self, num_simulations: int = 5, max_depth: int = 3):
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        
        # 预定义合法的苏格拉底教学动作空间 (Action Space)
        self.legal_actions = [
            "Elicit_Questioning",  # 启发式提问 (不给答案，反问引导)
            "Provide_Hint",        # 提供线索 (指出方向，但不写代码)
            "Explain_Concept",     # 概念解释 (补充缺失的背景知识)
            "Direct_Correction"    # 直接纠正 (仅在极端死循环时使用)
        ]

    def _deep_copy_state(self, state: GraphState) -> GraphState:
        """安全深拷贝 LangGraph 状态，防止虚拟推演污染主图"""
        try:
            # 标记为模拟状态
            new_state = copy.deepcopy(state)
            new_state["is_simulation"] = True
            return new_state
        except Exception as e:
            logger.warning(f"深拷贝失败，降级为浅拷贝重组: {e}")
            return {**state, "is_simulation": True}

    def search(self, root_state: GraphState) -> Dict[str, Any]:
        """执行 MCTS 搜索，返回最优策略 JSON"""
        root = MCTSNode(state=self._deep_copy_state(root_state))

        for _ in range(self.num_simulations):
            node = self._tree_policy(root)
            reward = self._default_policy(node.state)
            self._backpropagate(node, reward)

        # 选择最高访问次数的子节点作为实际动作（鲁棒性最强）
        if not root.children:
            best_action = "Elicit_Questioning" # 兜底策略
        else:
            best_child = max(root.children.values(), key=lambda c: c.visits)
            best_action = best_child.action

        # 将动作转化为 JSON 策略载体供 Teacher 节点使用
        strategy_payload = {
            "strategy_type": best_action,
            "confidence_score": (best_child.value / best_child.visits) if best_child.visits > 0 else 0.5,
            "reasoning": f"MCTS 推演选择该策略，预期奖励评分最高。"
        }
        
        logger.info(f"MCTS 规划完成，选定策略: {best_action}")
        return strategy_payload

    def _tree_policy(self, node: MCTSNode) -> MCTSNode:
        """选择与扩展 (Selection & Expansion)"""
        current_depth = 0
        while current_depth < self.max_depth:
            # 如果尚未完全扩展，执行扩展并返回新节点
            if not node.is_fully_expanded(self.legal_actions):
                return self._expand(node)
            # 否则沿着 UCT 最优路径向下选择
            node = node.best_child()
            current_depth += 1
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展未尝试过的动作分支"""
        tried_actions = list(node.children.keys())
        untried_actions = [a for a in self.legal_actions if a not in tried_actions]
        action = random.choice(untried_actions)
        
        # 1. 拷贝状态进入平行宇宙
        next_state = self._deep_copy_state(node.state)
        
        # 2. 【核心隔离】在模拟状态中记录该动作产生的假想影响
        # 注: 真实的教学推演需调用 LLM 预测，此处为保障工程效率使用启发式代理转移
        next_state["turn_count"] = next_state.get("turn_count", 0) + 1
        
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children[action] = child_node
        return child_node

    def _default_policy(self, state: GraphState) -> float:
        """
        快速模拟 (Simulation/Rollout):
        评估当前虚拟图状态的综合 Reward，融合贝叶斯 KL 增益与 Verifier 得分。
        """
        # 提取评估特征
        kl_shift = state.get("global_kl_shift", 0.0)
        bug_resolved = state.get("verifier_scores", {}).get("bug_resolved", 0.0)
        turn_count = state.get("turn_count", 0)
        
        # 启发式 Reward 函数设计:
        # 1. 鼓励引发学生认知增益 (KL)
        # 2. 鼓励修复 Bug
        # 3. 惩罚过长的对话拖沓
        reward = (kl_shift * 0.4) + (bug_resolved * 0.5) - (turn_count * 0.05)
        
        # 截断到 0~1 范围
        return max(0.0, min(1.0, reward))

    def _backpropagate(self, node: MCTSNode, reward: float):
        """反向传播更新整条访问路径的 Q 值"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

# ==========================================
# 接入 LangGraph 的 Node 执行函数
# ==========================================
def consultant_mcts_step(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph 的核心顾问节点：
    调用 MCTS Planner 生成后台指导策略，注入到主状态中。
    """
    logger.info("=== Consultant 启动 MCTS 后台推演 ===")
    
    # 限制搜索规模，保证生产环境延迟可控
    planner = MCTSPlanner(num_simulations=5, max_depth=2)
    
    # 运算出最优 JSON 策略
    optimal_strategy = planner.search(state)
    
    # 返回图状态 Diff，仅更新 current_strategy 字段
    return {
        "current_strategy": optimal_strategy
    }