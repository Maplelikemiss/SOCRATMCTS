# [重构] MCTS 核心引擎：新增代理状态转移机制与严格深度清洗拷贝
import math
import copy
import random
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

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

    def _clean_deep_copy(self, state: GraphState) -> GraphState:
        """
        【关键修复 1】严格的深度清洗拷贝 (Deep Clean Copy)
        替代容易引发异常和 State Bleeding 的朴素 copy.deepcopy。
        专门反序列化并重建 LangChain 的 Message 对象，确保平行宇宙间的物理内存绝对隔离。
        """
        new_state = {}
        for k, v in state.items():
            if k == "messages":
                # 重建所有消息对象，防止浅拷贝导致的并发写入污染
                new_msgs = []
                for msg in v:
                    if isinstance(msg, HumanMessage):
                        new_msgs.append(HumanMessage(content=msg.content, additional_kwargs=copy.deepcopy(msg.additional_kwargs)))
                    elif isinstance(msg, AIMessage):
                        new_msgs.append(AIMessage(content=msg.content, additional_kwargs=copy.deepcopy(msg.additional_kwargs)))
                    else:
                        # 兼容 BaseMessage 或字典
                        new_msgs.append(copy.deepcopy(msg))
                new_state[k] = new_msgs
            elif isinstance(v, (dict, list)):
                # 针对知识节点追踪状态 (student_kcs) 等复杂嵌套对象
                new_state[k] = copy.deepcopy(v)
            else:
                # 针对标量 (turn_count, global_kl_shift 等)
                new_state[k] = v
                
        # 标记为模拟状态
        new_state["is_simulation"] = True
        return new_state

    def _simulate_action_effect(self, state: GraphState, action: str) -> None:
        """
        【关键修复 2】启发式代理状态转移 (Heuristic Proxy State Transition)
        在无法于搜索树中直接调用 LLM 的工程约束下，模拟不同动作对认知环境的影响。
        赋予不同策略真实的“预期价值差异”，打破 MCTS 的“盲目搜索”陷阱。
        """
        kl = state.get("global_kl_shift", 0.0)
        bug = state.get("verifier_scores", {}).get("bug_resolved", 0.0)

        # 概率性增益模型（符合教育学预期）
        if action == "Elicit_Questioning":
            # 启发提问：认知增益大，但短期内解 Bug 慢
            kl += random.uniform(0.05, 0.15)
            bug += random.uniform(0.0, 0.05)
        elif action == "Provide_Hint":
            # 给定线索：两者较为均衡
            kl += random.uniform(0.02, 0.08)
            bug += random.uniform(0.1, 0.2)
        elif action == "Explain_Concept":
            # 解释概念：认知增益极大，稳定推进
            kl += random.uniform(0.1, 0.2)
            bug += random.uniform(0.05, 0.1)
        elif action == "Direct_Correction":
            # 【红线策略】直接纠正：Bug光速修复，但认知增益为 0 (严重拉低后续评估分数)
            kl += 0.0 
            bug += random.uniform(0.4, 0.6)

        # 回写状态
        state["global_kl_shift"] = min(1.0, kl)
        if "verifier_scores" not in state:
            state["verifier_scores"] = {}
        state["verifier_scores"]["bug_resolved"] = min(1.0, bug)
        state["turn_count"] = state.get("turn_count", 0) + 1

    def search(self, root_state: GraphState) -> Dict[str, Any]:
        """执行 MCTS 搜索，返回最优策略 JSON"""
        root = MCTSNode(state=self._clean_deep_copy(root_state))

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
            "reasoning": f"MCTS 潜空间推演完毕，评估 {best_action} 策略在平衡 KL 增益与 Bug 解决率上具有全局最优的长期价值。"
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
        
        # 1. 严格深拷贝进入平行宇宙
        next_state = self._clean_deep_copy(node.state)
        
        # 2. 注入启发式状态转移，令动作真正地“改变环境”
        self._simulate_action_effect(next_state, action)
        
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children[action] = child_node
        return child_node

    def _default_policy(self, state: GraphState) -> float:
        """
        快速模拟 (Simulation/Rollout):
        评估当前虚拟图状态的综合 Reward，融合贝叶斯 KL 增益与 Verifier 得分。
        """
        kl_shift = state.get("global_kl_shift", 0.0)
        bug_resolved = state.get("verifier_scores", {}).get("bug_resolved", 0.0)
        turn_count = state.get("turn_count", 0)
        
        # 启发式 Reward 函数设计: 鼓励启发认知、鼓励修复Bug、惩罚冗长推演
        reward = (kl_shift * 0.4) + (bug_resolved * 0.5) - (turn_count * 0.05)
        
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
    LangGraph 的核心顾问节点：调用 MCTS Planner 生成后台指导策略。
    """
    logger.info("=== Consultant 启动 MCTS 后台推演 ===")
    
    # 限制搜索规模，保证生产环境延迟可控
    planner = MCTSPlanner(num_simulations=10, max_depth=3)
    optimal_strategy = planner.search(state)
    
    return {"current_strategy": optimal_strategy}