# [重构] MCTS 核心引擎：新增代理状态转移机制与严格深度清洗拷贝
import math
import copy
import random
import logging
from typing import Dict, Any, List, Optional

# 【关键修复 2】引入更全面的 LangChain 消息基类，防范多智能体环境下的特殊消息类型
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

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

    def best_child(self, exploration_weight: float) -> 'MCTSNode':
        """使用 UCT (UCB1) 算法选择最优子节点 (去除了硬编码，由外部传入探索权重)"""
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
    # 【关键修复 1】暴露 exploration_weight (UCT 常数 C) 用于开发集调优
    def __init__(self, num_simulations: int = 5, max_depth: int = 3, exploration_weight: float = 1.414):
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        
        # 预定义合法的苏格拉底教学动作空间 (Action Space)
        self.legal_actions = [
            "Elicit_Questioning",  # 启发式提问 (不给答案，反问引导)
            "Provide_Hint",        # 提供线索 (指出方向，但不写代码)
            "Explain_Concept",     # 概念解释 (补充缺失的背景知识)
            "Direct_Correction"    # 直接纠正 (仅在极端死循环时使用)
        ]

    def _clean_deep_copy(self, state: GraphState) -> GraphState:
        """
        【关键修复 2】极其严格的深度清洗拷贝 (Deep Clean Copy)
        替代容易引发异常和 State Bleeding 的朴素 copy.deepcopy。
        专门反序列化并重建 LangChain 的各种 Message 对象，确保平行宇宙间的物理内存绝对隔离。
        """
        new_state = {}
        for k, v in state.items():
            if k == "messages":
                new_msgs = []
                for msg in v:
                    # 安全提取 additional_kwargs
                    msg_kwargs = copy.deepcopy(msg.additional_kwargs) if hasattr(msg, "additional_kwargs") else {}
                    
                    if isinstance(msg, HumanMessage):
                        new_msgs.append(HumanMessage(content=msg.content, additional_kwargs=msg_kwargs))
                    elif isinstance(msg, AIMessage):
                        new_msgs.append(AIMessage(content=msg.content, additional_kwargs=msg_kwargs))
                    elif isinstance(msg, SystemMessage):
                        new_msgs.append(SystemMessage(content=msg.content, additional_kwargs=msg_kwargs))
                    elif isinstance(msg, ToolMessage):
                        # ToolMessage 必须保留 tool_call_id
                        t_id = getattr(msg, 'tool_call_id', '')
                        new_msgs.append(ToolMessage(content=msg.content, tool_call_id=t_id, additional_kwargs=msg_kwargs))
                    else:
                        # 终极兜底方案：通过 Pydantic 字典导出与解包重建，彻底斩断底层引用
                        try:
                            new_msgs.append(msg.__class__(**msg.dict()))
                        except Exception:
                            # 极端情况下的安全降级
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
        【关键修复 3】稳定期望的启发式代理状态转移 (Stable Heuristic Proxy State Transition)
        使用 "稳定基线 + 微小噪声" 替代 "纯高方差随机分布"，
        防止在 num_simulations 较小时策略评估发生严重抖动 (Flapping)。
        """
        kl = state.get("global_kl_shift", 0.0)
        bug = state.get("verifier_scores", {}).get("bug_resolved", 0.0)

        # 辅助函数：基于核心期望值注入少量噪声
        noise = lambda base, var: base + random.uniform(-var, var)

        if action == "Elicit_Questioning":
            # 启发提问：认知增益大，但短期内解 Bug 慢
            kl += noise(0.10, 0.02)
            bug += noise(0.02, 0.01)
        elif action == "Provide_Hint":
            # 给定线索：两者较为均衡
            kl += noise(0.05, 0.01)
            bug += noise(0.15, 0.02)
        elif action == "Explain_Concept":
            # 解释概念：认知增益极大，稳定推进
            kl += noise(0.15, 0.02)
            bug += noise(0.08, 0.01)
        elif action == "Direct_Correction":
            # 【红线策略】直接纠正：Bug光速修复，但认知增益为 0
            kl += 0.0 
            bug += noise(0.50, 0.05)

        # 回写状态，确保在合法的 0-1 概率空间内
        state["global_kl_shift"] = min(1.0, max(0.0, kl))
        if "verifier_scores" not in state:
            state["verifier_scores"] = {}
        state["verifier_scores"]["bug_resolved"] = min(1.0, max(0.0, bug))
        state["turn_count"] = state.get("turn_count", 0) + 1

    def search(self, root_state: GraphState) -> Dict[str, Any]:
        """执行 MCTS 搜索，返回最优策略 JSON"""
        root = MCTSNode(state=self._clean_deep_copy(root_state))

        for _ in range(self.num_simulations):
            node = self._tree_policy(root)
            reward = self._default_policy(node.state)
            self._backpropagate(node, reward)

        if not root.children:
            best_action = "Elicit_Questioning" # 兜底策略
        else:
            best_child = max(root.children.values(), key=lambda c: c.visits)
            best_action = best_child.action

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
            if not node.is_fully_expanded(self.legal_actions):
                return self._expand(node)
            # 传入动态配置的探索常量 C
            node = node.best_child(self.exploration_weight)
            current_depth += 1
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展未尝试过的动作分支"""
        tried_actions = list(node.children.keys())
        untried_actions = [a for a in self.legal_actions if a not in tried_actions]
        action = random.choice(untried_actions)
        
        next_state = self._clean_deep_copy(node.state)
        self._simulate_action_effect(next_state, action)
        
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children[action] = child_node
        return child_node

    def _default_policy(self, state: GraphState) -> float:
        """快速模拟 (Simulation/Rollout): 评估当前虚拟图状态的综合 Reward"""
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
    # 在未来的开发集消融实验中，可以通过读取环境变量或配置文件传入 exploration_weight
    planner = MCTSPlanner(num_simulations=10, max_depth=3, exploration_weight=1.414)
    optimal_strategy = planner.search(state)
    
    return {"current_strategy": optimal_strategy}