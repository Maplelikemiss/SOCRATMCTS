# [重构] MCTS 核心引擎：新增角色互换(Role-Reversal)，引入 Asyncio 并发与虚拟大模型推演以激活 vLLM APC
# [修复] 引入动态动作屏蔽 (Dynamic Action Masking)，死守 NDAR 学术红线
import math
import copy
import random
import logging
import asyncio
import re
import os
from typing import Dict, Any, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        """检查当前节点的子节点是否已经覆盖了所有合法动作"""
        return len(self.children) == len(legal_actions)

    def best_child(self, exploration_weight: float) -> 'MCTSNode':
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
    【重构亮点】：支持 Root Parallelization (根节点并发) 和虚拟大模型代理推演
    """
    def __init__(self, num_trees: int = 4, simulations_per_tree: int = 3, max_depth: int = 2, exploration_weight: float = 1.414):
        self.num_trees = num_trees
        self.simulations_per_tree = simulations_per_tree
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        
        # 初始化潜空间虚拟推演代理
        self.virtual_llm = ChatOpenAI(
            model_name="llama-3-8b-instruct", 
            temperature=0.7,
            api_key="EMPTY", 
            base_url="http://192.168.123.8:8001/v1"  # 指向你的本地 vLLM 或其他服务商地址
        )

    def get_legal_actions(self, state: GraphState) -> List[str]:
        """
        【修复核心】：动态计算动作空间 (Dynamic Action Masking)
        根据当前的对话轮次，决定是否放出 'Direct_Correction' 这个剧毒动作。
        """
        current_turn = state.get("turn_count", 0)
        # 提供默认值防止异常，通常 main.py 传入 max_turns 为 6 或 8
        max_turns = state.get("max_turns", 6) 
        
        base_actions = [
            "Elicit_Questioning",  # 启发式提问
            "Provide_Hint",        # 提供线索
            "Explain_Concept",     # 概念解释
            "Role_Reversal"        # 角色互换 (ZPD自适应降维)
        ]
        
        # 动作屏蔽逻辑：除非走到绝境 (离最大轮次只剩最后1~2轮)，否则直接答案选项根本不存在于搜索树中
        if current_turn >= max_turns - 2:
            base_actions.append("Direct_Correction")
            
        return base_actions

    def _clean_deep_copy(self, state: GraphState) -> GraphState:
        """极其严格的深度清洗拷贝，防止平行推演时状态污染"""
        new_state = {}
        for k, v in state.items():
            if k == "messages":
                new_msgs = []
                for msg in v:
                    msg_kwargs = copy.deepcopy(msg.additional_kwargs) if hasattr(msg, "additional_kwargs") else {}
                    if isinstance(msg, HumanMessage):
                        new_msgs.append(HumanMessage(content=msg.content, additional_kwargs=msg_kwargs))
                    elif isinstance(msg, AIMessage):
                        new_msgs.append(AIMessage(content=msg.content, additional_kwargs=msg_kwargs))
                    elif isinstance(msg, SystemMessage):
                        new_msgs.append(SystemMessage(content=msg.content, additional_kwargs=msg_kwargs))
                    else:
                        try:
                            new_msgs.append(msg.__class__(**msg.model_dump()))
                        except:
                            new_msgs.append(copy.deepcopy(msg))
                new_state[k] = new_msgs
            elif k == "student_kcs" and isinstance(v, dict):
                new_kcs = {}
                for kc_id, kc_state in v.items():
                    if hasattr(kc_state, "model_copy"):
                        new_kcs[kc_id] = kc_state.model_copy(deep=True)
                    else:
                        new_kcs[kc_id] = copy.deepcopy(kc_state)
                new_state[k] = new_kcs
            elif isinstance(v, (dict, list)):
                new_state[k] = copy.deepcopy(v)
            else:
                new_state[k] = v
                
        new_state["is_simulation"] = True
        return new_state

    # ==========================================
    # 异步并发搜索架构 (Root Parallelization)
    # ==========================================
    def search(self, root_state: GraphState) -> Dict[str, Any]:
        """同步入口包装器，适配 LangGraph 的普通 Node 调用"""
        logger.info("=== Consultant 启动 MCTS 潜空间并发推演 ===")
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self._async_parallel_search(root_state))
        except RuntimeError:
            return asyncio.run(self._async_parallel_search(root_state))

    async def _async_parallel_search(self, root_state: GraphState) -> Dict[str, Any]:
        """执行根节点并发 MCTS，利用 asyncio.gather 激活 vLLM APC"""
        # 获取当前根节点状态下合法的动作空间
        root_legal_actions = self.get_legal_actions(root_state)
        
        # 并发启动多棵 MCTS 树
        tasks = [self._run_single_tree(root_state) for _ in range(self.num_trees)]
        roots = await asyncio.gather(*tasks)
        
        # 聚合所有根节点的访问次数
        action_visits = {action: 0 for action in root_legal_actions}
        action_values = {action: 0.0 for action in root_legal_actions}
        
        for r in roots:
            for action, child in r.children.items():
                if action in action_visits: # 安全校验，确保只统计根节点合法动作
                    action_visits[action] += child.visits
                    action_values[action] += child.value
                
        # 选出被探索次数最多的策略作为综合最优解
        best_action = max(root_legal_actions, key=lambda a: action_visits[a])
        total_visits = action_visits[best_action]
        avg_value = (action_values[best_action] / total_visits) if total_visits > 0 else 0.5
        
        strategy_payload = {
            "strategy_type": best_action,
            "confidence_score": round(avg_value, 4),
            "internal_reasoning": f"MCTS在潜空间完成 {self.num_trees} 组并发推演。评估表明 {best_action} 策略在当前状态下具有最大期望教育收益 (Value: {avg_value:.2f})。"
        }
        
        logger.info(f"MCTS 并发推演完成，选定最优战术: {best_action}")
        return strategy_payload

    async def _run_single_tree(self, root_state: GraphState) -> MCTSNode:
        """单棵 MCTS 树的推演循环"""
        root = MCTSNode(state=self._clean_deep_copy(root_state))
        for _ in range(self.simulations_per_tree):
            node = self._tree_policy(root)
            reward = await self._async_rollout_evaluate(node.state, node.action)
            self._backpropagate(node, reward)
        return root

    def _tree_policy(self, node: MCTSNode) -> MCTSNode:
        """选择与扩展"""
        current_depth = 0
        while current_depth < self.max_depth:
            legal_actions = self.get_legal_actions(node.state)
            if not node.is_fully_expanded(legal_actions):
                return self._expand(node)
            node = node.best_child(self.exploration_weight)
            current_depth += 1
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展未尝试过的动作分支"""
        legal_actions = self.get_legal_actions(node.state)
        tried_actions = list(node.children.keys())
        untried_actions = [a for a in legal_actions if a not in tried_actions]
        
        if not untried_actions:
            # 防御性回退：如果当前节点无动作可扩展，直接返回自身进行评估
            return node
            
        action = random.choice(untried_actions)
        
        next_state = self._clean_deep_copy(node.state)
        # 为深层节点塞入伪动作标记，模拟回合流转，以此让底层的 get_legal_actions 动态生效
        next_state["turn_count"] = next_state.get("turn_count", 0) + 1
        
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children[action] = child_node
        return child_node

    async def _async_rollout_evaluate(self, state: GraphState, action: str) -> float:
        """
        虚拟代理 Rollout：调用 LLM 进行一轮虚拟推演。
        """
        if not action:
            return 0.5
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个MCTS潜空间推演引擎。根据下方的师生对话历史，如果老师接下来采用【{action}】策略，"
                       "请推测学生的回应，并评估该策略带来的教育价值（综合考量Bug修复概率与认知增益）。\n"
                       "要求：在你的回复的最后一行，必须严格且唯一地输出一个 0.0 到 1.0 的浮点数作为最终 Reward。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "【系统推演指令】\n拟采取策略: {action}\n开始你的推演，并给出最后的浮点数评分。")
        ])
        
        try:
            # 真实并发生成，激活底层缓存
            response = await self.virtual_llm.ainvoke({
                "chat_history": state.get("messages", []),
                "action": action
            })
            
            # 【新增：闲聊惩罚机制】
            # 如果推演出学生在接下来的回复中没有包含实质性的代码词汇，说明当前策略无效
            if len(re.findall(r'[a-zA-Z0-9_]{3,}', response.content)) < 2:
                # 几乎没有技术词汇，判定为闲聊或无效回复
                reward = 0.1
            else:
                # 正常提取浮点数 Reward (保留了原有的双重正则容错机制)
                match = re.search(r'(0\.\d+|1\.0|0\.0)', response.content.split('\n')[-1])
                if match:
                    reward = float(match.group(1))
                else:
                    match_all = re.findall(r'(0\.\d+|1\.0|0\.0)', response.content)
                    reward = float(match_all[-1]) if match_all else 0.5
                
        except Exception as e:
            logger.debug(f"虚拟代理评估异常，返回中性值: {e}")
            reward = 0.5

        # ==========================================
        # 奖励后处理 (Reward Shaping)
        # ==========================================
        # 1. 毁灭性打击：直接给答案 (绝对守护 NDAR 红线)
        if action == "Direct_Correction":
            # 即使 LLM 觉得很好，强制置为极低分甚至负分，确保其只作为绝对的保底被选中
            reward = min(reward * 0.1, 0.1) 
            
        # 2. 启发性奖励：角色互换通常具有极高的突破性教育价值
        if action == "Role_Reversal":
            reward += 0.1

        return max(0.0, min(1.0, reward))

    def _backpropagate(self, node: MCTSNode, reward: float):
        """反向传播更新整条访问路径的 Q 值"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent