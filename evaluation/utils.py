# [复用] 各类数据集解析、打分辅助函数
import json
import logging
import os
from typing import List, Dict, Any

# 导入在步骤2中定义的贝叶斯微观状态结构
from state.graph_state import BayesianKnowledgeState

logger = logging.getLogger(__name__)

def load_socrat_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    安全加载 SocratDataset.json 等实验数据集文件。
    添加了完备的异常捕获与日志记录。
    """
    if not os.path.exists(filepath):
        logger.error(f"严重错误：数据集文件未找到: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        # 如果包裹在某个根 key 下，自动剥离 (兼容多种常见数据集结构)
        if isinstance(dataset, dict) and "data" in dataset:
            dataset = dataset["data"]
            
        logger.info(f"成功加载数据集: {filepath}，共包含 {len(dataset)} 条代码问题记录。")
        return dataset
    except json.JSONDecodeError as e:
        logger.error(f"JSON 文件解析失败，请检查格式是否合法: {e}")
        raise

def build_initial_graph_state(
    dataset_item: Dict[str, Any], 
    base_persona: str = "normal", 
    max_turns: int = 8
) -> Dict[str, Any]:
    """
    将数据集中的单条记录，转换为 LangGraph 可以直接执行的 Initial State (初始状态字典)。
    """
    # 1. 兼容性字段提取 (兼容 prompt/problem, code/buggy_code 等不同命名)
    problem_desc = dataset_item.get("problem_description", dataset_item.get("prompt", "未知问题"))
    buggy_code = dataset_item.get("buggy_code", dataset_item.get("code", "未知代码"))
    target_kc = dataset_item.get("target_kc", dataset_item.get("bug_type", "target_bug_understanding"))

    # 2. 【核心技巧】动态画像注入 (Persona Injection)
    # 不破坏图的起步逻辑，而是通过修改 System Prompt 强迫 Student 问出特定的题
    injected_persona = (
        f"{base_persona}\n\n"
        "【强制任务背景 - 请严格遵守】\n"
        f"你正在解决以下编程问题：\n{problem_desc}\n"
        f"你目前写出的（存在Bug的）代码如下：\n```python\n{buggy_code}\n```\n"
        "如果这是我们的第一轮对话，请在你的回复中直接给出这段代码，并抱怨它运行不对，向老师求助。"
    )

    # 3. 预先初始化本题对应的目标知识点 (KC) 的先验概率
    initial_kcs = {
        target_kc: BayesianKnowledgeState(kc_id=target_kc, prior_prob=0.5, posterior_prob=0.5)
    }

    # 4. 返回符合 GraphState TypedDict 规范的完整字典
    return {
        "messages": [], # 留空，让图状态机的 student_node 自行触发第一句话
        "student_kcs": initial_kcs,
        "global_kl_shift": 0.0,
        "current_strategy": None,
        "verifier_scores": {},
        "is_simulation": False,
        "student_persona": injected_persona, # 携带了题目和Bug代码的“作弊版”画像
        "turn_count": 0,
        "max_turns": max_turns
    }