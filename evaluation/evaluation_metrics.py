# [新增/重构] 独立出 9 维评价维度的具体算子
import os
import re
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_code_from_text(text: str) -> str:
    """
    使用正则表达式从大模型的回复文本中稳健提取 Python 代码块。
    (融合了防漏配与普通代码块兜底逻辑)
    """
    if not isinstance(text, str):
        return ""
        
    # 使用字符串拼接避免连续三个反引号导致 Markdown 解析器截断文件
    md_fence = "``" + "`" 
    
    # 优先匹配带有 python/py 标识的代码块
    pattern = rf"{md_fence}(?:python|py)\s*(.*?)\s*{md_fence}"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
        
    # 兜底匹配不带标识的普通代码块
    fallback_pattern = rf"{md_fence}\s*(.*?)\s*{md_fence}"
    fallback_matches = re.findall(fallback_pattern, text, flags=re.DOTALL)
    
    if fallback_matches:
        return fallback_matches[0].strip()
        
    return text.strip()  # 如果都没有匹配到，尝试直接返回原文本

def save_evaluation_results(results: List[Dict[str, Any]], output_path: str = "evaluation_results.json"):
    """
    将流水线评测结果安全落盘，支持嵌套目录的自动创建。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"评测结果已成功保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存评测结果失败: {e}")

def calculate_average_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    聚合计算所有实验样本的 9 维度平均分。
    用于生成最终的实验对比表格 (如论文中的 Table 1)
    """
    if not results:
        return {}
        
    aggregated = {}
    count = len(results)
    
    for res in results:
        scores = res.get("final_scores", {})
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                aggregated[metric] = aggregated.get(metric, 0.0) + float(value)
                
    # 计算均值并保留4位小数，方便直接贴入学术论文
    return {k: round(v / count, 4) for k, v in aggregated.items()}

def format_dialogue_history(messages: List[Any]) -> List[Dict[str, str]]:
    """
    辅助函数：将 LangGraph 中的 Message 对象列表格式化为标准字典，以便落盘保存。
    """
    formatted = []
    for msg in messages:
        # 通过类名判断是人类(Student)还是AI(Teacher)
        role = "student" if msg.__class__.__name__ == "HumanMessage" else "teacher"
        formatted.append({
            "role": role,
            "content": msg.content
        })
    return formatted