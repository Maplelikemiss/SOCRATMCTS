# [新增] 框架启动入口
from dotenv import load_dotenv
load_dotenv()  # 自动读取并加载同级目录下的 .env 文件
import os
import time
import logging
import argparse
from typing import List, Dict, Any

# 1. 导入 LangGraph 状态机编排应用
# 注意：如果你的 langgraph_app.py 中函数名是 create_socrat_graph，请在此处修改或保持对齐
from langgraph_app import build_socrat_mcts_graph

# 2. 导入数据工具与指标计算模块
from evaluation.utils import load_socrat_dataset, build_initial_graph_state
from evaluation.evaluation_metrics import (
    format_dialogue_history, 
    save_evaluation_results, 
    calculate_average_metrics
)
from state.graph_state import BayesianKnowledgeState

# 配置全局日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SocratMCTS_Pipeline")

def run_evaluation_pipeline(
    dataset_path: str, 
    output_path: str, 
    sample_size: int = -1, 
    personas_to_test: List[str] = None
):
    """
    运行宏观实验评测流水线。
    遍历数据集中的代码问题，并为每种对抗性画像运行 SocratMCTS 状态机。
    """
    if personas_to_test is None:
        # 默认运行论文中规划的 3 组对比画像
        personas_to_test = ["normal", "zero_base", "random_noise"]
        
    logger.info("=" * 60)
    logger.info("🚀 SocratMCTS 学术实验评估流水线正式启动 🚀")
    logger.info("=" * 60)

    # 1. 加载数据集
    dataset = load_socrat_dataset(dataset_path)
    if sample_size > 0:
        dataset = dataset[:sample_size]
        logger.info(f"已开启截断测试，当前测试样本量: {sample_size}")

    # 2. 构建并编译 LangGraph 状态机
    socrat_app = build_socrat_mcts_graph()
    
    all_results = []
    start_time = time.time()

    # 3. 嵌套循环：遍历数据 x 遍历画像
    for idx, item in enumerate(dataset):
        # 兼容不同的题目 ID 字段
        question_id = item.get("id", item.get("question_id", f"Task_{idx+1}"))
        
        logger.info("-" * 40)
        logger.info(f"开始评测题目 [{question_id}] ({idx+1}/{len(dataset)})")
        
        for persona in personas_to_test:
            logger.info(f"  -> 注入对抗性画像: {persona}")
            
            # 初始化本轮对话的安全隔离状态
            initial_state = build_initial_graph_state(item, base_persona=persona, max_turns=6)
            
            # 【终极拦截补丁】强制抹除 evaluation/utils.py 内部可能残留的 0.5 历史遗留！
            initial_state["student_kcs"] = {
                "target_bug_understanding": BayesianKnowledgeState(
                    kc_id="target_bug_understanding",
                    prior_prob=0.2,     # 强制上锁 0.2
                    posterior_prob=0.2  # 强制上锁 0.2
                )
            }
            
            # 【核心注入】确保图状态中拥有空白的历史数组起步
            initial_state["verifier_history"] = []
            
            try:
                # 【核心驱动】触发 LangGraph 状态机流转
                final_state = socrat_app.invoke(initial_state, config={"recursion_limit": 100})
                
                # 从最终状态中提取分析所需的数据
                formatted_history = format_dialogue_history(final_state.get("messages", []))
                
                # 【核心修改点】提取历史分数并进行智能聚合计算
                history = final_state.get("verifier_history", [])
                if not history:
                    history = [final_state.get("verifier_scores", {})]
                    
                # 聚合规则：Bug态取Max, ndar取Min, 其他所有过程维度(prr/spr/iar/5维)全取Avg
                final_scores = {
                    "bug_resolved": max((h.get("bug_resolved", 0.0) for h in history), default=0.0),
                    "ndar": min((h.get("ndar", 1.0) for h in history), default=1.0),
                    "prr": round(sum(h.get("prr", 1.0) for h in history) / len(history), 2),
                    "spr": round(sum(h.get("spr", 1.0) for h in history) / len(history), 2),
                    "iar": round(sum(h.get("iar", 1.0) for h in history) / len(history), 2),
                    "logicality": round(sum(h.get("logicality", 0.6) for h in history) / len(history), 2),
                    "repetitiveness": round(sum(h.get("repetitiveness", 0.6) for h in history) / len(history), 2),
                    "guidance": round(sum(h.get("guidance", 0.6) for h in history) / len(history), 2),
                    "flexibility": round(sum(h.get("flexibility", 0.6) for h in history) / len(history), 2),
                    "clarity": round(sum(h.get("clarity", 0.6) for h in history) / len(history), 2),
                }
                
                global_kl = final_state.get("global_kl_shift", 0.0)
                
                # 组装单条落盘记录
                result_entry = {
                    "question_id": question_id,
                    "persona": persona,
                    "total_turns": final_state.get("turn_count", 0),
                    "final_kl_shift": round(global_kl, 4),
                    "final_scores": final_scores,
                    "dialogue_history": formatted_history
                }
                all_results.append(result_entry)
                logger.info(f"     ✅ 测评成功 | Bug解决率: {final_scores.get('bug_resolved', 0):.2f} | 轮次: {final_state.get('turn_count', 0)}")
                
                # 【核心修复：实时断点落盘】每跑完一道题的某一个画像，立刻覆盖保存一次！
                # 这样即使手动 Ctrl+C 或者断电，之前跑完的数据绝不会丢失。
                save_evaluation_results(all_results, output_path)
                
            except Exception as e:
                logger.error(f"     ❌ 测评崩溃 (问题ID: {question_id}, 画像: {persona}) - 错误信息: {e}")
                all_results.append({
                    "question_id": question_id,
                    "persona": persona,
                    "error": str(e)
                })
                
                # 【核心修复：崩溃也落盘】把带 error 的残缺记录也存下来，方便排查
                save_evaluation_results(all_results, output_path)
                
    # 4. 统计与聚合输出
    end_time = time.time()
    logger.info("=" * 60)
    logger.info(f"🏁 评测流水线运行完毕！耗时: {(end_time - start_time):.2f} 秒")
    
    valid_results = [r for r in all_results if "error" not in r]
    avg_metrics = calculate_average_metrics(valid_results)
    
    logger.info("📊 全局平均指标概览 (可用于撰写论文 Table 1):")
    for metric, val in avg_metrics.items():
        logger.info(f"    - {metric}: {val}")
        
    save_evaluation_results(all_results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SocratMCTS 批量评估引擎")
    parser.add_argument("--dataset", type=str, default="SocratDataset.json", help="数据集 JSON 文件路径")
    parser.add_argument("--output", type=str, default="evaluation_results/final_report.json", help="结果输出路径")
    parser.add_argument("--sample_size", type=int, default=3, help="限制测试样本量 (-1 表示全量测试)")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        logger.warning(f"⚠️ 未找到数据集 {args.dataset}。请确保从您的原 KELE 仓库中将数据文件拷贝到此同级目录。")
        logger.warning("我将自动创建一个极其简单的 mock_dataset.json 以供系统预热测试...")
        
        mock_data = [
            {
                "id": "mock_001",
                "problem_description": "写一个函数实现两数之和。",
                "buggy_code": "def twoSum(nums, target):\n    for i in range(len(nums)):\n        if nums[i] + nums[i+1] == target: return [i, i+1]",
                "target_kc": "array_index_out_of_bounds"
            }
        ]
        with open("mock_dataset.json", "w", encoding="utf-8") as f:
            import json
            json.dump(mock_data, f, ensure_ascii=False, indent=2)
        args.dataset = "mock_dataset.json"
        
    # 执行流水线
    run_evaluation_pipeline(
        dataset_path=args.dataset,
        output_path=args.output,
        sample_size=args.sample_size,
        #personas_to_test=["normal", "stubborn"] # 演示模式下仅跑两种典型画像
        personas_to_test=["normal", "zero_base", "random_noise"]
    )