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
    output_dir: str, 
    mode: str, 
    personas_to_test: List[str],
    sample_size: int = -1
):
    """
    运行宏观实验评测流水线。
    遍历数据集中的代码问题，并为每种对抗性画像运行指定模式下的 SocratMCTS 状态机。
    """
    
    # 动态构建基于模式的独立输出路径
    output_path = os.path.join(output_dir, f"{mode}_results.json")
        
    logger.info("=" * 60)
    logger.info(f"🚀 SocratMCTS 评测流水线启动 | 模式: {mode}")
    logger.info(f"👥 注入画像集: {', '.join(personas_to_test)}")
    logger.info(f"💾 数据落盘至: {output_path}")
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
            
            # 初始化本轮对话的安全隔离状态，【新增】注入 experiment_mode
            initial_state = build_initial_graph_state(
                item, 
                base_persona=persona, 
                max_turns=8, 
                experiment_mode=mode
            )
            
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
                
                # 【提取历史单轮分数】
                history = final_state.get("verifier_history", [])
                if not history:
                    history = [final_state.get("verifier_scores", {})]
                
                # 【提取终局宏观定调分数】
                global_scores = final_state.get("global_evaluation_scores", {})
                if not global_scores:
                    logger.warning(f"⚠️ 未找到全局评价分数 (问题ID: {question_id})，启用中立底分")
                    global_scores = {
                        "logicality": 0.6,
                        "repetitiveness": 0.6,
                        "guidance": 0.6,
                        "flexibility": 0.6,
                        "clarity": 0.6
                    }
                    
                # 聚合规则：
                # Bug态取Max, ndar取Min(红线豁免权)
                # prr/spr/iar 单轮行为，取Avg
                # 5个全局指标 直接读取 global_evaluator_node 生成的权威定调分！
                final_scores = {
                    "bug_resolved": max((h.get("bug_resolved", 0.0) for h in history), default=0.0),
                    "ndar": min((h.get("ndar", 1.0) for h in history), default=1.0),
                    "prr": round(sum(h.get("prr", 1.0) for h in history) / len(history), 2),
                    "spr": round(sum(h.get("spr", 1.0) for h in history) / len(history), 2),
                    "iar": round(sum(h.get("iar", 1.0) for h in history) / len(history), 2),
                    
                    # 彻底解耦：从全局评价结果中读取
                    "logicality": global_scores.get("logicality", 0.6),
                    "repetitiveness": global_scores.get("repetitiveness", 0.6),
                    "guidance": global_scores.get("guidance", 0.6),
                    "flexibility": global_scores.get("flexibility", 0.6),
                    "clarity": global_scores.get("clarity", 0.6),
                }
                
                global_kl = final_state.get("global_kl_shift", 0.0)
                
                # 组装单条落盘记录，【新增】带上 experiment_mode 方便后期验证
                result_entry = {
                    "question_id": question_id,
                    "experiment_mode": mode,
                    "persona": persona,
                    "total_turns": final_state.get("turn_count", 0),
                    "final_kl_shift": round(global_kl, 4),
                    "final_scores": final_scores,
                    "dialogue_history": formatted_history
                }
                all_results.append(result_entry)
                logger.info(f"     ✅ 测评成功 | Bug解决率: {final_scores.get('bug_resolved', 0):.2f} | 轮次: {final_state.get('turn_count', 0)}")
                
                # 【核心修复：实时断点落盘】每跑完一道题的某一个画像，立刻覆盖保存一次！
                save_evaluation_results(all_results, output_path)
                
            except Exception as e:
                logger.error(f"     ❌ 测评崩溃 (问题ID: {question_id}, 画像: {persona}) - 错误信息: {e}")
                all_results.append({
                    "question_id": question_id,
                    "experiment_mode": mode,
                    "persona": persona,
                    "error": str(e)
                })
                
                # 【核心修复：崩溃也落盘】
                save_evaluation_results(all_results, output_path)
                
    # 4. 统计与聚合输出
    end_time = time.time()
    logger.info("=" * 60)
    logger.info(f"🏁 模式 [{mode}] 评测流水线运行完毕！耗时: {(end_time - start_time):.2f} 秒")
    
    valid_results = [r for r in all_results if "error" not in r]
    avg_metrics = calculate_average_metrics(valid_results)
    
    logger.info("📊 模式全局平均指标概览 (可用于撰写论文 Table 1):")
    for metric, val in avg_metrics.items():
        logger.info(f"    - {metric}: {val}")
        
    save_evaluation_results(all_results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SocratMCTS 批量评估引擎")
    parser.add_argument("--dataset", type=str, default="SocratDataset.json", help="数据集 JSON 文件路径")
    # 【修改点】改为接收 output_dir，而不是具体的文件名
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="结果输出目录")
    parser.add_argument("--sample_size", type=int, default=3, help="限制测试样本量 (-1 表示全量测试)")
    
    # 【新增点】添加 mode 选项控制实验变量
    parser.add_argument(
        "--mode", 
        type=str, 
        default="Socrat_Full", 
        choices=[
            "Socrat_Full", 
            "Vanilla_Prompting", 
            "TreeInstruct_Baseline", 
            "Ablation_No_MCTS", 
            "Ablation_No_LLMKT",
            "all"
        ], 
        help="指定运行的实验模式，自动隔离输出结果。"
    )
    
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
        
    # 【新增点 2】批处理循环执行逻辑
    if args.mode == "all":
        all_modes = [
            "Socrat_Full", 
            "Vanilla_Prompting", 
            "TreeInstruct_Baseline", 
            "Ablation_No_MCTS", 
            "Ablation_No_LLMKT"
        ]
        
        logger.info("=" * 60)
        logger.info("🔄 检测到 'all' 模式，即将开始 5 组实验的自动化批处理评测...")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        for current_mode in all_modes:
            logger.info("▼" * 60)
            logger.info(f"🧪 [自动批处理] 正在切换到实验组: 【 {current_mode} 】")
            logger.info("▲" * 60)
            
            run_evaluation_pipeline(
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                mode=current_mode,
                personas_to_test=["normal", "zero_base", "random_noise"], # 演示模式
                sample_size=args.sample_size
            )
            
        total_end_time = time.time()
        logger.info("=" * 60)
        logger.info(f"🎉 批处理任务圆满结束！5 种模式已全部跑完，总耗时: {(total_end_time - total_start_time):.2f} 秒")
        logger.info(f"📁 各模式对应的独立 JSON 结果文件已保存在: {args.output_dir}/ 目录下")
        logger.info("=" * 60)
        
    else:
        # 单一模式直接执行
        run_evaluation_pipeline(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            mode=args.mode,
            personas_to_test=["normal", "zero_base", "random_noise"], # 演示模式
            #personas_to_test=["normal"],
            sample_size=args.sample_size
        )