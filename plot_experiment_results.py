import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# 1. 全局样式设置
# ==========================================
plt.style.use('ggplot')

RESULTS_DIR = "evaluation_results"
FILES = [
    "Vanilla_Prompting_results.json",
    "TreeInstruct_Baseline_results.json",
    "Ablation_No_MCTS_results.json",
    "Ablation_No_LLMKT_results.json",
    "Socrat_Full_results.json"
]

# 去掉了 "(生死红线)" 的中文标注，保持纯英文
RADAR_METRICS = ["ndar", "bug_resolved", "logicality", "repetitiveness", "guidance", "clarity"]
METRIC_LABELS = ["NDAR", "Bug Resolved", "Logicality", "Repetitiveness", "Guidance", "Clarity"]

# 配置雷达图各模型线条样式
MODES_CONFIG = {
    "Vanilla_Prompting": {"label": "Vanilla (Baseline)", "color": "#9E9E9E", "ls": "--", "alpha": 0.0, "lw": 1.5},
    "TreeInstruct_Baseline": {"label": "TreeInstruct", "color": "#2196F3", "ls": "-.", "alpha": 0.15, "lw": 2.0},
    "Ablation_No_MCTS": {"label": "w/o MCTS", "color": "#FF9800", "ls": ":", "alpha": 0.15, "lw": 2.0},
    "Ablation_No_LLMKT": {"label": "w/o LLMKT", "color": "#9C27B0", "ls": ":", "alpha": 0.15, "lw": 2.0},
    "Socrat_Full": {"label": "SocratMCTS (Ours)", "color": "#E53935", "ls": "-", "alpha": 0.3, "lw": 3.0} 
}

PERSONAS = {
    "normal": "Normal Persona",
    "zero_base": "Zero-Base Persona",
    "random_noise": "Random Noise Persona"
}

def load_data():
    all_records = []
    for filename in FILES:
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath): continue
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                record = {"mode": item["experiment_mode"], "persona": item["persona"], "kl_shift": item.get("final_kl_shift", 0.0)}
                for metric in RADAR_METRICS:
                    record[metric] = item["final_scores"].get(metric, 0.0)
                all_records.append(record)
    return pd.DataFrame(all_records).groupby(['mode', 'persona']).mean().reset_index()

# ==========================================
# 2. 生成 单画像雷达图 
# ==========================================
def generate_persona_charts():
    df = load_data()
    if df.empty: return
    
    angles = [n / float(len(RADAR_METRICS)) * 2 * np.pi for n in range(len(RADAR_METRICS))]
    angles += angles[:1]
    
    for persona_key, persona_title in PERSONAS.items():
        fig = plt.figure(figsize=(8, 8))
        
        # --- 雷达图绘制 ---
        ax_radar = fig.add_subplot(1, 1, 1, polar=True)
        ax_radar.set_theta_offset(np.pi / 2) # 设置 12 点钟为起点 (NDAR)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        
        # 为 X 轴标签加粗
        ax_radar.set_xticklabels(METRIC_LABELS, fontsize=12, fontweight='bold')
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
        ax_radar.set_ylim(0, 1.05)
        
        for i, (mode, config) in enumerate(MODES_CONFIG.items()):
            row = df[(df['mode'] == mode) & (df['persona'] == persona_key)]
            if row.empty: continue
            
            # --- 绘制雷达图数据 ---
            values = row[RADAR_METRICS].values.flatten().tolist()
            values += values[:1]
            zorder = 10 if mode == "Socrat_Full" else 5
            
            ax_radar.plot(angles, values, linewidth=config.get('lw', 1.5), linestyle=config['ls'], color=config['color'], label=config['label'], zorder=zorder)
            if config['alpha'] > 0:
                ax_radar.fill(angles, values, color=config['color'], alpha=config['alpha'], zorder=zorder-1)

        # 统一标题与图例
        fig.suptitle(f"System Performance Evaluation\n{persona_title}", fontsize=16, fontweight='bold', y=1.02)
        handles = [Patch(facecolor=config['color'], label=config['label']) for config in MODES_CONFIG.values()]
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        out_file = f"eval_optimized_{persona_key}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"✅ Generated Persona chart: {out_file}")
        plt.close(fig)

# ==========================================
# 3. 生成 全局鲁棒性对比柱状图 
# ==========================================
def generate_global_robustness_chart():
    df = load_data()
    if df.empty: return

    fig, ax = plt.subplots(figsize=(10, 6))

    personas_keys = list(PERSONAS.keys())
    persona_labels = list(PERSONAS.values())
    
    # 指定仅对比的两个模式
    modes_to_plot = ["Ablation_No_MCTS", "Socrat_Full"]

    x = np.arange(len(personas_keys))
    width = 0.3  

    # 分组绘制柱状图
    for i, mode in enumerate(modes_to_plot):
        config = MODES_CONFIG[mode]
        y_values = []
        for p in personas_keys:
            row = df[(df['mode'] == mode) & (df['persona'] == p)]
            y_values.append(row['kl_shift'].values[0] if not row.empty else 0.0)
            
        # 计算每根柱子的偏移量
        offset = (i - len(modes_to_plot)/2 + 0.5) * width
        bars = ax.bar(x + offset, y_values, width, label=config['label'], color=config['color'], edgecolor='black', linewidth=0.8)
        
        # 在柱子上添加数值
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=11)

    ax.set_ylabel('Cognitive Gain (KL Shift)', fontsize=13, fontweight='bold')
    # 去掉了标题里的中文 "(全局鲁棒性分析)"
    ax.set_title('MCTS Contribution Across Student Personas', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(persona_labels, fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 图例配置
    handles = [Patch(facecolor=MODES_CONFIG[m]['color'], label=MODES_CONFIG[m]['label']) for m in modes_to_plot]
    ax.legend(handles=handles, title='System Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    out_file = "eval_global_robustness.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Generated Global Robustness chart: {out_file}")
    plt.close(fig)

if __name__ == "__main__":
    generate_persona_charts()
    generate_global_robustness_chart()