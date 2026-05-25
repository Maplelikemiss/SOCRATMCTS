import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# 1. Global Settings (Removed Chinese fonts, using default default)
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

RADAR_METRICS = ["bug_resolved", "ndar", "logicality", "repetitiveness", "guidance", "clarity"]
# Translated Metric Labels
METRIC_LABELS = ["Bug Resolved", "NDAR", "Logicality", "Repetitiveness", "Guidance", "Clarity"]

# Translated Mode Labels
MODES_CONFIG = {
    "Vanilla_Prompting": {"label": "Vanilla (Baseline)", "color": "#9E9E9E", "ls": "--", "alpha": 0.3},
    "TreeInstruct_Baseline": {"label": "TreeInstruct", "color": "#2196F3", "ls": "-.", "alpha": 0.5},
    "Ablation_No_MCTS": {"label": "w/o MCTS", "color": "#FF9800", "ls": ":", "alpha": 0.5},
    "Ablation_No_LLMKT": {"label": "w/o LLMKT", "color": "#9C27B0", "ls": ":", "alpha": 0.5},
    "Socrat_Full": {"label": "SocratMCTS (Ours)", "color": "#E53935", "ls": "-", "alpha": 0.8, "lw": 2.5}
}

# Translated Persona Labels
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
# 2. Generate Charts for Each Persona
# ==========================================
def generate_persona_charts():
    df = load_data()
    if df.empty: return
    
    angles = [n / float(len(RADAR_METRICS)) * 2 * np.pi for n in range(len(RADAR_METRICS))]
    angles += angles[:1]
    
    for persona_key, persona_title in PERSONAS.items():
        fig = plt.figure(figsize=(14, 6))
        
        # --- Left: Radar Chart ---
        ax_radar = fig.add_subplot(1, 2, 1, polar=True)
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(METRIC_LABELS, fontsize=11)
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
        ax_radar.set_ylim(0, 1.05)
        
        # --- Right: Bar Chart ---
        ax_bar = fig.add_subplot(1, 2, 2)
        
        bar_x = np.arange(len(MODES_CONFIG))
        bar_y = []
        bar_colors = []
        bar_labels = []
        
        for i, (mode, config) in enumerate(MODES_CONFIG.items()):
            row = df[(df['mode'] == mode) & (df['persona'] == persona_key)]
            if row.empty: continue
            
            # Draw radar line
            values = row[RADAR_METRICS].values.flatten().tolist()
            values += values[:1]
            ax_radar.plot(angles, values, linewidth=config.get('lw', 1.5), linestyle=config['ls'], color=config['color'], label=config['label'])
            ax_radar.fill(angles, values, color=config['color'], alpha=0.05)
            
            # Collect bar data
            bar_y.append(row['kl_shift'].values[0])
            bar_colors.append(config['color'])
            bar_labels.append(config['label'])

        # Draw bar chart
        bars = ax_bar.bar(bar_x, bar_y, color=bar_colors, width=0.6)
        ax_bar.bar_label(bars, fmt='%.2f', padding=3, fontsize=11)
        ax_bar.set_xticks(bar_x)
        ax_bar.set_xticklabels(bar_labels, rotation=15, ha='right', fontsize=10)
        ax_bar.set_ylabel('Cognitive Gain (KL Shift)', fontsize=12, fontweight='bold')
        ax_bar.set_title('Cognitive Tracking Performance (LLMKT)', fontsize=12)
        
        # Unified Title and Legend
        fig.suptitle(f"System Performance Evaluation - {persona_title}", fontsize=16, fontweight='bold', y=1.05)
        handles = [Patch(facecolor=config['color'], label=config['label']) for config in MODES_CONFIG.values()]
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=11, bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        out_file = f"eval_{persona_key}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"✅ Generated chart: {out_file}")
        plt.close(fig)

if __name__ == "__main__":
    generate_persona_charts()