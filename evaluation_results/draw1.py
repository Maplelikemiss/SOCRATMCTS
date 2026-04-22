import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

def draw_radar_chart(json_path: str, save_path: str = "evaluation_results/radar_chart.png"):
    # 1. 读取数据
    if not os.path.exists(json_path):
        print(f"找不到数据文件: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data:
        return
        
    # 计算所有样本的平均分 (目前只有 task_001，也就是单样本的分数)
    scores_list = [item['final_scores'] for item in data]
    avg_scores = {}
    for key in scores_list[0].keys():
        if key == "bug_resolved": # bug_resolved 不在雷达图中展示
            continue
        avg_scores[key] = np.mean([s[key] for s in scores_list])

    # 2. 准备雷达图参数
    categories = list(avg_scores.keys())
    values = list(avg_scores.values())
    N = len(categories)

    # 闭合雷达图的线
    values += values[:1]
    
    # 计算角度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # 3. 开始绘图 (使用学术风格的极坐标)
    plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.subplot(111, polar=True)

    # 第一根轴线放在正上方
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # 设置刻度轴与标签
    plt.xticks(angles[:-1], categories, color='black', size=12, fontweight='bold')
    
    # 设置 y 轴刻度
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.1)

    # 绘制 SocratMCTS 的数据面
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='SocratMCTS', color='#1f77b4')
    ax.fill(angles, values, '#1f77b4', alpha=0.25)

    # 添加标题与图例
    plt.title('Multi-dimensional Socratic Teaching Evaluation', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"雷达图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    draw_radar_chart("./evaluation_results/final_report.json")