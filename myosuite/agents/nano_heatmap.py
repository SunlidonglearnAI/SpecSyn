import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import math
import os

# === 改动 1: 使用清新明亮的风格 ===
plt.style.use('seaborn-v0_8-whitegrid')

def visualize_evolution_bright_radar(save_path="bright_evolution_radar.png", show=True):
    print(f"Generating Bright Bio-Radar Visualization...")

    # 1. 模拟数据 (保持不变)
    muscle_groups = {
        "Wrist": ["ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"],
        "Thumb": ["EPL", "EPB", "FPL", "APL", "OP"],
        "Flexors": ["FDS5", "FDS4", "FDS3", "FDS2", "FDP5", "FDP4", "FDP3", "FDP2"],
        "Extensors": ["EDC5", "EDC4", "EDC3", "EDC2", "EDM", "EIP"],
        "Intrinsics": ["RI2", "LU2", "UI2", "RI3", "LU3", "UI3", "RI4", "LU4", "UI4", "RI5", "LU5", "UI5"]
    }

    all_names = []
    all_data = []
    group_spans = [] 

    current_idx = 0
    np.random.seed(42) # 固定随机种子以便复现效果
    for group_name, muscles in muscle_groups.items():
        start = current_idx
        for m_name in muscles:
            all_names.append(m_name)
            # === 模拟 T2A 策略 (Log Scale) ===
            f_val, l_val, s_val = 0, 0, 0 
            if group_name == "Flexors":
                f_val = np.random.uniform(0.5, 1.5)  # 变强
                s_val = np.random.uniform(0.5, 1.2)  # 变硬
                l_val = np.random.uniform(-0.2, 0.2)
            elif group_name == "Extensors":
                f_val = np.random.uniform(-1.5, -0.5) # 变弱
                s_val = np.random.uniform(-1.0, -0.2) # 变软
            elif group_name == "Wrist":
                s_val = np.random.uniform(1.0, 2.0)   # 极硬
                f_val = np.random.uniform(-0.2, 0.2)
            elif group_name == "Thumb":
                l_val = np.random.uniform(0.5, 1.0)   # 范围变大
            
            f_val += np.random.normal(0, 0.15)
            l_val += np.random.normal(0, 0.15)
            s_val += np.random.normal(0, 0.15)
            all_data.append([f_val, l_val, s_val])
            current_idx += 1
        group_spans.append((start, current_idx, group_name))

    data = np.array(all_data)
    N = len(all_names)
    
    # 2. 极坐标设置
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += np.pi / 2 
    width = (2 * np.pi / N) * 0.8 

    # 创建画布 (白色背景)
    fig = plt.figure(figsize=(14, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    # 移除极坐标默认的灰色网格和背景
    ax.set_facecolor('white')
    ax.grid(False)

    # === 改动 2: 新的配色方案 (适合白底) ===
    # 使用更深、对比度更高的颜色
    # F0: Deep Red, Length: Emerald Green, Stiffness: Strong Blue
    colors = ['#D70040', '#009E73', '#0072B2'] 
    labels = ['Force ($F_0$)', 'Length ($L_{opt}$)', 'Stiffness']
    
    BASE_R = 6.0
    
    # === 改动 3: 绘制解剖学分组背景块 (极浅灰色交替) ===
    # 交替使用极浅的灰色和纯白色
    bg_colors = ['#F8F8F8', '#FFFFFF', '#F8F8F8', '#FFFFFF', '#F8F8F8']
    for idx, (start, end, name) in enumerate(group_spans):
        start_angle = angles[start] - width/1.5
        end_angle = angles[end-1] + width/1.5
        if start_angle < 0: start_angle += 2*np.pi
        
        # 使用 bar 绘制扇形背景
        ax.bar(x=(start_angle + end_angle)/2, height=15, width=(end_angle - start_angle), 
               bottom=0, color=bg_colors[idx], edgecolor='none', zorder=0)
        
        mid_angle = (angles[start] + angles[end-1]) / 2
        # 分组标签改为深灰色
        ax.text(mid_angle, 14.2, name, color='#333333', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 分隔线改为浅灰色
        separator_angle = angles[end-1] + (angles[(end)%N] - angles[end-1])/2
        if idx < len(group_spans) - 1:
            ax.plot([separator_angle, separator_angle], [1.5, 13.5], color='#DDDDDD', linestyle='-', linewidth=1.5, zorder=1)

    # 4. 绘制基准线和参考线 (深灰色)
    ax.plot(np.linspace(0, 2*np.pi, 100), [BASE_R]*100, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    ax.text(0, BASE_R + 0.5, "Original (1.0x)", color='#333333', ha='center', fontsize=10, fontweight='bold')

    # 参考圆环 (浅灰色细线)
    ax.plot(np.linspace(0, 2*np.pi, 100), [BASE_R + 2.5]*100, color='#999999', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
    ax.text(np.pi/2, BASE_R + 2.8, "Stronger (2.0x)", color='#777777', fontsize=9, ha='center')
    
    ax.plot(np.linspace(0, 2*np.pi, 100), [BASE_R - 2.5]*100, color='#999999', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
    ax.text(np.pi/2, BASE_R - 3.2, "Weaker (0.5x)", color='#777777', fontsize=9, ha='center')

    # 5. 绘制数据点和连线
    for i in range(3):
        values = data[:, i] 
        radii = BASE_R + values * 2.5 
        
        # 绘制连线 (Stem) - 降低透明度，避免在白底上太乱
        for j in range(N):
            ax.plot([angles[j], angles[j]], [BASE_R, radii[j]], color=colors[i], linewidth=1.5, alpha=0.3, zorder=3)
            
        # 绘制散点 (Lollipops) - 增加不透明度，加一个细白边增加层次感
        ax.scatter(angles, radii, c=colors[i], s=70, alpha=0.9, edgecolors='white', linewidth=0.5, label=labels[i], zorder=4)

    # 6. 美化坐标轴和标签
    ax.set_ylim(0, 15) 
    ax.set_yticklabels([]) 
    ax.set_xticklabels([]) 

    # 添加肌肉名称标签 (中灰色)
    for j in range(N):
        angle = angles[j]
        name = all_names[j]
        rotation = angle * 180 / np.pi - 90
        if 90 < abs(rotation) < 270:
            rotation += 180
        ax.text(angle, 1.2, name, color='#555555', fontsize=8, ha='center', va='center', rotation=angle*180/np.pi - 90)

    # 7. 图例与标题 (黑色文字)
    plt.legend(loc='lower right', bbox_to_anchor=(1.12, 0.0), frameon=False, labelcolor='black', fontsize=12)
    
    plt.title("MORPHOLOGICAL EVOLUTION SPECTRUM", color='black', fontsize=24, pad=40, fontweight='bold')
    plt.figtext(0.5, 0.93, "Task: Object Hold | Agent: T2A-PPO | Style: Bright Journal", color='#555555', ha='center', fontsize=12)

    # Save (白色背景)
    save_path = os.path.abspath(save_path)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✅ Bright Viz saved to: {save_path}")
    
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    visualize_evolution_bright_radar()