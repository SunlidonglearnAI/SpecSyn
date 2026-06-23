import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================================================================
#  配置区域
# ==============================================================================
# 设置学术配色
COLORS = {
    'T2Apca': '#0072B2',  # 蓝色 (PCA 降维进化)
    'T2A': '#D70040',     # 红色 (全维度进化)
    'Baseline': '#555555' # 灰色 (原始环境)
}

def plot_comparison(csv_path, save_dir):
    # 1. 读取数据
    df = pd.read_csv(csv_path)
    file_name = os.path.basename(csv_path)
    terrain_name = file_name.replace('compare_', '').replace('.csv', '').capitalize()
    
    print(f"Processing {terrain_name}...")

    plt.figure(figsize=(9, 6), dpi=150)
    
    # 2. 定义匹配逻辑 (适配你 CSV 里的具体列名)
    # T2Apca-v0, T2A-v0, Walk-v0
    configs = [
        ('T2Apca-v0', 'SDE', COLORS['T2Apca']),
        ('T2A-v0', 'T2A', COLORS['T2A']),
        ('Walk-v0', 'Fixed Morphology', COLORS['Baseline']),
    ]

    for key, label, color in configs:
        # 寻找包含该关键词的列
        # Mean 列：包含关键词但没有 MIN/MAX
        mean_cols = [c for c in df.columns if key in c and '__MIN' not in c and '__MAX' not in c]
        min_cols = [c for c in df.columns if key in c and '__MIN' in c]
        max_cols = [c for c in df.columns if key in c and '__MAX' in c]

        if mean_cols and min_cols and max_cols:
            x = df['Step']
            y_mean = df[mean_cols[0]]
            y_min = df[min_cols[0]]
            y_max = df[max_cols[0]]

            # 绘制阴影 (Alpha 透明度设为 0.15)
            plt.fill_between(x, y_min, y_max, color=color, alpha=0.15, edgecolor='none')
            # 绘制主线 (粗细设为 2)
            plt.plot(x, y_mean, label=label, color=color, linewidth=2)

    # 3. 图表美化
    plt.title(f'Learning Curves on {terrain_name} Terrain', fontsize=24, pad=15, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Episode Reward Mean', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=20, frameon=True, shadow=False)
    
    # 自动调整布局避免文字重叠
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"plot_{terrain_name.lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"SUCCESS: Plot saved to {save_path}")

# ==============================================================================
#  执行区域
# ==============================================================================
if __name__ == "__main__":
    # csv 所在的文件夹
    csv_folder = "./" 
    # 图片输出的文件夹
    output_folder = "./plots"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv') and f.startswith('compare_')]
    
    if not csv_files:
        print("Error: No 'compare_*.csv' files found in current directory.")
    else:
        for file in csv_files:
            plot_comparison(os.path.join(csv_folder, file), output_folder)
        print("\nAll tasks finished. Check the './plots' folder.")