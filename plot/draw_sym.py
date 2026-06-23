import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================================================================
#  学术配色与样式设置
# ==============================================================================
COLORS = {
    'Symmetry': '#0072B2',    # 蓝色 (T2Apca0)
    'Asymmetry': '#D70040',   # 红色 (T2Apca 无0)
    'Baseline': '#555555'     # 灰色 (Walk)
}

def plot_symmetry_ablation_final(csv_path, save_name='Symmetry_vs_Asymmetry_Rough_Fixed.png'):
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 1. 加载数据并截断到 10000 步
    df = pd.read_csv(csv_path)
    df = df[df['Step'] <= 10000]

    plt.figure(figsize=(10, 6), dpi=150)

    # 2. 逻辑分层匹配
    all_columns = df.columns.tolist()
    
    # --- 分组 A: Symmetry (严格匹配 T2Apca0) ---
    sym_mean = [c for c in all_columns if 'T2Apca0' in c and '__' not in c]
    sym_min = [c for c in all_columns if 'T2Apca0' in c and '__MIN' in c]
    sym_max = [c for c in all_columns if 'T2Apca0' in c and '__MAX' in c]

    # --- 分组 B: Asymmetry (包含 T2Apca 但不含 T2Apca0) ---
    asym_mean = [c for c in all_columns if ('T2Apca' in c and 'T2Apca0' not in c) and '__' not in c]
    asym_min = [c for c in all_columns if ('T2Apca' in c and 'T2Apca0' not in c) and '__MIN' in c]
    asym_max = [c for c in all_columns if ('T2Apca' in c and 'T2Apca0' not in c) and '__MAX' in c]

    # --- 分组 C: Baseline (Walk) ---
    base_mean = [c for c in all_columns if 'Walk' in c and '__' not in c]
    
    # 绘图函数化，减少冗余
    def draw_curve(mean_list, min_list, max_list, label, color):
        if mean_list:
            x = df['Step']
            y_mean = df[mean_list[0]].rolling(window=15, min_periods=1).mean()
            plt.plot(x, y_mean, label=label, color=color, linewidth=2.5)
            if min_list and max_list:
                y_min = df[min_list[0]].rolling(window=15, min_periods=1).mean()
                y_max = df[max_list[0]].rolling(window=15, min_periods=1).mean()
                plt.fill_between(x, y_min, y_max, color=color, alpha=0.15, edgecolor='none')
            print(f"成功绘制: {label}")

    # 执行绘图
    
    draw_curve(asym_mean, asym_min, asym_max, 'SDE (Asymmetric)', COLORS['Asymmetry'])
    draw_curve(sym_mean, sym_min, sym_max, 'SDE (Bilateral Symmetry)', COLORS['Symmetry'])
    draw_curve(base_mean, [], [], 'Baseline', COLORS['Baseline'])

    # 3. 细节美化
    plt.title('Symmetry vs. Asymmetry (Rough_terrain)', fontsize=24, fontweight='bold', pad=15)
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Episode Reward Mean', fontsize=24)
    
    plt.xlim(0, 10000)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=20, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    print(f"\n图片已保存至: {save_name}")
    plt.show()

if __name__ == "__main__":
    # 请确保这是你包含这两条曲线的 CSV 文件名
    target_csv = "symmetry_rough.csv" 
    plot_symmetry_ablation_final(target_csv)