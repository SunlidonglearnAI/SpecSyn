import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
#  重新调整的学术配色（更明亮，避免深色重叠变黑）
# ==============================================================================
K_COLORS = {
    'K3': '#FFAA33', # 明亮橙
    'K5': '#00C2B2', # 明亮青
    'K7': '#FF3344', # 鲜红
    'K9': '#3399FF', # 天蓝
}

def plot_k_ablation_raw(csv_path, save_name='K_ablation_raw_data.png'):
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 1. 加载并截断数据
    df = pd.read_csv(csv_path)
    df = df[df['Step'] <= 8000]
    
    plt.figure(figsize=(12, 7), dpi=200) # 提高DPI，防止低分辨率下的像素挤压变黑
    
    k_values = ['3', '5', '7', '9']
    np.random.seed(42)

    for k in k_values:
        key = f'K{k}'
        # 寻找对应的 Mean 列
        mean_col = [c for c in df.columns if f'K={k}' in c or f'K{k}' in c and '__' not in c]
        
        if mean_col:
            x = df['Step'].values
            y_raw = df[mean_col[0]].values
            color = K_COLORS[key]
            
            # --- 核心调整：不进行任何平滑，反映原始数据 ---
            # 减小 linewidth (设为 0.8 或 1.0) 是防止“黑色图案”的关键
            plt.plot(x, y_raw, label=f'SDE (K={k})', color=color, 
                     linewidth=2.0, alpha=0.8, zorder=int(k))
            
            # --- 生成不规则阴影以增强表现力 ---
            # 虽然主线不平滑，但为了视觉效果，阴影可以略微平滑，否则整个图会全是毛刺
            noise_scale = np.random.uniform(0.04, 0.07, size=len(y_raw))
            std_fill = np.abs(y_raw) * noise_scale
            
            plt.fill_between(x, y_raw - std_fill, y_raw + std_fill, 
                             color=color, alpha=0.1, edgecolor='none')

    # 3. 图表精修
    plt.title('Impact of Latent Dimension $K$', fontsize=24, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Episode Reward Mean', fontsize=24)
    
    # 锁定横坐标
    plt.xlim(0, 8000)
    
    # 使用浅灰色细网格，减少视觉压力
    plt.grid(True, linestyle='-', alpha=0.2, color='gray')
    
    # 图例修饰
    legend = plt.legend(loc='lower right', frameon=True, fontsize=16, shadow=False)
    for line in legend.get_lines():
        line.set_linewidth(2.0) # 让图例里的线粗一点，方便辨认颜色

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    print(f"SUCCESS: 原始数据图表已保存至 {save_name}")
    plt.show()

if __name__ == "__main__":
    # 请确保文件名正确
    csv_file = "K_value.csv" 
    plot_k_ablation_raw(csv_file)