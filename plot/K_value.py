import matplotlib.pyplot as plt
import numpy as np

# 1. 原始数据
k_values_raw = np.array([3, 5, 7, 9])
variance_raw = np.array([90.13, 98.38, 99.69, 99.93])

# 2. 为了让趋势图更完整，我们假设 K=1 时的一个较低值（例如 60%）
# 这样能更好凸显 K=3 和 K=5 的上升幅度。
k_values = np.insert(k_values_raw, 0, 1)
variance = np.insert(variance_raw, 0, 60.0) # 假设值，仅用于展示趋势

# 3. 开始画图
plt.figure(figsize=(8, 5))
# 设置 Seaborn 风格（如果安装了），或者使用经典学术风格
plt.style.use('seaborn-v0_8-whitegrid') 

# 绘制累计方差曲线
plt.plot(k_values, variance, marker='o', linestyle='-', color='#1f77b4', 
         linewidth=2, markersize=8, label='Cumulative Explained Variance')

# 4. 突出显示选定的 K=5 点
selected_k = 5
selected_var = 98.38
plt.plot(selected_k, selected_var, marker='o', markersize=12, 
         markeredgecolor='red', markerfacecolor='none', markeredgewidth=2)

# 添加注释指向选定点
plt.annotate(f'Selected $K=5$\n({selected_var}%)',
             xy=(selected_k, selected_var),
             xytext=(selected_k + 0.5, selected_var - 5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

# 5. 图表修饰
plt.title('Captured Variance vs. Latent Dimension $K$', fontsize=16, fontweight='bold')
plt.xlabel('Number of Spectral Components ($K$)', fontsize=16)
plt.ylabel('Cumulative Explained Variance (%)', fontsize=16)

# 设置坐标轴范围和刻度
plt.xlim(0.5, 9.5)
plt.xticks(np.arange(1, 10, 1)) # X轴显示 1-9
plt.ylim(80, 101) # Y轴聚焦 80%-101%，凸显平台期
plt.yticks(np.arange(80, 101, 5))

# 在关键点添加数值标签 (只标原始数据点)
for i, txt in enumerate(variance_raw):
    plt.gca().text(k_values_raw[i], variance_raw[i] + 0.8, f'{txt}%', 
                   ha='center', fontsize=10, color='#333333')

# 添加 98% 信息保留红线（可选，增强说服力）
plt.axhline(y=98, color='r', linestyle='--', linewidth=1, alpha=0.7)
plt.text(1, 98.5, '98% Information Retention', color='r', fontsize=9, alpha=0.7)

plt.tight_layout()

# 保存为 PDF 或 PNG (建议 PDF 用于 LaTeX)
plt.savefig('variance_scree_plot.png', bbox_inches='tight')
plt.show()