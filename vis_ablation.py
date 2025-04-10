# import matplotlib.pyplot as plt
# import os

# # 创建目录（如果不存在）
# save_dir = "/data/haoran/Point2Roof/vis_fpfh_mrgd"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "metrics_comparison.png")

# # 数据
# metrics = ["CR", "CF1", "ER", "EF1"]
# data = {
#     "xyz+mrgd": [0.703, 0.813, 0.555, 0.69],
#     "xyz+mrgd+fpfh": [0.705, 0.815, 0.56, 0.695],
#     "xyz+fpfh": [0.695, 0.807, 0.537, 0.675],
#     "xyz only": [0.686, 0.801, 0.525, 0.662]
# }

# # 绘图
# plt.figure(figsize=(10, 6))

# for label, values in data.items():
#     plt.plot(metrics, values, marker='o', label=label)

# plt.title("Performance Metrics for Different Feature Combinations")
# plt.xlabel("Metrics")
# plt.ylabel("Value")
# plt.ylim(0.5, 0.9)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()

# # 保存图片
# plt.savefig(save_path, dpi=300)
# plt.close()


import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np
import os

plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# 指标（去除ACO、CP、EP）
metrics = ["CR", "CF1", "ER", "EF1"]
num_metrics = len(metrics)

# 方法顺序调整，并将所有数值 ×100
data = {
    "xyz+mrgd+fpfh": [0.705, 0.815, 0.56, 0.695],
    "xyz+mrgd": [0.703, 0.813, 0.555, 0.69],
    "xyz+fpfh": [0.695, 0.807, 0.537, 0.675],
    "xyz only": [0.686, 0.801, 0.525, 0.662]
}
methods = list(data.keys())
values = np.array(list(data.values())) * 100  # 转换为百分比

# 色彩：科学、协调、风格一致
colors = ['#b36a6f', '#98a1b1', '#aebfce', '#ceb5b9']

# 柱状图参数
num_methods = len(methods)
width = 0.18
group_gap = 0.1  # 指标之间的组间距
x = np.arange(num_metrics) * (num_methods * width + group_gap)

# 保存路径
save_dir = "/data/haoran/Point2Roof/vis_fpfh_mrgd"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "bar_comparison_metrics_pretty_percent_spacing_font.png")

# 绘图
plt.figure(figsize=(8, 5.5))
for i, (method, color) in enumerate(zip(methods, colors)):
    plt.bar(x + i * width, values[i], width=width, label=method, color=color)

# 添加数值标签
for i in range(num_methods):
    for j in range(num_metrics):
        val = values[i][j]
        plt.text(x[j] + i * width, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# 设置图表元素
plt.xticks(x + width * (num_methods - 1) / 2, metrics)
plt.ylabel("Score (%)")
plt.ylim(50, 85)
plt.title("Comparison of Feature Combinations on Semantic Metrics")
plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(loc="upper left", frameon=False)
plt.tight_layout()

# 保存并关闭
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.close()
