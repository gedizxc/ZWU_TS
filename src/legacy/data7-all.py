import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 如果本地没有安装tqdm，可以去掉这行和下方的tqdm包装，但这能显示进度条

# --- 1. 基础配置 ---
image_dir = "batch_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

stats_file = "batch_stats.txt"
seq_len = 96  # 每一批的长度

# 读取数据
df = pd.read_csv('ETTh1.csv')
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
df_features = df[features]

total_len = len(df_features)
num_batches = total_len // seq_len  # 计算总共有多少个完整的批次

print(f"数据总行数: {total_len}")
print(f"每批次长度: {seq_len}")
print(f"预计生成批次: {num_batches} 个")


# --- 2. 定义计算函数 ---

def simple_dtw_distance(s1, s2):
    """计算DTW距离"""
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix.fill(np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]

print("d")

# --- 3. 主循环处理 ---

# 打开文件准备写入文本
with open(stats_file, "w", encoding='utf-8') as f:
    # 使用 tqdm 显示进度条 (如果你不想用 tqdm，可以将 range 改为普通的 range(num_batches))
    # for batch_idx in range(num_batches):
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):

        # 确定当前批次的起止索引
        start_idx = batch_idx * seq_len
        end_idx = start_idx + seq_len

        # 获取当前批次数据
        batch_df = df_features.iloc[start_idx:end_idx]
        batch_data = batch_df.values

        # ---------------------------
        # A. 生成文本统计特征
        # ---------------------------
        f.write(f"--- Batch {batch_idx} ---\n")
        stats_list = []
        for i, col in enumerate(features):
            series = batch_data[:, i]
            # 保留两位小数
            stat_str = f"{col}: Mean={series.mean():.2f}, Max={series.max():.2f}, Var={series.var():.2f}"
            stats_list.append(stat_str)

        # 写入文件
        f.write("\n".join(stats_list))
        f.write("\n\n")  # 批次之间空两行

        # ---------------------------
        # B. 生成图片特征 (DTW / FFT-Cov / FFT-Pearson)
        # ---------------------------

        # 数据标准化 (Z-Score) - 消除量纲差异，对DTW和相关性计算很重要
        data_mean = batch_data.mean(axis=0)
        data_std = batch_data.std(axis=0)
        # 加上1e-5防止除以0
        data_norm = (batch_data - data_mean) / (data_std + 1e-5)

        # 1. FFT 变换
        fft_data = np.fft.rfft(data_norm, axis=0)
        fft_magnitude = np.abs(fft_data)

        # 2. 计算 Pearson 相关系数 (频域)
        pearson_matrix = np.corrcoef(fft_magnitude, rowvar=False)

        # 3. 计算 协方差 (频域)
        cov_matrix = np.cov(fft_magnitude, rowvar=False)

        # 4. 计算 DTW 矩阵 (时域)
        num_vars = len(features)
        dtw_matrix = np.zeros((num_vars, num_vars))
        for i in range(num_vars):
            for j in range(num_vars):
                if i == j:
                    dtw_matrix[i, j] = 0
                else:
                    dtw_matrix[i, j] = simple_dtw_distance(data_norm[:, i], data_norm[:, j])

        # ---------------------------
        # C. 绘图并保存
        # ---------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # DTW Heatmap
        sns.heatmap(dtw_matrix, annot=True, fmt=".1f", cmap="viridis",
                    xticklabels=features, yticklabels=features, ax=axes[0])
        axes[0].set_title(f"Batch {batch_idx} DTW Distance (Time Domain)")

        # Covariance Heatmap
        sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=features, yticklabels=features, ax=axes[1])
        axes[1].set_title(f"Batch {batch_idx} Covariance (FFT Magnitude)")

        # Pearson Heatmap
        sns.heatmap(pearson_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    xticklabels=features, yticklabels=features, ax=axes[2])
        axes[2].set_title(f"Batch {batch_idx} Pearson Correlation (FFT Magnitude)")

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(image_dir, f"batch_{batch_idx}.png")
        plt.savefig(save_path)
        plt.close(fig)  # 重要：关闭图像以释放内存，否则循环180次会爆内存

print(f"处理完成！")
print(f"图片已保存至: {os.path.abspath(image_dir)}")
print(f"文本统计已保存至: {os.path.abspath(stats_file)}")