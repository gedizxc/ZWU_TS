import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 1. 基础配置 ---
image_dir = "batch_images_time"  # 新的文件夹名字，避免和之前的混淆
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

stats_file = "batch_stats_time.txt"
seq_len = 96  # 序列长度

# 读取数据
df = pd.read_csv('ETTh1.csv')
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
df_features = df[features]

total_len = len(df_features)
num_batches = total_len // seq_len

print(f"--- 配置信息 ---")
print(f"输出目录: {os.path.abspath(image_dir)}")
print(f"统计文件: {os.path.abspath(stats_file)}")
print(f"预计处理批次: {num_batches}")


# --- 2. 定义计算函数 ---

def simple_dtw_distance(s1, s2):
    """
    计算两个向量之间的DTW距离。
    在这里，输入s1, s2是长度为7的特征向量。
    """
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


# --- 3. 主循环处理 ---

with open(stats_file, "w", encoding='utf-8') as f:
    # 使用 tqdm 显示进度
    for batch_idx in tqdm(range(num_batches), desc="Processing Time-Correlations"):

        # 获取当前批次数据
        start_idx = batch_idx * seq_len
        end_idx = start_idx + seq_len
        batch_df = df_features.iloc[start_idx:end_idx]
        batch_data = batch_df.values  # Shape: (96, 7)

        # ---------------------------
        # A. 生成文本统计 (保持不变，方便大模型参考)
        # ---------------------------
        f.write(f"--- Batch {batch_idx} ---\n")
        stats_list = []
        for i, col in enumerate(features):
            series = batch_data[:, i]
            stat_str = f"{col}: Mean={series.mean():.2f}, Max={series.max():.2f}, Var={series.var():.2f}"
            stats_list.append(stat_str)
        f.write("\n".join(stats_list))
        f.write("\n\n")

        # ---------------------------
        # B. 计算 96x96 时间相关性矩阵
        # ---------------------------

        # 标准化 (Z-Score)
        # 这一步很关键，确保不同特征的量纲一致，否则DTW会被大数值特征主导
        data_mean = batch_data.mean(axis=0)
        data_std = batch_data.std(axis=0)
        data_norm = (batch_data - data_mean) / (data_std + 1e-5)

        # 1. Pearson Correlation (Time vs Time)
        # np.corrcoef 默认计算行与行的相关性，这正是我们想要的
        # 输入: (96, 7) -> 输出: (96, 96)
        pearson_matrix = np.corrcoef(data_norm)

        # 2. Covariance (Time vs Time)
        # np.cov 默认也是计算行与行的协方差
        cov_matrix = np.cov(data_norm)

        # 3. DTW Distance Matrix (Time vs Time)
        # 这是一个 (96, 96) 的矩阵
        # 含义：第 i 个时刻的特征向量 vs 第 j 个时刻的特征向量 的距离
        num_timesteps = seq_len
        dtw_matrix = np.zeros((num_timesteps, num_timesteps))

        # 双重循环计算两两时间点的距离
        for i in range(num_timesteps):
            for j in range(num_timesteps):
                if i == j:
                    dtw_matrix[i, j] = 0
                elif i > j:
                    # 矩阵是对称的，利用这一点减少一半计算量
                    dtw_matrix[i, j] = dtw_matrix[j, i]
                else:
                    # 计算两个长度为7的向量之间的DTW距离
                    dtw_matrix[i, j] = simple_dtw_distance(data_norm[i, :], data_norm[j, :])

        # ---------------------------
        # C. 绘图并保存
        # ---------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 绘制热力图 (关闭 annot 因为格子太密了)

        # DTW
        sns.heatmap(dtw_matrix, annot=False, cmap="viridis_r",  # 使用 viridis_r (反转)，让深色代表距离近(相似)，更符合直觉
                    xticklabels=10, yticklabels=10, ax=axes[0])
        axes[0].set_title(f"Batch {batch_idx} DTW (Time Distance)")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Time Step")

        # Covariance
        sns.heatmap(cov_matrix, annot=False, cmap="coolwarm",
                    xticklabels=10, yticklabels=10, ax=axes[1])
        axes[1].set_title(f"Batch {batch_idx} Covariance (Time)")

        # Pearson
        sns.heatmap(pearson_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1,
                    xticklabels=10, yticklabels=10, ax=axes[2])
        axes[2].set_title(f"Batch {batch_idx} Pearson (Time)")

        plt.tight_layout()

        # 保存
        save_path = os.path.join(image_dir, f"batch_{batch_idx}.png")
        plt.savefig(save_path)
        plt.close(fig)  # 释放内存

print("所有批次处理完成！")