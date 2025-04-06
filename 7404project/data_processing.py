# data_processing.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

def data_exploration(features, labels):
    """
    数据探索：输出热力图，显示相关性分析。
    """
    print("开始数据探索...")

    correlation_matrix = np.corrcoef(features, rowvar=False)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Matrix (Pearson)")
    plt.show()

    # 输出关联特征分组
    positive_corr = []
    negative_corr = []
    no_corr = []

    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[0]):  # 避免重复计算
            r = correlation_matrix[i, j]
            if r > 0.7:
                positive_corr.append((i, j, r))
            elif r < -0.7:
                negative_corr.append((i, j, r))
            elif abs(r) < 0.2:
                no_corr.append((i, j, r))

    print(f"强正相关特征对: {positive_corr}")
    print(f"强负相关特征对: {negative_corr}")
    print(f"弱相关特征对: {no_corr}")

    return positive_corr


def drop_redundant_features(features, positive_corr):
    """
    根据强相关性结果，选择性保留特征，并输出保留特征的编号和名称。

    参数:
        features: 特征数据 (numpy array 或 pandas DataFrame)。
        feature_names: 特征名称 (列表或 pandas DataFrame 列名)。
        positive_corr: 强相关性特征对 (列表形式，例如 [(0, 1, 0.9), (2, 3, 0.85)])。

    返回:
        features: 剔除冗余特征后的特征数据。
    """
    redundant_features = set()  # 保存冗余特征的索引
    for (i, j, r) in positive_corr:  # 遍历相关性结果
        if j in redundant_features:
            continue
        redundant_features.add(j)  # 标记冗余特征（始终保留第一个）

    # 保留的特征索引
    retained_feature_indices = [i for i in range(features.shape[1]) if i not in redundant_features]

    # 输出保留的特征编号和名称
    print("保留的特征编号和名称:")
    for index in retained_feature_indices:
        print(f"特征编号: {index} ")

        # 剔除冗余特征并返回数据
    pruned_features = features[:, retained_feature_indices]
    return pruned_features, retained_feature_indices
def feature_selection(features, labels, num_features_to_select=15):
    """
    使用递归特征消除 (RFE) 筛选重要特征
    """
    print("开始特征选择 (RFE)...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)
    rfe.fit(features, labels)

    # 获取选择的重要特征
    selected_features = rfe.support_
    selected_indices = np.where(selected_features)[0]
    print(f"选择的重要特征索引: {selected_indices}")

    # 返回筛选后的特征和索引
    return features[:, selected_features], selected_indices