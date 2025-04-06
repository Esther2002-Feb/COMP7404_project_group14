import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SVMClassifier:
    def __init__(self, kernel, C, gamma):
        """
        初始化支持向量机 (SVM) 模型

        参数：
        - kernel: SVM 的核函数类型（如 'linear', 'rbf', 'poly' 等）
        - C: 正则化参数
        - gamma: 核函数的系数
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, train_features, train_labels):
        """
        训练 SVM 模型

        参数：
        - train_features: 训练特征
        - train_labels: 训练标签
        """
        print("开始训练 SVM 模型...")
        self.model.fit(train_features, train_labels)

    def evaluate(self, test_features, test_labels, result_path):
        """
        评估 SVM 模型性能

        参数：
        - test_features: 测试特征
        - test_labels: 测试标签
        - result_path: 模型预测结果保存路径
        """
        print("开始测试 SVM 模型...")
        predicted_labels = self.model.predict(test_features)

        # 保存预测结果
        np.savez(
            result_path,
            actual=test_labels,
            predicted=predicted_labels
        )

        # 打印模型评估结果
        print(f"测试集准确率: {accuracy_score(test_labels, predicted_labels):.4f}")
        print("\n分类报告:")
        print(classification_report(test_labels, predicted_labels))
        print("\n混淆矩阵:")
        print(confusion_matrix(test_labels, predicted_labels))


# 主程序逻辑
def main():
    # 超参数
    KERNEL = 'rbf'  # 核函数类型
    C = 1.0         # 正则化参数
    GAMMA = 'scale' # 核函数系数

    # 加载数据
    data = load_breast_cancer()
    features = data.data
    labels = data.target

    # 数据标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 划分训练集和测试集 (70% 训练，30% 测试)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, stratify=labels, random_state=42
    )

    # 创建 SVM 模型
    model = SVMClassifier(
        kernel=KERNEL,
        C=C,
        gamma=GAMMA
    )

    # 训练模型
    model.train(train_features, train_labels)

    # 设置结果存储路径
    result_path = "./svm_results.npz"

    # 测试和评估模型
    model.evaluate(test_features, test_labels, result_path)


if __name__ == "__main__":
    main()