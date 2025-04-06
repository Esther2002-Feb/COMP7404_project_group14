import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SVMClassifier:
    def __init__(self, kernel, C, gamma):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, train_features, train_labels):
        print("开始训练 SVM 模型...")
        self.model.fit(train_features, train_labels)

    def evaluate(self, test_features, test_labels, result_path):
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
