import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler
from data_processing import data_exploration, feature_selection

def load_and_preprocess():
    """加载并预处理数据"""
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    features = data.data
    labels = data.target
    
    # 数据归一化
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    # 特征选择
    features, _ = feature_selection(features, labels, num_features_to_select=15)
    
    return features, labels

def optimize_naive_bayes(X_train, y_train):
    """优化朴素贝叶斯模型"""
    print("\n优化朴素贝叶斯模型...")
    # 使用校准后的朴素贝叶斯（概率更准确）
    model = CalibratedClassifierCV(GaussianNB(), cv=5)
    model.fit(X_train, y_train)
    return model

def build_and_evaluate_naive_bayes(model, X_test, y_test):
    """评估朴素贝叶斯模型"""
    print("\n评估朴素贝叶斯模型...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"测试集准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化部分
    plt.figure(figsize=(12, 5))
    
    # 子图1：混淆矩阵
    plt.subplot(1, 2, 1)
    ConfusionMatrixDisplay(cm, display_labels=['良性', '恶性']).plot(cmap='Blues')
    plt.title(f"混淆矩阵 (准确率: {acc:.3f})")
    
    # 子图2：概率分布
    plt.subplot(1, 2, 2)
    probas = model.predict_proba(X_test)[:, 1]
    for label in [0, 1]:
        plt.hist(probas[y_test==label], bins=20, alpha=0.7,
                label=['良性', '恶性'][label])
    plt.title("预测概率分布")
    plt.xlabel("恶性概率")
    plt.ylabel("样本数")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('naive_bayes_results.png')
    plt.show()

def run_naive_bayes():
    """运行朴素贝叶斯模型的完整流程"""
    # 加载和预处理数据
    X, y = load_and_preprocess()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 训练模型
    model = optimize_naive_bayes(X_train, y_train)
    
    # 评估模型
    build_and_evaluate_naive_bayes(model, X_test, y_test)

if __name__ == "__main__":
    run_naive_bayes()