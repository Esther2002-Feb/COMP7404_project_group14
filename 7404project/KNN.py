import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

def optimize_knn(X_train, y_train):
    """优化KNN模型的超参数"""
    print("\n优化KNN超参数...")
    # 这里可以添加网格搜索或其他优化方法
    # 目前先使用默认参数
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def build_and_evaluate_knn(model, X_test, y_test):
    """评估KNN模型"""
    print("\n评估KNN模型...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"测试集准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=['良性', '恶性']).plot(cmap='Blues')
    plt.title(f"KNN模型评估 (准确率: {acc:.3f})")
    plt.tight_layout()
    plt.savefig('knn_results.png')
    plt.show()

def run_knn():
    """运行KNN模型的完整流程"""
    # 加载和预处理数据
    X, y = load_and_preprocess()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 训练模型
    model = optimize_knn(X_train, y_train)
    
    # 评估模型
    build_and_evaluate_knn(model, X_test, y_test)

if __name__ == "__main__":
    run_knn()