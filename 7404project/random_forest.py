import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler
from data_processing import data_exploration, feature_selection

def load_and_preprocess():
    """加载并预处理数据"""
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    features = data.data
    labels = data.target
    feature_names = data.feature_names
    
    # 数据归一化
    #scaler = MinMaxScaler()
    #features = scaler.fit_transform(features)
    
    # 特征选择
    features, selected_indices = feature_selection(features, labels, num_features_to_select=15)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    return features, labels, selected_feature_names

def optimize_random_forest(X_train, y_train):
    """优化随机森林模型"""
    print("\n优化随机森林模型...")
    # 这里可以添加网格搜索或其他优化方法
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def build_and_evaluate_random_forest(model, X_test, y_test, feature_names):
    """评估随机森林模型"""
    print("\n评估随机森林模型...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"测试集准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化部分
    plt.figure(figsize=(15, 6))
    
    # 子图1：特征重要性
    plt.subplot(1, 2, 1)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # 显示最重要的10个特征
    plt.barh(range(len(indices)), importances[indices], color='lightgreen', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("特征重要性 Top 10")
    plt.xlabel("相对重要性")
    
    # 子图2：混淆矩阵
    plt.subplot(1, 2, 2)
    ConfusionMatrixDisplay(cm, display_labels=['良性', '恶性']).plot(cmap='Greens')
    plt.title(f"模型评估 (准确率: {acc:.3f})")
    
    plt.tight_layout()
    plt.savefig('random_forest_results.png')
    plt.show()

def run_random_forest():
    """运行随机森林模型的完整流程"""
    # 加载和预处理数据
    X, y, feature_names = load_and_preprocess()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 训练模型
    model = optimize_random_forest(X_train, y_train)
    
    # 评估模型
    build_and_evaluate_random_forest(model, X_test, y_test, feature_names)

if __name__ == "__main__":
    run_random_forest()