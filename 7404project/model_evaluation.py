# model_evaluation.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_and_evaluate_model(best_model, test_features, test_labels):
    """
    构建并评估模型
    - 利用优化后的模型进行预测
    - 测试集性能评估（准确率、分类报告、混淆矩阵）
    """
    print("测试集性能评估...")

    # 测试集预测
    predictions = best_model.predict(test_features)

    # 输出性能结果
    print(f"测试集准确率: {accuracy_score(test_labels, predictions):.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels, predictions))
    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, predictions))