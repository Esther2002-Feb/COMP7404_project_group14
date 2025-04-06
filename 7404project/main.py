# main.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_processing import data_exploration, feature_selection
from hyperparameter_optimization_MLP import optimize_mlp
from hyperparameter_optimization_SVM import optimize_svm
from model_evaluation import build_and_evaluate_model

# 新增算法调用
from KNN import optimize_knn, build_and_evaluate_knn
from random_forest import optimize_random_forest, build_and_evaluate_random_forest
from naive_bayes import optimize_naive_bayes, build_and_evaluate_naive_bayes
from logistic_regression import optimize_logistic_regression, build_and_evaluate_logistic_regression
# class MLP:
#     def __init__(self, alpha, batch_size, node_size, num_classes, num_features):
#         """
#         初始化多层感知机 (MLP) 模型
#
#         参数：
#         - alpha: 学习率
#         - batch_size: 批量大小
#         - node_size: 隐藏层的神经元个数（列表）
#         - num_classes: 输出类别数
#         - num_features: 输入特征数
#         """
#         self.alpha = alpha
#         self.batch_size = batch_size
#         self.node_size = node_size
#         self.num_classes = num_classes
#         self.num_features = num_features
#         self._build_model()
#
#     def _build_model(self):
#         """
#         构建 MLP 模型
#         """
#         # 使用 Keras Sequential API 构建模型
#         self.model = tf.keras.Sequential()
#         self.model.add(tf.keras.layers.InputLayer(input_shape=(self.num_features,)))
#
#         # 添加隐藏层
#         for nodes in self.node_size:
#             self.model.add(tf.keras.layers.Dense(nodes, activation='relu'))
#
#         # 输出层
#         self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
#
#         # 编译模型
#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
#             loss='sparse_categorical_crossentropy',  # 适用于非独热编码标签
#             metrics=['accuracy']
#         )
#
#     def train(self, num_epochs, log_path, train_data, train_size, test_data, test_size, result_path):
#         """
#         训练模型
#
#         参数：
#         - num_epochs: 训练轮数
#         - log_path: TensorBoard 日志保存路径
#         - train_data: 训练数据（[train_features, train_labels]）
#         - train_size: 训练数据大小
#         - test_data: 测试数据（[test_features, test_labels]）
#         - test_size: 测试数据大小
#         - result_path: 模型预测结果保存路径
#         """
#         train_features, train_labels = train_data
#         test_features, test_labels = test_data
#
#         # 设置 TensorBoard 回调
#         tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
#
#         # 训练模型
#         print("开始训练模型...")
#         self.model.fit(
#             train_features,
#             train_labels,
#             validation_data=(test_features, test_labels),
#             epochs=num_epochs,
#             batch_size=self.batch_size,
#             callbacks=[tensorboard_callback],
#             verbose=2
#         )
#
#         # 在测试集上进行预测
#         print("训练结束，开始测试...")
#         predictions = self.model.predict(test_features)
#         predicted_labels = np.argmax(predictions, axis=1)
#
#         # 保存预测结果
#         np.savez(
#             result_path,
#             actual=test_labels,
#             predicted=predicted_labels
#         )
#
#         # 打印模型评估结果
#         print(f"测试集准确率: {accuracy_score(test_labels, predicted_labels):.4f}")
#         print("\n分类报告:")
#         print(classification_report(test_labels, predicted_labels))
#         print("\n混淆矩阵:")
#         print(confusion_matrix(test_labels, predicted_labels))


# 主程序逻辑
def main():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    features = data.data
    labels = data.target
    feature_names = data.feature_names

    # 数据归一化（将特征缩放到 [0, 1] 之间）
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # 数据探索与特征分析
    positive_corr = data_exploration(features, labels)

    # 使用相关性分析进行特征选择
    # features, selected_indices = drop_redundant_features(features, positive_corr)

    # 使用 RFE 进行特征选择（选择 15 个重要特征）
    features, selected_indices = feature_selection(features, labels, num_features_to_select=15)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    print(f"选择的重要特征: {selected_feature_names}")

    # 划分训练集与测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # ===== 原有算法调用 =====
    print("\n" + "="*50)
    print("=== 开始优化和评估 MLP 模型 ===")
    best_model_mlp = optimize_mlp(train_features, train_labels)
    build_and_evaluate_model(best_model_mlp, test_features, test_labels)

    print("\n" + "="*50)
    print("=== 开始优化和评估 SVM 模型 ===")
    best_model_svm = optimize_svm(train_features, train_labels)
    build_and_evaluate_model(best_model_svm, test_features, test_labels)

    # ===== 新增算法调用 =====
    print("\n" + "="*50)
    print("=== 开始优化和评估 KNN 模型 ===")
    best_model_knn = optimize_knn(train_features, train_labels)
    build_and_evaluate_knn(best_model_knn, test_features, test_labels)

    print("\n" + "="*50)
    print("=== 开始优化和评估 Random Forest 模型 ===")
    best_model_rf = optimize_random_forest(train_features, train_labels)
    build_and_evaluate_random_forest(best_model_rf, test_features, test_labels, selected_feature_names)

    print("\n" + "="*50)
    print("=== 开始优化和评估 Naive Bayes 模型 ===")
    best_model_nb = optimize_naive_bayes(train_features, train_labels)
    build_and_evaluate_naive_bayes(best_model_nb, test_features, test_labels)

    print("\n" + "="*50)
    print("=== 开始优化和评估 Logistic Regression 模型 ===")
    best_model_lr = optimize_logistic_regression(train_features, train_labels)
    build_and_evaluate_logistic_regression(best_model_lr, test_features, test_labels)

if __name__ == "__main__":
    main()

    # # 超参数
    # BATCH_SIZE = 128
    # LEARNING_RATE = 0.0001
    # NUM_CLASSES = 2
    # NUM_NODES = [500, 500, 500]
    # NUM_EPOCHS = 100
    #
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    #
    # # 划分训练集和测试集 (70% 训练，30% 测试)
    # train_features, test_features, train_labels, test_labels = train_test_split(
    #     features, labels, test_size=0.2, stratify=labels, random_state=42
    # )

    # # 创建 MLP 模型
    # model = MLP(
    #     alpha=LEARNING_RATE,
    #     batch_size=BATCH_SIZE,
    #     node_size=NUM_NODES,
    #     num_classes=NUM_CLASSES,
    #     num_features=features.shape[1],
    # )
    # # 设置日志和结果存储路径
    # log_path = "./logs"
    # result_path = "./results.npz"
    #
    # # 训练和评估模型
    # model.train(
    #     num_epochs=NUM_EPOCHS,
    #     log_path=log_path,
    #     train_data=[train_features, train_labels],
    #     train_size=train_features.shape[0],
    #     test_data=[test_features, test_labels],
    #     test_size=test_features.shape[0],
    #     result_path=result_path,
    # )


if __name__ == "__main__":
    main()