import tensorflow as tf  
import numpy as np  
from sklearn.datasets import load_breast_cancer  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  

class MLP:  
    def __init__(self, alpha, batch_size, node_size, num_classes, num_features):  

        self.alpha = alpha  
        self.batch_size = batch_size  
        self.node_size = node_size  
        self.num_classes = num_classes  
        self.num_features = num_features  
        self._build_model()  

    def _build_model(self):  

        # 使用 Keras Sequential API 构建模型  
        self.model = tf.keras.Sequential()  
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.num_features,)))  

        # 添加隐藏层  
        for nodes in self.node_size:  
            self.model.add(tf.keras.layers.Dense(nodes, activation='relu'))  

        # 输出层  
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))  

        # 编译模型  
        self.model.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),  
            loss='sparse_categorical_crossentropy',  # 适用于非独热编码标签  
            metrics=['accuracy']  
        )  

    def train(self, num_epochs, log_path, train_data, train_size, test_data, test_size, result_path):  

        train_features, train_labels = train_data  
        test_features, test_labels = test_data  

        # 设置 TensorBoard 回调  
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)  

        # 训练模型  
        print("开始训练模型...")  
        self.model.fit(  
            train_features,  
            train_labels,  
            validation_data=(test_features, test_labels),  
            epochs=num_epochs,  
            batch_size=self.batch_size,  
            callbacks=[tensorboard_callback],  
            verbose=2  
        )  

        # 在测试集上进行预测  
        print("训练结束，开始测试...")  
        predictions = self.model.predict(test_features)  
        predicted_labels = np.argmax(predictions, axis=1)  

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
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_CLASSES = 2  
    NUM_NODES = [500, 500, 500]
    NUM_EPOCHS = 100

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

    # 创建 MLP 模型  
    model = MLP(  
        alpha=LEARNING_RATE,  
        batch_size=BATCH_SIZE,  
        node_size=NUM_NODES,  
        num_classes=NUM_CLASSES,  
        num_features=features.shape[1],  
    )  

    # 设置日志和结果存储路径  
    log_path = "./logs"  
    result_path = "./results.npz"  

    # 训练和评估模型  
    model.train(  
        num_epochs=NUM_EPOCHS,  
        log_path=log_path,  
        train_data=[train_features, train_labels],  
        train_size=train_features.shape[0],  
        test_data=[test_features, test_labels],  
        test_size=test_features.shape[0],  
        result_path=result_path,  
    )  


if __name__ == "__main__":  
    main()  