

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def optimize_mlp(train_features, train_labels):
    """
    使用 Grid Search 优化 MLP 模型超参数
    """
    print("开始 MLP 超参数优化...")

    # 定义参数网格
    param_grid = {
        "hidden_layer_sizes": [(100, 50), (100, 50, 50), (100, 100)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64, 128]
    }

    # 创建 MLP 分类器
    mlp = MLPClassifier(max_iter=500, random_state=42)

    # 调用 GridSearchCV
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring="accuracy", cv=3, verbose=2, n_jobs=1)
    grid_search.fit(train_features, train_labels)

    # 输出最优参数结果
    print("最优超参数组合: ", grid_search.best_params_)
    print("最佳交叉验证得分: ", grid_search.best_score_)

    # 返回优化后的模型
    return grid_search.best_estimator_