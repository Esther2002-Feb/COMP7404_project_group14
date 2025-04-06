

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def optimize_svm(train_features, train_labels):
    """
    使用 Grid Search 优化 SVM 模型超参数
    """
    print("开始 SVM 超参数优化...")

    # 定义参数网格
    param_grid = {
        "C": [0.1, 1, 10, 100],  # 正则化参数
        "gamma": [0.1, 0.01, 0.001],  # 核函数系数
        "kernel": ["linear", "rbf", "poly", "sigmoid"],  # 核函数类型
        "degree": [2, 3, 4],  # 核函数为 'poly' 时的多项式次数
    }

    # 创建 SVM 分类器
    svm = SVC(random_state=42)

    # 调用 GridSearchCV
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring="accuracy", cv=3, verbose=2, n_jobs=1)
    grid_search.fit(train_features, train_labels)

    # 输出最优参数结果
    print("最优超参数组合: ", grid_search.best_params_)
    print("最佳交叉验证得分: ", grid_search.best_score_)

    # 返回优化后的模型
    return grid_search.best_estimator_