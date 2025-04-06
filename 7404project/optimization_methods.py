import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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

def optimize_knn(train_features, train_labels):
    # 定义 KNN 模型
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': np.arange(1, 31),  # k 值范围
        'weights': ['uniform', 'distance'],  # 权重选项
        'metric': ['euclidean', 'manhattan']  # 距离度量
    }

    # 网格搜索
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features, train_labels)

    # 返回最佳模型
    return grid_search.best_estimator_

def optimize_random_forest(train_features, train_labels):
    rf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [50, 100, 200],  # 树的数量
        'max_depth': [None, 10, 20, 30],  # 最大深度
        'min_samples_split': [2, 5, 10],  # 分裂所需的最小样本数
        'min_samples_leaf': [1, 2, 4]     # 叶子节点的最小样本数
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features, train_labels)

    return grid_search.best_estimator_

def optimize_naive_bayes(train_features, train_labels):
    nb = GaussianNB()

    param_grid = {
        'var_smoothing': np.logspace(0, -9, num=100)  # 方差平滑参数
    }

    grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features, train_labels)

    return grid_search.best_estimator_

def optimize_logistic_regression(train_features, train_labels):
    lr = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': np.logspace(-4, 4, 20),  # 正则化参数
        'solver': ['lbfgs', 'liblinear'],  # 求解器
        'penalty': ['l2', 'none']  # 惩罚类型
    }

    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features, train_labels)

    return grid_search.best_estimator_
