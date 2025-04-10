#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install ucimlrepo


# In[1]:


from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def validity_ratio(data, k):
    kmeans = KMeans(n_clusters=k, random_state=1024, n_init=10)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    
    intra_cluster_distances = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) == 0:
            continue
        centroid = centroids[i]
        avg_distance = np.sum(np.linalg.norm(cluster_points - centroid, axis=1))
        intra_cluster_distances.append(avg_distance)
    
    inter_cluster_distances = []
    for i in range(k):
        for j in range(i + 1, k):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            inter_cluster_distances.append(distance)
    
    min_inter_distance = min(inter_cluster_distances) if inter_cluster_distances else 1
    
    return np.sum(intra_cluster_distances) / min_inter_distance / 569

def find_best_k(data, max_k=30):   
    ratios = []
    K_values = range(2, max_k+1)
    
    for k in K_values:
        ratio = validity_ratio(data.to_numpy(), k)
        ratios.append(ratio)
    
    plt.plot(K_values, ratios, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Validity Ratio')
    plt.title('Optimal K using Validity Ratio')
    plt.show()
    
    best_k = K_values[np.argmin(ratios)]
    return best_k

'''
def calculate_davg(X, labels, centers):
    K = centers.shape[0]  # 聚类的数量
    N = X.shape[0]  # 数据点的数量
    davg = 0.0

    for k in range(K):
        # 取出属于第 k 个聚类的数据点
        cluster_points = X[labels == k]
        # 计算到聚类中心的距离
        distances = np.linalg.norm(cluster_points - centers[k], axis=1)
        # 计算该聚类的平均距离
        if len(cluster_points) > 0:  # 避免空聚类
            davg += np.sum(distances) / len(cluster_points)

    return davg / K  # 返回所有聚类的平均距离

'''
'''
def calculate_dmin(centers):
    K = centers.shape[0]
    dmin = np.inf  # 初始化为无穷大

    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            # 计算两个聚类中心之间的距离
            distance = np.linalg.norm(centers[k1] - centers[k2])
            dmin = min(dmin, distance)  # 更新最小距离

    return dmin
'''

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets  

'''
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
X.describe()
min_max_values = {}
prefixes = ['radius', 'texture', 'perimeter', 'area', 
            'smoothness', 'compactness', 'concavity', 
            'concave_points', 'symmetry', 'fractal_dimension']
for prefix in prefixes:
    min_max_values[prefix] = {
        'min': min(X.filter(like=prefix).min()),
        'max': max(X.filter(like=prefix).max())
    }
    
min_max_df = pd.DataFrame({
    'Attribute': [f"{prefix} (min)" for prefix in prefixes] + [f"{prefix} (max)" for prefix in prefixes],
    'Value': [min_max_values[prefix]['min'] for prefix in prefixes] + [min_max_values[prefix]['max'] for prefix in prefixes]
})
X.describe().filter(like='texture')
X.filter(like='radius').mean()

print(min_max_df)
'''
'''
validity_ratios = []
K_range = range(2, 31)
X_scaled = X_minmax
for K in K_range:
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # 计算有效性比率
    davg = calculate_davg(X_scaled, labels, centers)    
    # 计算聚类内距离
    dmin = calculate_dmin(centers)
    
    validity_ratio = dmin / davg
    validity_ratios.append(validity_ratio)

plt.figure(figsize=(10, 6))
plt.plot(K_range, validity_ratios, marker='o', linestyle='-', color='b')
plt.title('Validity Ratio vs. Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Validity Ratio')
plt.xticks(K_range)
#plt.ylim(0.7, 1.5)
plt.grid(False) 
plt.axvline(x=3, color='r', linestyle='--')  # 绘制 K=3 的垂直线
plt.annotate('Local Minimum', xy=(3, validity_ratios[1]), xytext=(4, validity_ratios[1] + 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()
'''

y = y.squeeze()

if y.dtype == 'object':  
    y = y.map({'B': 0, 'M': 1})
X['Diagnosis'] = y

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X.drop(columns=['Diagnosis'])), columns=X.columns[:-1])
X_scaled['Diagnosis'] = X['Diagnosis'].values

data_benign = X_scaled[X_scaled['Diagnosis'] == 0].drop(columns=['Diagnosis'])
data_malignant = X_scaled[X_scaled['Diagnosis'] == 1].drop(columns=['Diagnosis'])

best_k_benign = find_best_k(data_benign)
best_k_malignant = find_best_k(data_malignant)

if best_k_benign:
    kmeans_benign = KMeans(n_clusters=best_k_benign, random_state=42, n_init=10)
    X_benign = data_benign.copy()
    X_benign['Cluster'] = kmeans_benign.fit_predict(data_benign)
    print(f'Optimal K（Benign）：{best_k_benign}')

if best_k_malignant:
    kmeans_malignant = KMeans(n_clusters=best_k_malignant, random_state=42, n_init=10)
    X_malignant = data_malignant.copy()
    X_malignant['Cluster'] = kmeans_malignant.fit_predict(data_malignant)
    print(f'Optimal K（Malignant）：{best_k_malignant}')


# In[3]:


from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X_original = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets  

y = y.squeeze()

if y.dtype == 'object':  
    y = y.map({'B': 0, 'M': 1})
X_original['Diagnosis'] = y

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_original.drop(columns=['Diagnosis'])), columns=X_original.columns[:-1])
X_scaled['Diagnosis'] = X_original['Diagnosis'].values

data_benign = X_scaled[X_scaled['Diagnosis'] == 0].drop(columns=['Diagnosis'])
data_malignant = X_scaled[X_scaled['Diagnosis'] == 1].drop(columns=['Diagnosis'])

kmeans_benign = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_malignant = KMeans(n_clusters=3, random_state=42, n_init=10)

X_benign = data_benign.copy()
X_malignant = data_malignant.copy()

X_benign['Cluster'] = kmeans_benign.fit_predict(data_benign)
X_malignant['Cluster'] = kmeans_malignant.fit_predict(data_malignant)

centroids_benign_scaled = kmeans_benign.cluster_centers_
centroids_malignant_scaled = kmeans_malignant.cluster_centers_

centroids_benign = scaler.inverse_transform(centroids_benign_scaled)
centroids_malignant = scaler.inverse_transform(centroids_malignant_scaled)

feature_x = 'radius1'
feature_y = 'radius2'
feature_z = 'texture2'

colors = ['red', 'blue', 'green']

X_benign_original = X_original[X_original['Diagnosis'] == 0]
X_malignant_original = X_original[X_original['Diagnosis'] == 1]

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
for i in range(3):
    cluster_points = X_benign_original[X_benign['Cluster'] == i]
    ax1.scatter(cluster_points[feature_x], cluster_points[feature_y], cluster_points[feature_z], 
                color=colors[i], label=f'Cluster {i}')
ax1.scatter(centroids_benign[:, X_original.columns.get_loc(feature_x)], 
            centroids_benign[:, X_original.columns.get_loc(feature_y)], 
            centroids_benign[:, X_original.columns.get_loc(feature_z)], 
            marker='s', color='black', s=200, label='Centroids')

ax1.set_xlabel(feature_x)
ax1.set_ylabel(feature_y)
ax1.set_zlabel(feature_z)
ax1.set_title("Benign Tumor Clusters")
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
for i in range(3):
    cluster_points = X_malignant_original[X_malignant['Cluster'] == i]
    ax2.scatter(cluster_points[feature_x], cluster_points[feature_y], cluster_points[feature_z], 
                color=colors[i], label=f'Cluster {i}')
ax2.scatter(centroids_malignant[:, X_original.columns.get_loc(feature_x)], 
            centroids_malignant[:, X_original.columns.get_loc(feature_y)], 
            centroids_malignant[:, X_original.columns.get_loc(feature_z)], 
            marker='s', color='black', s=200, label='Centroids')

ax2.set_xlabel(feature_x)
ax2.set_ylabel(feature_y)
ax2.set_zlabel(feature_z)
ax2.set_title("Malignant Tumor Clusters")
ax2.legend()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




