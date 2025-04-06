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

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets  

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




