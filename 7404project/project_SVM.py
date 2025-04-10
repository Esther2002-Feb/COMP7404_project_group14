#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 


# In[2]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_B = X[y['Diagnosis'] == 'B']
X_M = X[y['Diagnosis'] == 'M']

features_B = X_B.drop(columns=['Diagnosis'] if 'Diagnosis' in X_B.columns else [])
features_M = X_M.drop(columns=['Diagnosis'] if 'Diagnosis' in X_M.columns else [])

scaler = StandardScaler()
X_B_normalized = scaler.fit_transform(features_B)
X_M_normalized = scaler.fit_transform(features_M)

kmeans_B = KMeans(n_clusters=3, random_state=42)
X_B['cluster'] = kmeans_B.fit_predict(X_B_normalized)


kmeans_M = KMeans(n_clusters=3, random_state=42)
X_M['cluster'] = kmeans_M.fit_predict(X_M_normalized)

centers_B_normalized = kmeans_B.cluster_centers_
centers_M_normalized = kmeans_M.cluster_centers_

centers_B_original = scaler.inverse_transform(centers_B_normalized)
centers_M_original = scaler.inverse_transform(centers_M_normalized)

centers_B_df = pd.DataFrame(centers_B_original, columns=features_B.columns)
centers_M_df = pd.DataFrame(centers_M_original, columns=features_M.columns)

def get_cluster_min_max(data, cluster_col, features):
    cluster_stats = {}
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        min_values = cluster_data[features].min()
        max_values = cluster_data[features].max()
        cluster_stats[cluster] = {'min': min_values, 'max': max_values}
    return cluster_stats

cluster_stats_B = get_cluster_min_max(X_B, 'cluster', features_B.columns)
cluster_stats_M = get_cluster_min_max(X_M, 'cluster', features_M.columns)

centers_B_df.to_csv('cluster_centers_B.csv', index=False)
centers_M_df.to_csv('cluster_centers_M.csv', index=False)

cluster_stats_B_df = pd.DataFrame(cluster_stats_B).T
cluster_stats_M_df = pd.DataFrame(cluster_stats_M).T
cluster_stats_B_df.to_csv('cluster_stats_B.csv')
cluster_stats_M_df.to_csv('cluster_stats_M.csv')

def calculate_similarity_score(row, cluster_center, cluster_min, cluster_max):
    score = 0
    num_features = len(row)
    for feature in row.index:
        if feature in cluster_min.index and feature in cluster_max.index:
            if cluster_min[feature] <= row[feature] <= cluster_max[feature]:
                center_value = cluster_center[feature]
                row_value = row[feature]
                max_diff = max(abs(center_value - cluster_min[feature]), abs(center_value - cluster_max[feature]))
                feature_score = 1 - (abs(center_value - row_value) / max_diff)
                score += feature_score
    return score / num_features 

X['similarity_B1'] = 0
X['similarity_B2'] = 0
X['similarity_B3'] = 0
X['similarity_M1'] = 0
X['similarity_M2'] = 0
X['similarity_M3'] = 0

for i, cluster_center in centers_B_df.iterrows():
    cluster_min = cluster_stats_B[i]['min']
    cluster_max = cluster_stats_B[i]['max']
    X[f'similarity_B{i+1}'] = X.apply(lambda row: calculate_similarity_score(row, cluster_center, cluster_min, cluster_max), axis=1)

for i, cluster_center in centers_M_df.iterrows():
    cluster_min = cluster_stats_M[i]['min']
    cluster_max = cluster_stats_M[i]['max']
    X[f'similarity_M{i+1}'] = X.apply(lambda row: calculate_similarity_score(row, cluster_center, cluster_min, cluster_max), axis=1)

X.to_csv('data_with_similarity_scores.csv', index=False)

print(X.head())


# In[5]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_similarity = X[['similarity_B1', 'similarity_B2', 'similarity_B3', 'similarity_M1', 'similarity_M2', 'similarity_M3']]

X_similarity = scaler.fit_transform(X_similarity)
y = y['Diagnosis'].map({'B': 0, 'M': 1})

svm_classifier = SVC(kernel='sigmoid', random_state=42)  

cv_scores = cross_val_score(svm_classifier, X_similarity, y, cv=10, scoring='accuracy')

print("10 Folds Cross-Validation Accuracy：", cv_scores)
print("10 Folds Cross-Validation Average Accuracy：", cv_scores.mean())

svm_classifier.fit(X_similarity, y)


y_pred = svm_classifier.predict(X_similarity)
accuracy = accuracy_score(y, y_pred)
print("Whole Dataset Accuracy：", accuracy)


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(cv_scores)
plt.title('10-fold accuracy box plot')
plt.xlabel('Cross-Validation Scores')

plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Accuracy on Full Dataset: {accuracy:.2f}')
plt.legend()

plt.show()


# In[ ]:




