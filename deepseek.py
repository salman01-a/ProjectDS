import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("ObesityDataSet.csv")

# Preprocessing untuk clustering
df_cluster = df.copy()

# Encode categorical variables
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 
                   'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
le = LabelEncoder()
for col in categorical_cols:
    df_cluster[col] = le.fit_transform(df_cluster[col])

# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cluster)

# Menentukan jumlah cluster optimal dengan Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Berdasarkan elbow, pilih k=4 atau k=5
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Tambahkan cluster labels ke dataframe
df['Cluster'] = clusters

# Analisis karakteristik setiap cluster
cluster_profiles = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Weight': 'mean', 
    'Height': 'mean',
    'family_history_with_overweight': lambda x: x.value_counts().index[0],
    'FAVC': lambda x: x.value_counts().index[0],
    'FCVC': 'mean',
    'NCP': 'mean',
    'CAEC': lambda x: x.value_counts().index[0],
    'FAF': 'mean',
    'TUE': 'mean',
    'NObeyesdad': lambda x: x.value_counts().index[0]
}).round(2)

print("PROFIL SETIAP CLUSTER:")
print(cluster_profiles)

# Visualisasi dengan PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                     c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Visualisasi Cluster dengan PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analisis distribusi obesitas per cluster
plt.figure(figsize=(12, 6))
cluster_obesity = pd.crosstab(df['Cluster'], df['NObeyesdad'], normalize='index')
cluster_obesity.plot(kind='bar', stacked=True)
plt.title('Distribusi Tingkat Obesitas di Setiap Cluster')
plt.ylabel('Proporsi')
plt.legend(title='Tingkat Obesitas', bbox_to_anchor=(1.05, 1))
plt.show()

# Jika silhouette score < 0.3, coba teknik ini:

# 1. Feature selection untuk clustering
important_features = ['Age', 'Weight', 'Height', 'FCVC', 'NCP', 'FAF', 'TUE']
df_selected = df_cluster[important_features]
scaled_selected = scaler.fit_transform(df_selected)

# 2. Coba algoritma clustering lain
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_features)
if len(np.unique(dbscan_labels)) > 1:
    dbscan_score = silhouette_score(scaled_features, dbscan_labels)
    print(f"DBSCAN Silhouette Score: {dbscan_score:.4f}")

# Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(scaled_features)
agglo_score = silhouette_score(scaled_features, agglo_labels)
print(f"Agglomerative Clustering Score: {agglo_score:.4f}")

# 3. PCA sebelum clustering
pca = PCA(n_components=0.95)  # Pertahankan 95% variance
pca_features = pca.fit_transform(scaled_features)
kmeans_pca = KMeans(n_clusters=4, random_state=42)
pca_labels = kmeans_pca.fit_predict(pca_features)
pca_score = silhouette_score(pca_features, pca_labels)
print(f"PCA + KMeans Score: {pca_score:.4f}")