import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

try:
    df_label = pd.read_csv('DataCleanLabel.csv')
    df_onehot = pd.read_csv('DataCleanOneHot.csv')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'DataCleanLabel.csv' and 'DataCleanOneHot.csv' are in the directory.")
    exit()

df_combined = pd.DataFrame({
    "FCVC": df_onehot['FCVC'],
    "FAF": df_onehot['FAF'],
    "FAVC": df_label['FAVC'],  # Label Encoded (0=No, 1=Yes likely)
    "CAEC_Sometimes": df_onehot['CAEC_Sometimes'],
    "CAEC_no": df_onehot['CAEC_no'],
    "CAEC_Frequently": df_onehot['CAEC_Frequently'],
    "NCP": df_onehot['NCP'],
    "SCC_yes": df_onehot['SCC_yes'],
    "SMOKE_yes": df_onehot['SMOKE_yes'],
})

X = df_combined
print(f"Training with features: {list(X.columns)}")

print("Scaling data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Training K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

print("Training GMM...")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

print("Training Hierarchical (via KNN proxy)...")
hira = AgglomerativeClustering(n_clusters=3)
hira_labels = hira.fit_predict(X_scaled)

hira_predictor = KNeighborsClassifier(n_neighbors=5)
hira_predictor.fit(X_scaled, hira_labels)

print("Saving models to .pkl files...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(gmm, 'gmm_model.pkl')
joblib.dump(hira_predictor, 'hira_model.pkl')

print("Done! Files created.")