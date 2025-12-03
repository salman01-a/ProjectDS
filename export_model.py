import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

# 1. Load Data
# Ensure DataCleanLabel.csv is in the same directory
print("Loading data...")
df = pd.read_csv('CleanDataLabelAndOneHot.csv')
df_label = pd.read_csv('DataCleanLabel.csv') 
df_onehot = pd.read_csv('DataCleanOneHot.csv')

# 2. Select Features (Based on your notebooks)
features = [                                   
    'FAVC', 'CAEC_Sometimes', 'CAEC_no', 'CAEC_Frequently', 'FAF', 'NCP', 'SMOKE', 'SCC', 'FCVC'              
]


df_combined = pd.DataFrame({
    "FCVC": df_onehot['FCVC'],
    "FAF": df_onehot['FAF'],
    "FAVC": df_label['FAVC'],
    "CAEC_Sometimes": df_onehot['CAEC_Sometimes'],
    "CAEC_no": df_onehot['CAEC_no'],
    "CAEC_Frequently": df_onehot['CAEC_Frequently'],
    "NCP": df_onehot['NCP'],
    "SCC_yes": df_onehot['SCC_yes'],
    "SMOKE_yes": df_onehot['SMOKE_yes'],
})

X = df_combined

# 3. Scale Data
print("Scaling data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Models

# --- K-Means (Optimal K=6) ---
print("Training K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# --- GMM (Optimal K=3) ---
print("Training GMM...")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

# --- Hierarchical (Optimal K=4) ---
# Note: We train a KNN classifier to emulate the Hira model for prediction
print("Training Hierarchical (via KNN proxy)...")
hira = AgglomerativeClustering(n_clusters=4)
hira_labels = hira.fit_predict(X_scaled)

hira_predictor = KNeighborsClassifier(n_neighbors=5)
hira_predictor.fit(X_scaled, hira_labels)

# 5. Export Models and Scaler
print("Saving models to .pkl files...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(gmm, 'gmm_model.pkl')
joblib.dump(hira_predictor, 'hira_model.pkl')

print("Done! Files created: scaler.pkl, kmeans_model.pkl, gmm_model.pkl, hira_model.pkl")