import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Obesity Clustering Predictor", page_icon="🥗")

# 1. Load Models
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    gmm = joblib.load('gmm_model.pkl')
    hira = joblib.load('hira_model.pkl')
    return scaler, kmeans, gmm, hira

scaler, kmeans, gmm, hira = load_models()

# 2. App Title and Config
st.title("Obesity Level Clustering System")
st.write("Enter your lifestyle details below to identify your health cluster.")

# 3. Sidebar - Model Selection
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Clustering Algorithm:",
    ("K-Means", "Hierarchical (HIRA)", "Gaussian Mixture (GMM)")
)

# 4. Input Form
st.subheader("User Lifestyle Data")

col1, col2 = st.columns(2)

with col1:
    # FAVC: Frequent consumption of high caloric food (Yes=1, No=0)
    favc_input = st.selectbox("Do you eat high caloric food frequently?", ["No", "Yes"])
    favc = 1 if favc_input == "Yes" else 0

    # SMOKE: (Yes=1, No=0)
    smoke_input = st.selectbox("Do you smoke?", ["No", "Yes"])
    smoke = 1 if smoke_input == "Yes" else 0

    # SCC: Calories consumption monitoring (Yes=1, No=0)
    scc_input = st.selectbox("Do you monitor your calories?", ["No", "Yes"])
    scc = 1 if scc_input == "Yes" else 0

    # CAEC: Consumption of food between meals
    # Mapping assumes LabelEncoder order: Always=0, Frequently=1, Sometimes=2, no=3 
    # (Check your encoder mappings, but usually alphabetical)
    # Let's assume ordinal for simplicity: 0=No, 1=Sometimes, 2=Freq, 3=Always
    caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    caec_input = st.selectbox("Consumption of food between meals?", list(caec_map.keys()))
    caec = caec_map[caec_input]

with col2:
    # FAF: Physical activity frequency (0 to 3)
    faf = st.slider("Physical activity frequency (days/week)?", 0.0, 3.0, 1.0, 0.1)

    # FCVC: Frequency of consumption of vegetables (1 to 3)
    fcvc = st.slider("Vegetable consumption frequency?", 1.0, 3.0, 2.0, 0.1)

    # NCP: Number of main meals (1 to 4)
    ncp = st.slider("Number of main meals daily?", 1.0, 4.0, 3.0, 0.1)

# 5. Prediction Logic
if st.button("Predict Cluster"):
    # Prepare input array
    # Order must match training: 'FAVC', 'CAEC', 'FAF', 'NCP', 'SMOKE', 'SCC', 'FCVC'
    input_data = np.array([[favc, caec, faf, ncp, smoke, scc, fcvc]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    cluster = None
    
    if model_choice == "K-Means":
        cluster = kmeans.predict(input_scaled)[0]
        st.info(f"Model Used: K-Means (6 Clusters)")
        
    elif model_choice == "Gaussian Mixture (GMM)":
        cluster = gmm.predict(input_scaled)[0]
        st.info(f"Model Used: GMM (3 Components)")
        
    elif model_choice == "Hierarchical (HIRA)":
        cluster = hira.predict(input_scaled)[0]
        st.info(f"Model Used: Hierarchical (4 Clusters)")

    # 6. Result Display
    st.success(f"You belong to **Cluster {cluster}**")
    
    # Interpretation (Placeholder - update based on your Interpretasi.ipynb)
    st.markdown("---")
    st.write("**Cluster Interpretation:**")
    if model_choice == "K-Means":
         st.write(f"Cluster {cluster} in K-Means typically represents...")
    elif model_choice == "Gaussian Mixture (GMM)":
         st.write(f"Group {cluster} in GMM indicates...")
    else:
         st.write(f"Cluster {cluster} in Hierarchical clustering suggests...")