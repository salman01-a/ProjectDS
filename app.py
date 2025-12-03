import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Clustering Tingkat Obesitas", page_icon="🍕", layout="centered")

# 1. Memuat Model & Scaler
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        gmm = joblib.load('gmm_model.pkl')
        hira = joblib.load('hira_model.pkl')
        return scaler, kmeans, gmm, hira
    except FileNotFoundError:
        st.error("File model tidak ditemukan. export model terlebih dahulu.")
        return None, None, None, None

scaler, kmeans, gmm, hira = load_models()

if scaler is not None:
    # 2. Header
    st.title("🍕 Clustering Gaya Hidup Obesitas")
    st.markdown("Prediksi cluster kesehatan Anda berdasarkan kebiasaan gaya hidup.")

    # 3. Sidebar - Pemilihan Model
    st.sidebar.header("Pengaturan")
    model_choice = st.sidebar.selectbox(
        "Pilih Algoritma",
        ("K-Means (3 Cluster)", "Gaussian Mixture (3 Cluster)", "Hierarchical (4 Cluster)")
    )

    # 4. Formulir Input
    st.subheader("Kebiasaan Gaya Hidup Anda")
    
    col1, col2 = st.columns(2)

    with col1:
        # FAVC (Label Encoded: No=0, Yes=1)
        favc_input = st.radio("Apakah Anda sering makan makanan tinggi kalori?", ["Tidak", "Ya"], horizontal=True)
        favc_val = 1 if favc_input == "Tidak" else 0

        # SMOKE (OneHot: SMOKE_yes: 1/0)
        smoke_input = st.radio("Apakah Anda merokok?", ["Tidak", "Ya"], horizontal=True)
        smoke_val = 1 if smoke_input == "Ya" else 0

        # SCC (OneHot: SCC_yes: 1/0)
        scc_input = st.radio("Apakah Anda memantau asupan kalori Anda?", ["Tidak", "Ya"], horizontal=True)
        scc_val = 1 if scc_input == "Ya" else 0

        # CAEC (OneHot: Sometimes, no, Frequently)
        caec_options = ["Kadang-kadang", "Sering", "Selalu", "Tidak"]
        caec_input_raw = st.selectbox("Seberapa sering Anda nyemil di antara waktu makan utama?", caec_options)
        
        # Mapping kembali ke nilai asli untuk logika
        if "Kadang-kadang" in caec_input_raw: caec_input = "Sometimes"
        elif "Sering" in caec_input_raw: caec_input = "Frequently"
        elif "Selalu" in caec_input_raw: caec_input = "Always"
        else: caec_input = "no"

    with col2:
        # FAF (Numerik)
        faf_val = st.slider("Seberapa sering Anda beraktivitas fisik (hari/minggu)?", 0.0, 3.0, 1.0, 0.1)

        # FCVC (Numerik)
        fcvc_val = st.slider("Seberapa sering Anda makan sayur (skala 1-3)?", 1.0, 3.0, 2.0, 0.1)

        # NCP (Numerik)
        ncp_val = st.slider("Berapa kali Anda makan besar sehari (1-4)?", 1.0, 4.0, 3.0, 1.0)

    # 5. Proses Input untuk Prediksi
    if st.button("Prediksi Cluster"):
        
        # --- Logika One-Hot Encoding ---
        caec_sometimes = 1 if caec_input == "Sometimes" else 0
        caec_no = 1 if caec_input == "no" else 0
        caec_frequently = 1 if caec_input == "Frequently" else 0
        
        # --- Membuat DataFrame ---
        input_data = pd.DataFrame({
            "FCVC": [fcvc_val],
            "FAF": [faf_val],
            "FAVC": [favc_val],
            "CAEC_Sometimes": [caec_sometimes],
            "CAEC_no": [caec_no],
            "CAEC_Frequently": [caec_frequently],
            "NCP": [ncp_val],
            "SCC_yes": [scc_val],
            "SMOKE_yes": [smoke_val]
        })

        # Skala Input
        input_scaled = scaler.transform(input_data)

        # Prediksi
        cluster = None
        algo_name = ""

        if "K-Means" in model_choice:
            cluster = kmeans.predict(input_scaled)[0]
            algo_name = "K-Means"
        elif "Gaussian Mixture" in model_choice:
            cluster = gmm.predict(input_scaled)[0]
            algo_name = "GMM"
        elif "Hierarchical" in model_choice:
            cluster = hira.predict(input_scaled)[0]
            algo_name = "Hierarchical (4 Cluster)"

        # Menampilkan Hasil
        st.success(f"Berdasarkan kebiasaan Anda, Anda termasuk dalam **Cluster {cluster}**")
        st.caption(f"Model yang digunakan: {algo_name}")

        # --- Logika Interpretasi ---
        st.divider()
        st.subheader("Apa artinya?")

        # Interpretasi untuk Model 3 Cluster (K-Means, GMM)
        if "Hierarchical" not in model_choice:
            if cluster == 0:
                st.info("""
                **Cluster 0: Profil 'Sehat / Aktif'**
                * **Kebiasaan:** Anda kemungkinan sering berolahraga dan makan sayur secara teratur.
                * **Diet:** Anda cenderung menghindari makanan tinggi kalori yang berlebihan.
                * **Status:** Sering dikaitkan dengan Berat Badan Normal atau individu yang bugar secara fisik.
                """)
            elif cluster == 1:
                st.warning("""
                **Cluster 1: Profil 'Moderat / Beresiko'**
                * **Kebiasaan:** Aktivitas fisik Anda rendah hingga sedang.
                * **Diet:** Anda mungkin mengonsumsi makanan tinggi kalori atau ngemil sesekali.
                * **Status:** Sering dikaitkan dengan status Kelebihan Berat Badan atau kebiasaan kesehatan yang campur aduk.
                """)
            elif cluster == 2:
                st.error("""
                **Cluster 2: Profil 'Tidak Sehat'**
                * **Kebiasaan:** Aktivitas fisik sangat rendah.
                * **Diet:** Konsumsi tinggi makanan berkalori tinggi dan sering ngemil.
                * **Status:** Cocok dengan karakteristik yang sering ditemukan dalam kategori Obesitas.
                """)

        # Interpretasi untuk Model 4 Cluster (Hierarchical)
        else:
            if cluster == 0:
                st.info("""
                **Cluster 0: Kelompok 'Aktif & Sehat'**
                * **Kebiasaan:** Aktivitas fisik tinggi dan konsumsi sayur yang baik.
                * **Status:** Kemungkinan Berat Badan Normal.
                """)
            elif cluster == 1:
                st.warning("""
                **Cluster 1: Kelompok 'Kebiasaan Moderat'**
                * **Kebiasaan:** Aktivitas dan diet rata-rata.
                * **Status:** Kemungkinan Kelebihan Berat Badan atau sedikit berisiko.
                """)
            elif cluster == 2:
                st.error("""
                **Cluster 2: Kelompok 'Resiko Tinggi'**
                * **Kebiasaan:** Aktivitas rendah, asupan kalori tinggi.
                * **Status:** Kemungkinan Obesitas (Tipe I atau II).
                """)
            elif cluster == 3:
                st.error("""
                **Cluster 3: Kelompok 'Ekstrem / Spesifik'**
                * **Kebiasaan:** Kelompok ini sering menangkap ujung ekstrem dari spektrum (misalnya, sangat sering ngemil + nol aktivitas).
                * **Status:** Kemungkinan Obesitas Tipe III atau pengguna dengan pola makan yang sangat spesifik.
                """)