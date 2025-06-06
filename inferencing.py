import streamlit as st
import pickle
from modelling import NetflixRecommender

# --- Load preprocessed dataframe from pickle ---
@st.cache_resource
def load_model():
    # Load DataFrame dari pickle
    with open("netflix_df.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Buat objek NetflixRecommender dan inject df
    model = NetflixRecommender("dummy.csv")  # path tidak dipakai
    model.df = df
    model.build_model()
    return model

recommender = load_model()

# --- Streamlit UI ---
st.title("🎬 Netflix Content-Based Recommender")

# Text input judul
selected_title = st.text_input("Masukkan judul film/series (case-sensitive):")

# Tombol tampilkan
if st.button("Tampilkan Rekomendasi") and selected_title:
    result = recommender.get_recommendations(selected_title, topn=5)

    # Jika hasil berupa DataFrame, tampilkan
    if isinstance(result, str):
        st.warning(result)  # kalau judul tidak ditemukan
    else:
        st.success(f"Rekomendasi mirip dengan: **{selected_title}**")
        for _, row in result.iterrows():
            st.markdown(f"### 🎥 {row['title']} ({row['release_year']})")
            st.markdown(f"**Tipe**: {row['type']}  \n**Genre**: {', '.join(row['genres'])}")
            st.markdown(f"_{row['description']}_")
            st.markdown("---")


