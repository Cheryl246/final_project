import streamlit as st
import pickle
from modelling import NetflixRecommender  # pastikan file class-nya bernama netflix_recommender.py

# --- Load preprocessed dataframe from pickle ---
@st.cache_resource
def load_model():
    with open('netflix_df.pkl', 'rb') as f:
        df = pickle.load(f)

    model = NetflixRecommender("dummy_path.csv")  # path tidak dipakai, karena df akan kita inject langsung
    model.df = df
    model.build_model()
    return model

# Load model once
recommender = load_model()

# --- Streamlit UI ---
st.title("🎬 Netflix Movie Recommender")
st.write("Rekomendasi film berdasarkan judul, genre, cast, dan sutradara.")

# Dropdown atau text input
selected_title = st.text_input("Pilih judul film:")

# Tombol rekomendasi
if st.button("Tampilkan Rekomendasi"):
    recommendations = recommender.get_recommendations(selected_title, topn=5)

    if isinstance(recommendations, str):  
        st.warning(recommendations)
    else:
        st.subheader(f"Rekomendasi mirip dengan **{selected_title}**:")
        for idx, row in recommendations.iterrows():
            st.markdown(f"### 🎥 {row['title']} ({row['release_year']})")
            st.markdown(f"**Tipe**: {row['type'].title()}")
            st.markdown(f"**Genre**: {', '.join(row['genres'])}")
            st.markdown(f"**Deskripsi**: {row['description']}")
            st.markdown("---")


