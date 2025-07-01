import streamlit as st
import pandas as pd
from quick_clean_example import quick_clean_tmdb_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="🎬 Movie Posters", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = quick_clean_tmdb_data('bollywood_movies.csv')
    df['genres_str'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    return df

def get_recommendations(selected_movie, df, num_recs=5):
    # Simple content-based filtering
    df['content'] = df['overview'] + ' ' + df['genres_str']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Get movie index
    idx = df[df['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# Main app
st.title("🎬 Bollywood Movie Recommendations")
st.write("Select a movie to get recommendations!")

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.header("🎯 Choose Movie")
    selected_movie = st.selectbox("Pick a movie:", df['title'].sort_values())
    num_recs = st.slider("Number of recommendations:", 3, 8, 5)

# Selected movie
selected_data = df[df['title'] == selected_movie].iloc[0]
st.subheader(f"🎯 Selected: {selected_movie}")

# Show poster and info
col1, col2 = st.columns([1, 2])
with col1:
    if 'poster_url' in selected_data:
        st.image(selected_data['poster_url'], width=200, caption=f"{selected_movie} ({int(selected_data['release_year'])})")
    else:
        st.write("📽️ No poster available")

with col2:
    st.write(f"**Director:** {selected_data['director']}")
    st.write(f"**Year:** {int(selected_data['release_year'])}")
    st.write(f"**Rating:** ⭐ {selected_data['imdb_rating']}/10")
    st.write(f"**Genres:** {selected_data['genres_str']}")
    st.write(f"**Box Office:** ₹{selected_data['box_office_collection']:.0f} Cr")

# Recommendations
st.subheader("🎬 Movies You Might Like")
recommendations = get_recommendations(selected_movie, df, num_recs)

# Display recommendations in columns
cols = st.columns(min(3, len(recommendations)))
for i, (_, movie) in enumerate(recommendations.iterrows()):
    with cols[i % len(cols)]:
        if 'poster_url' in movie:
            st.image(movie['poster_url'], width=150, caption=f"#{i+1}")
        st.write(f"**{movie['title']}** ({int(movie['release_year'])})")
        st.write(f"⭐ {movie['imdb_rating']}")
        st.write(f"💰 ₹{movie['box_office_collection']:.0f} Cr")
        st.write("---")

# Movie details
with st.expander("📖 Movie Details"):
    st.write(f"**{selected_movie} Plot:**")
    st.write(selected_data['overview'])
    
    st.write("**Recommended Movies:**")
    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
        st.write(f"{i}. **{movie['title']}** - {movie['overview'][:100]}...")
