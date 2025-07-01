import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from quick_clean_example import quick_clean_tmdb_data
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🎬 Bollywood Movie Posters",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for movie poster app
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .poster-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    .selected-poster {
        text-align: center;
        margin: 2rem 0;
    }
    .recommendation-text {
        font-size: 1.3rem;
        color: #4ecdc4;
        text-align: center;
        margin: 2rem 0;
        font-weight: bold;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the movie dataset"""
    df = quick_clean_tmdb_data('bollywood_movies.csv')
    
    # Create genres string for display
    df['genres_str'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else 'Unknown')
    
    # Prepare content for TF-IDF
    df['content'] = df.apply(lambda x: f"{x['overview']} {x['genres_str']} {x['director']}", axis=1)
    
    return df

def get_movie_recommendations(title, df, num_recommendations=5):
    """Get movie recommendations using TF-IDF and cosine similarity"""
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get movie index
    movie_indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if title not in movie_indices:
        return None
    
    idx = movie_indices[title]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get movie indices
    movie_indices_list = [i[0] for i in sim_scores]
    
    # Create recommendations dataframe
    recommendations = df.iloc[movie_indices_list][['title', 'director', 'genres_str', 
                                                 'release_year', 'imdb_rating', 
                                                 'box_office_collection', 'poster_url', 'overview']].copy()
    recommendations['similarity_score'] = [score[1] for score in sim_scores]
    
    return recommendations

def create_movie_poster_card(movie_data, rank=None, is_selected=False):
    """Create a movie poster card"""
    poster_url = movie_data.get('poster_url', 'https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster')
    
    # Different styling for selected vs recommended movies
    if is_selected:
        border_color = "#ff6b6b"
        background = "linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%)"
        badge_text = "SELECTED"
        badge_color = "#fff"
    else:
        border_color = "#4ecdc4"
        background = "linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%)"
        badge_text = f"#{rank}"
        badge_color = "#fff"
    
    card_html = f"""
    <div style="
        background: {background};
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
        text-align: center;
    " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
        
        <!-- Poster -->
        <div style="position: relative; margin-bottom: 1rem;">
            <img src="{poster_url}" 
                 style="width: 100%; max-width: 250px; height: 350px; object-fit: cover; border-radius: 15px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);"
                 onerror="this.onerror=null; this.src='https://via.placeholder.com/250x350/cccccc/666666?text=No+Poster';">
            
            <!-- Badge -->
            <div style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); color: {badge_color}; padding: 8px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">
                {badge_text}
            </div>
            
            <!-- Rating -->
            <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8); color: white; padding: 6px 12px; border-radius: 15px; font-size: 0.8rem;">
                ⭐ {movie_data['imdb_rating']}
            </div>
        </div>
        
        <!-- Movie Info -->
        <div>
            <h3 style="margin: 0 0 0.5rem 0; font-size: 1.3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                {movie_data['title']}
            </h3>
            <p style="margin: 0 0 0.5rem 0; font-size: 1rem; opacity: 0.9;">
                {movie_data['release_year']} • {movie_data['director']}
            </p>
            <p style="margin: 0 0 1rem 0; font-size: 0.9rem; opacity: 0.8;">
                {movie_data['genres_str']}
            </p>
            <p style="margin: 0; font-size: 1rem; font-weight: bold;">
                💰 ₹{movie_data['box_office_collection']:.0f} Cr
            </p>
            {f'<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;"><strong>Match:</strong> {movie_data["similarity_score"]:.1%}</p>' if 'similarity_score' in movie_data else ''}
        </div>
    </div>
    """
    
    return card_html

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Bollywood Movie Posters 🎭</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Discover movies through stunning posters!</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🎬 Loading Bollywood movies...'):
        df = load_and_process_data()
    
    # Sidebar
    st.sidebar.header("🎯 Movie Selection")
    
    # Movie selection
    movie_titles = sorted(df['title'].tolist())
    selected_movie = st.sidebar.selectbox(
        "🎬 Choose your favorite movie:",
        movie_titles,
        help="Select a movie to get similar recommendations"
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "📊 Number of recommendations:",
        min_value=3,
        max_value=8,
        value=5,
        help="How many similar movies to recommend"
    )
    
    # Get selected movie data
    selected_movie_data = df[df['title'] == selected_movie].iloc[0]
    
    # Display selected movie
    st.markdown('<div class="recommendation-text">🎯 Your Selected Movie</div>', unsafe_allow_html=True)
    
    col_selected = st.columns(1)[0]
    with col_selected:
        st.markdown(create_movie_poster_card(selected_movie_data, is_selected=True), unsafe_allow_html=True)
    
    # Get and display recommendations
    st.markdown('<div class="recommendation-text">🎬 Movies You Might Love</div>', unsafe_allow_html=True)
    
    with st.spinner('🔍 Finding similar movies...'):
        recommendations = get_movie_recommendations(selected_movie, df, num_recommendations)
    
    if recommendations is not None and len(recommendations) > 0:
        # Create columns for poster display
        num_cols = min(3, len(recommendations))
        cols = st.columns(num_cols)
        
        for idx, (_, movie) in enumerate(recommendations.iterrows()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(create_movie_poster_card(movie, idx + 1), unsafe_allow_html=True)
        
        # Movie details section
        with st.expander("📝 View Movie Details", expanded=False):
            st.markdown("### 🎬 Selected Movie Details")
            st.write(f"**Plot:** {selected_movie_data['overview']}")
            
            st.markdown("### 🎯 Recommended Movies Details")
            for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                st.markdown(f"**{idx}. {movie['title']}**")
                st.write(f"*Plot:* {movie['overview'][:200]}...")
                st.write("---")
    
    else:
        st.error("❌ No recommendations found for the selected movie.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: white; opacity: 0.7;">🎭 Bollywood Movie Poster Recommender • Made with ❤️ using Streamlit</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
