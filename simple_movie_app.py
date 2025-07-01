import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from quick_clean_example import quick_clean_tmdb_data
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🎬 Bollywood Movie Recommender",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ecdc4;
        margin-bottom: 1rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .selected-movie-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #ff6b6b;
    }
    .recommendation-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 2rem 0;
    }
    .similarity-badge {
        background: #2ecc71;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the movie dataset"""
    try:
        df = quick_clean_tmdb_data('bollywood_movies.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please run quick_clean_example.py first to create the dataset.")
        st.stop()

@st.cache_data
def build_recommendation_system(df):
    """Build the TF-IDF based recommendation system"""
    
    # Add taglines if not present
    if 'tagline' not in df.columns:
        taglines = [
            'The extraordinary true story of one man\'s journey to find his family',
            'All it takes is a little confidence!',
            'The conclusion',
            'A stranger in our planet',
            'Believe in the impossible',
            'The ring is his world',
            'Some mistakes can never be forgotten',
            'An epic love story',
            'He\'s not the only one who came back',
            'The biggest action entertainer of the year',
            'Shahid is Kabir Singh',
            'How\'s the Josh?',
            'Dream big, Aim high',
            'The fun has just begun',
            'Life mein thoda hasna zaroori hai',
            'The unsung warrior',
            'Rohit Shetty brings the biggest action entertainer',
            'A tale of fire',
            'Mother of all action films',
            'Wildfire'
        ]
        df['tagline'] = taglines[:len(df)]
    
    # Convert genres list to string
    df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x) if x else '')
    
    # Create combined features
    df['content_features'] = (
        df['overview'].fillna('') + ' ' +
        df['genres_str'] + ' ' + df['genres_str'] + ' ' +  # Give genres more weight
        df['tagline'].fillna('') + ' ' +
        df['director'].fillna('') + ' ' +
        df['language'].fillna('')
    )
    
    # Clean the content features
    df['content_features'] = df['content_features'].str.lower()
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    
    # Fit and transform
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content_features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create movie index mapping
    movie_indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return df, cosine_sim, movie_indices

def get_recommendations(title, cosine_sim, movie_indices, df, num_recommendations=5):
    """Get movie recommendations based on content similarity"""
    
    if title not in movie_indices:
        return None
    
    # Get movie index
    idx = movie_indices[title]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top similar movies (excluding the input movie)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get movie indices
    movie_indices_list = [i[0] for i in sim_scores]
    
    # Create recommendations dataframe
    recommendations = df.iloc[movie_indices_list][['title', 'director', 'genres_str', 
                                                  'imdb_rating', 'box_office_collection', 
                                                  'release_year', 'language', 'runtime', 'overview']].copy()
    
    # Add similarity scores
    recommendations['similarity_score'] = [score[1] for score in sim_scores]
    
    return recommendations

def create_selected_movie_card(movie_data):
    """Create a styled card for the selected movie"""
    
    # Handle genres - check if genres_str exists, otherwise create it from genres_list
    if 'genres_str' in movie_data:
        genres_display = movie_data['genres_str']
    elif 'genres_list' in movie_data and isinstance(movie_data['genres_list'], list):
        genres_display = ', '.join(movie_data['genres_list'])
    else:
        genres_display = 'N/A'
    
    card_html = f"""
    <div class="selected-movie-card">
        <h2>🎬 {movie_data['title']} ({int(movie_data['release_year'])})</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div><strong>🎬 Director:</strong> {movie_data['director']}</div>
            <div><strong>🌍 Language:</strong> {movie_data['language']}</div>
            <div><strong>🎭 Genres:</strong> {genres_display}</div>
            <div><strong>⏱️ Runtime:</strong> {int(movie_data['runtime'])} min</div>
            <div><strong>⭐ IMDB Rating:</strong> {movie_data['imdb_rating']}/10</div>
            <div><strong>💰 Box Office:</strong> ₹{movie_data['box_office_collection']:.0f} Cr</div>
        </div>
        <div style="margin: 1rem 0;">
            <strong>📝 Overview:</strong><br>
            {movie_data['overview']}
        </div>
    </div>
    """
    
    return card_html

def create_recommendation_card(movie_data, rank):
    """Create a styled recommendation card"""
    
    # Determine similarity level
    similarity = movie_data['similarity_score']
    if similarity > 0.15:
        sim_level = "Very High"
        sim_color = "#27ae60"
    elif similarity > 0.10:
        sim_level = "High" 
        sim_color = "#f39c12"
    elif similarity > 0.05:
        sim_level = "Medium"
        sim_color = "#e67e22"
    else:
        sim_level = "Low"
        sim_color = "#e74c3c"
    
    card_html = f"""
    <div class="movie-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3>🏆 {rank}. {movie_data['title']} ({int(movie_data['release_year'])})</h3>
            <div style="background: {sim_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                {sim_level} Match ({similarity:.3f})
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div><strong>🎬 Director:</strong> {movie_data['director']}</div>
            <div><strong>🌍 Language:</strong> {movie_data['language']}</div>
            <div><strong>🎭 Genres:</strong> {movie_data['genres_str']}</div>
            <div><strong>⏱️ Runtime:</strong> {int(movie_data['runtime'])} min</div>
            <div><strong>⭐ IMDB Rating:</strong> {movie_data['imdb_rating']}/10</div>
            <div><strong>💰 Box Office:</strong> ₹{movie_data['box_office_collection']:.0f} Cr</div>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong>📝 Overview:</strong><br>
            {movie_data['overview'][:200]}{'...' if len(movie_data['overview']) > 200 else ''}
        </div>
    </div>
    """
    
    return card_html

def create_comparison_chart(recommendations, selected_movie_data):
    """Create a simple comparison chart using matplotlib"""
    
    # Prepare data
    movies = ['Selected Movie'] + [f"Rec {i+1}" for i in range(len(recommendations))]
    ratings = [selected_movie_data['imdb_rating']] + recommendations['imdb_rating'].tolist()
    collections = [selected_movie_data['box_office_collection']] + recommendations['box_office_collection'].tolist()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors
    colors = ['#ff6b6b'] + ['#4ecdc4'] * len(recommendations)
    
    # IMDB Ratings comparison
    bars1 = ax1.bar(movies, ratings, color=colors, alpha=0.8)
    ax1.set_title('⭐ IMDB Ratings Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('IMDB Rating')
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rating in zip(bars1, ratings):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{rating:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Box Office comparison
    bars2 = ax2.bar(movies, collections, color=colors, alpha=0.8)
    ax2.set_title('💰 Box Office Collection Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Collection (₹ Crores)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, collection in zip(bars2, collections):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'₹{collection:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Bollywood Movie Recommender 🎭</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h3>🎯 Discover Your Next Favorite Bollywood Movie!</h3>
        <p>Our AI-powered recommendation system analyzes movie content, genres, directors, and more to find movies similar to your favorites.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and build system
    with st.spinner('🚀 Loading movie database and building recommendation system...'):
        df = load_data()
        df_processed, cosine_sim, movie_indices = build_recommendation_system(df)
    
    st.success('✅ Recommendation system ready!')
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎬 Select a Movie</h2>', unsafe_allow_html=True)
        
        # Movie selection
        movie_titles = sorted(df['title'].tolist())
        selected_movie = st.selectbox(
            "Choose a movie:",
            options=movie_titles,
            index=0,
            help="Select a movie from the dropdown to get similar movie recommendations"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "📊 Number of Recommendations:",
            min_value=3,
            max_value=8,
            value=5,
            help="Choose how many similar movies you want to see"
        )
        
        # Get selected movie data from processed dataframe
        selected_movie_data = df_processed[df_processed['title'] == selected_movie].iloc[0]
        
        # Display selected movie
        st.markdown(create_selected_movie_card(selected_movie_data), unsafe_allow_html=True)
        
        # Dataset statistics
        st.markdown('<h3 class="sub-header">📊 Dataset Info</h3>', unsafe_allow_html=True)
        
        # Create metrics in a nice layout
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("🎬 Total Movies", len(df))
            st.metric("⭐ Avg Rating", f"{df['imdb_rating'].mean():.1f}")
        
        with metric_col2:
            all_genres = set()
            for genres in df['genres_list']:
                all_genres.update(genres)
            st.metric("🎭 Unique Genres", len(all_genres))
            st.metric("🌍 Languages", len(df['language'].unique()))
    
    with col2:
        st.markdown('<h2 class="recommendation-header">🎯 Recommended Movies</h2>', unsafe_allow_html=True)
        
        # Get recommendations
        with st.spinner('🔍 Finding similar movies...'):
            recommendations = get_recommendations(
                selected_movie, cosine_sim, movie_indices, df_processed, num_recommendations
            )
        
        if recommendations is not None and len(recommendations) > 0:
            
            # Display recommendations
            for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                st.markdown(create_recommendation_card(movie, idx), unsafe_allow_html=True)
            
            # Visualization section
            st.markdown('<h3 class="sub-header">📊 Visual Comparison</h3>', unsafe_allow_html=True)
            
            try:
                # Create and display comparison chart
                fig = create_comparison_chart(recommendations, selected_movie_data)
                st.pyplot(fig)
            except Exception as e:
                st.warning("⚠️ Could not generate comparison chart.")
                st.write(f"Debug: {str(e)}")
            
            # Summary statistics
            st.markdown('<h3 class="sub-header">📈 Recommendation Summary</h3>', unsafe_allow_html=True)
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                avg_rating = recommendations['imdb_rating'].mean()
                st.metric("⭐ Avg Rating of Recommendations", f"{avg_rating:.1f}")
            
            with summary_col2:
                avg_collection = recommendations['box_office_collection'].mean()
                st.metric("💰 Avg Collection", f"₹{avg_collection:.0f} Cr")
            
            with summary_col3:
                avg_similarity = recommendations['similarity_score'].mean()
                st.metric("📊 Avg Similarity Score", f"{avg_similarity:.3f}")
            
            # Download section
            st.markdown('<h3 class="sub-header">📥 Export Recommendations</h3>', unsafe_allow_html=True)
            
            # Prepare data for download
            download_data = recommendations[['title', 'director', 'genres_str', 
                                          'imdb_rating', 'box_office_collection', 
                                          'release_year', 'language', 'similarity_score']].copy()
            
            csv = download_data.to_csv(index=False)
            st.download_button(
                label="📄 Download Recommendations as CSV",
                data=csv,
                file_name=f'recommendations_for_{selected_movie.replace(" ", "_")}.csv',
                mime='text/csv',
                help="Download the recommendations as a CSV file"
            )
            
        else:
            st.error("❌ Could not generate recommendations for this movie!")
    
    # Footer with additional information
    st.markdown("---")
    
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        **🔬 Content-Based Filtering:**
        
        Our recommendation system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to analyze:
        
        - 📝 **Movie Overviews**: Plot descriptions and storylines
        - 🎭 **Genres**: Action, Drama, Comedy, etc. (weighted higher)
        - 🎬 **Directors**: Directorial styles and preferences
        - 🌍 **Languages**: Regional cinema preferences
        - 🏷️ **Taglines**: Marketing descriptions
        
        **📊 Similarity Calculation:**
        - Uses **Cosine Similarity** to measure content overlap
        - Scores range from 0 (no similarity) to 1 (identical content)
        - Higher scores indicate more similar movies
        
        **🎯 Why Content-Based?**
        - Works well for new users (no cold start problem)
        - Provides explainable recommendations
        - Focuses on movie characteristics rather than user behavior
        """)
    
    st.markdown(
        '<p style="text-align: center; color: #666; margin-top: 2rem;">🎬 Built with ❤️ using Streamlit & Scikit-learn | Movie Recommendation System</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
