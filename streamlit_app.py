import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        padding: 1rem;
        margin: 0.5rem 0;
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
    /* Hide any code blocks or unwanted content */
    code, pre, .stCode {
        display: none !important;
    }
    /* Hide HTML comments that might be showing */
    .stMarkdown div[data-testid="stMarkdownContainer"] > div > div {
        line-height: 1.2;
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
    if 'genres_list' in df.columns:
        df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x) if x and isinstance(x, list) else '')
    else:
        # Fallback: create genres_str from raw genres column
        def parse_genres_fallback(genres_str):
            if pd.isna(genres_str):
                return ''
            try:
                import ast
                genres = ast.literal_eval(genres_str)
                return ' '.join([g['name'] for g in genres if isinstance(g, dict) and 'name' in g])
            except:
                return ''
        df['genres_str'] = df['genres'].apply(parse_genres_fallback)
    
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
    required_columns = ['title', 'director', 'imdb_rating', 'box_office_collection', 
                       'release_year', 'language', 'runtime', 'overview']
    
    # Add genres_str if it exists
    if 'genres_str' in df.columns:
        required_columns.insert(2, 'genres_str')  # Insert after director
    
    # Add poster_url if it exists
    if 'poster_url' in df.columns:
        required_columns.append('poster_url')
    
    recommendations = df.iloc[movie_indices_list][required_columns].copy()
    
    # Add similarity scores
    recommendations['similarity_score'] = [score[1] for score in sim_scores]
    
    return recommendations

def create_movie_card(movie_data, rank=None):
    """Create a visually appealing movie card with poster"""
    poster_url = movie_data.get('poster_url', 'https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster')
    
    # Clean the HTML and remove any potential code display issues
    card_html = f"""<div class="movie-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 0; margin: 1rem 0; color: white; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); overflow: hidden;">
        <div style="display: flex; height: 280px;">
            <div style="flex: 0 0 200px; position: relative;">
                <img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 15px 0 0 15px;" onerror="this.onerror=null; this.src='https://via.placeholder.com/200x280/cccccc/666666?text=No+Poster';">
                {f'<div style="position: absolute; top: 10px; left: 10px; background: #ff6b6b; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">#{rank}</div>' if rank else ''}
            </div>
            <div style="flex: 1; padding: 1.5rem; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h3 style="margin: 0 0 1rem 0; font-size: 1.4rem; color: #fff;">{movie_data['title']} ({int(movie_data['release_year'])})</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 1rem; font-size: 0.9rem;">
                        <div><strong>🎬 Director:</strong> {movie_data['director']}</div>
                        <div><strong>🌍 Language:</strong> {movie_data['language']}</div>
                        <div><strong>⭐ IMDB:</strong> {movie_data['imdb_rating']}/10</div>
                        <div><strong>⏱️ Runtime:</strong> {int(movie_data['runtime'])} min</div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div><strong>🎭 Genres:</strong> {movie_data.get('genres_str', 'N/A')}</div>
                        <div style="margin-top: 0.5rem;"><strong>💰 Box Office:</strong> ₹{movie_data['box_office_collection']:.0f} Cr</div>
                    </div>
                </div>
                <div>
                    <div style="margin-bottom: 1rem; font-size: 0.9rem; line-height: 1.4; opacity: 0.9;"><strong>📝 Overview:</strong> {movie_data['overview'][:120]}{'...' if len(movie_data['overview']) > 120 else ''}</div>
                    {f'<div style="text-align: right; font-size: 0.8rem; opacity: 0.8;"><strong>📊 Similarity:</strong> {movie_data["similarity_score"]:.1%}</div>' if 'similarity_score' in movie_data else ''}
                </div>
            </div>
        </div>
    </div>"""
    
    return card_html

def create_hero_movie_card(movie_data):
    """Create a hero movie card for the selected movie"""
    poster_url = movie_data.get('poster_url', 'https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster')
    
    card_html = f"""<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%); border-radius: 20px; padding: 0; margin: 1.5rem 0; color: white; box-shadow: 0 12px 35px rgba(255, 107, 107, 0.3); overflow: hidden;">
        <div style="display: flex; height: 320px;">
            <div style="flex: 0 0 240px; position: relative;">
                <img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 20px 0 0 20px;" onerror="this.onerror=null; this.src='https://via.placeholder.com/240x320/cccccc/666666?text=No+Poster';">
                <div style="position: absolute; top: 15px; left: 15px; background: rgba(0,0,0,0.7); color: white; padding: 8px 15px; border-radius: 25px; font-weight: bold; font-size: 0.9rem;">🎯 SELECTED</div>
            </div>
            <div style="flex: 1; padding: 2rem; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <h2 style="margin: 0 0 1.5rem 0; font-size: 1.8rem; color: #fff; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎬 {movie_data['title']} ({int(movie_data['release_year'])})</h2>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; font-size: 1rem;">
                        <div><strong>🎬 Director:</strong> {movie_data['director']}</div>
                        <div><strong>🌍 Language:</strong> {movie_data['language']}</div>
                        <div><strong>⭐ IMDB Rating:</strong> {movie_data['imdb_rating']}/10</div>
                        <div><strong>⏱️ Runtime:</strong> {int(movie_data['runtime'])} min</div>
                    </div>
                    <div style="margin-bottom: 1.5rem;">
                        <div style="margin-bottom: 0.5rem;"><strong>🎭 Genres:</strong> {movie_data.get('genres_str', 'N/A')}</div>
                        <div><strong>💰 Box Office Collection:</strong> ₹{movie_data['box_office_collection']:.0f} Crores</div>
                    </div>
                </div>
                <div>
                    <div style="font-size: 1rem; line-height: 1.5; opacity: 0.95;"><strong>📝 Overview:</strong> {movie_data['overview'][:200]}{'...' if len(movie_data['overview']) > 200 else ''}</div>
                </div>
            </div>
        </div>
    </div>"""
    
    return card_html

def create_recommendation_charts(recommendations, input_movie_data):
    """Create visualization charts for recommendations"""
    
    # Prepare data for visualization
    movies = [input_movie_data['title']] + recommendations['title'].tolist()
    ratings = [input_movie_data['imdb_rating']] + recommendations['imdb_rating'].tolist()
    collections = [input_movie_data['box_office_collection']] + recommendations['box_office_collection'].tolist()
    years = [input_movie_data['release_year']] + recommendations['release_year'].tolist()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('IMDB Ratings Comparison', 'Box Office Collection', 
                       'Release Year Distribution', 'Similarity Scores'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Colors
    colors = ['#ff6b6b'] + ['#4ecdc4'] * len(recommendations)
    
    # 1. IMDB Ratings
    fig.add_trace(
        go.Bar(x=movies, y=ratings, marker_color=colors, name="IMDB Rating"),
        row=1, col=1
    )
    
    # 2. Box Office Collection
    fig.add_trace(
        go.Bar(x=movies, y=collections, marker_color=colors, name="Box Office"),
        row=1, col=2
    )
    
    # 3. Release Years
    fig.add_trace(
        go.Scatter(x=years, y=ratings, mode='markers+text', 
                  text=movies, textposition="top center",
                  marker=dict(size=12, color=colors), name="Year vs Rating"),
        row=2, col=1
    )
    
    # 4. Similarity Scores
    if len(recommendations) > 0:
        fig.add_trace(
            go.Bar(x=recommendations['title'], y=recommendations['similarity_score'], 
                  marker_color='#4ecdc4', name="Similarity"),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="📊 Recommendation Analysis Dashboard",
        title_x=0.5
    )
    
    # Update x-axis labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_poster_gallery(recommendations):
    """Create a poster gallery view for recommendations"""
    if len(recommendations) == 0:
        return "<p>No recommendations found.</p>"
    
    gallery_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin: 2rem 0;">'
    
    for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
        poster_url = movie.get('poster_url', 'https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster')
        
        gallery_html += f'''<div style="background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); position: relative;">
            <div style="position: relative;">
                <img src="{poster_url}" style="width: 100%; height: 300px; object-fit: cover;" onerror="this.onerror=null; this.src='https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster';">
                <div style="position: absolute; top: 10px; right: 10px; background: #ff6b6b; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; font-size: 0.8rem;">#{idx}</div>
                <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8); color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem;">⭐ {movie['imdb_rating']}</div>
                {f'<div style="position: absolute; bottom: 10px; right: 10px; background: rgba(0,0,0,0.8); color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem;">{movie["similarity_score"]:.1%}</div>' if 'similarity_score' in movie else ''}
            </div>
            <div style="padding: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #333; line-height: 1.2;">{movie['title'][:30]}{'...' if len(movie['title']) > 30 else ''}</h4>
                <p style="margin: 0 0 0.5rem 0; font-size: 0.8rem; color: #666;">{movie['release_year']} • {movie['language']}</p>
                <p style="margin: 0; font-size: 0.8rem; color: #888; line-height: 1.3;">{movie.get('genres_str', 'N/A')[:40]}{'...' if len(movie.get('genres_str', '')) > 40 else ''}</p>
            </div>
        </div>'''
    
    gallery_html += "</div>"
    return gallery_html

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Bollywood Movie Recommender 🎭</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Discover your next favorite Bollywood movie based on content similarity!</p>', unsafe_allow_html=True)
    
    # Load data and build system
    with st.spinner('🚀 Loading movie database and building recommendation system...'):
        df = load_data()
        df_processed, cosine_sim, movie_indices = build_recommendation_system(df)
    
    # Sidebar
    st.sidebar.header("🎯 Recommendation Settings")
    
    # Movie selection
    st.sidebar.subheader("🎬 Select a Movie")
    movie_titles = sorted(df['title'].tolist())
    
    # Search box with autocomplete
    selected_movie = st.sidebar.selectbox(
        "Choose a movie to get recommendations:",
        options=movie_titles,
        index=0,
        help="Select a movie from the dropdown to get similar movie recommendations"
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "📊 Number of Recommendations:",
        min_value=3,
        max_value=10,
        value=5,
        help="Choose how many similar movies you want to see"
    )
    
    # Advanced filters
    st.sidebar.subheader("🔧 Advanced Filters")
    
    # Language filter
    languages = ['All'] + sorted(df['language'].unique().tolist())
    selected_language = st.sidebar.selectbox("🌍 Language Filter:", languages)
    
    # Genre filter
    all_genres = set()
    for genres in df['genres_list']:
        all_genres.update(genres)
    genres_list = ['All'] + sorted(list(all_genres))
    selected_genre = st.sidebar.selectbox("🎭 Genre Filter:", genres_list)
    
    # Year range filter
    min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
    year_range = st.sidebar.slider(
        "📅 Release Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Display mode
    st.sidebar.subheader("🎨 Display Options")
    display_mode = st.sidebar.radio(
        "Choose display style:",
        ["🎬 Detailed Cards", "🖼️ Poster Gallery"],
        help="Choose how you want to view the movie recommendations"
    )
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎬 Selected Movie</h2>', unsafe_allow_html=True)
        
        # Get selected movie data
        selected_movie_data = df[df['title'] == selected_movie].iloc[0]
        
        # Display selected movie card
        st.markdown(create_hero_movie_card(selected_movie_data), unsafe_allow_html=True)
        
        # Movie statistics
        st.markdown('<h3 class="sub-header">📊 Dataset Statistics</h3>', unsafe_allow_html=True)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("🎬 Total Movies", len(df))
            st.metric("🎭 Unique Genres", len(all_genres))
        
        with col_stat2:
            st.metric("⭐ Avg Rating", f"{df['imdb_rating'].mean():.1f}")
            st.metric("🌍 Languages", len(df['language'].unique()))
    
    with col2:
        st.markdown('<h2 class="recommendation-header">🎯 Recommended Movies</h2>', unsafe_allow_html=True)
        
        # Get recommendations
        with st.spinner('🔍 Finding similar movies...'):
            recommendations = get_recommendations(
                selected_movie, cosine_sim, movie_indices, df_processed, num_recommendations
            )
        
        if recommendations is not None:
            # Apply filters
            filtered_recommendations = recommendations.copy()
            
            if selected_language != 'All':
                filtered_recommendations = filtered_recommendations[
                    filtered_recommendations['language'] == selected_language
                ]
            
            if selected_genre != 'All':
                if 'genres_str' in filtered_recommendations.columns:
                    filtered_recommendations = filtered_recommendations[
                        filtered_recommendations['genres_str'].str.contains(selected_genre, case=False, na=False)
                    ]
            
            filtered_recommendations = filtered_recommendations[
                (filtered_recommendations['release_year'] >= year_range[0]) &
                (filtered_recommendations['release_year'] <= year_range[1])
            ]
            
            if len(filtered_recommendations) > 0:
                # Display recommendations based on selected mode
                if display_mode == "🎬 Detailed Cards":
                    for idx, (_, movie) in enumerate(filtered_recommendations.iterrows(), 1):
                        st.markdown(create_movie_card(movie, idx), unsafe_allow_html=True)
                else:  # Poster Gallery
                    st.markdown(create_poster_gallery(filtered_recommendations), unsafe_allow_html=True)
                
                # Visualization section
                st.markdown('<h2 class="sub-header">📊 Visual Analysis</h2>', unsafe_allow_html=True)
                
                # Create and display charts
                fig = create_recommendation_charts(filtered_recommendations, selected_movie_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download recommendations
                st.markdown('<h3 class="sub-header">📥 Export Recommendations</h3>', unsafe_allow_html=True)
                
                # Prepare data for download
                download_columns = ['title', 'director', 'imdb_rating', 'box_office_collection', 
                                  'release_year', 'language', 'similarity_score']
                if 'genres_str' in filtered_recommendations.columns:
                    download_columns.insert(2, 'genres_str')
                
                download_data = filtered_recommendations[download_columns].copy()
                
                csv = download_data.to_csv(index=False)
                st.download_button(
                    label="📄 Download Recommendations as CSV",
                    data=csv,
                    file_name=f'recommendations_for_{selected_movie.replace(" ", "_")}.csv',
                    mime='text/csv'
                )
                
                # Poster gallery
                st.markdown('<h2 class="sub-header">🖼️ Movie Poster Gallery</h2>', unsafe_allow_html=True)
                st.markdown(create_poster_gallery(filtered_recommendations), unsafe_allow_html=True)
                
            else:
                st.warning("🔍 No movies found matching your filters. Please adjust the filter settings.")
        
        else:
            st.error("❌ Movie not found in the database!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">🎬 Built with ❤️ using Streamlit | Movie Recommendation System</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
