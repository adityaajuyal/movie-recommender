import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from quick_clean_example import quick_clean_tmdb_data
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movie_indices = None
        self.tfidf_vectorizer = None
        
    def load_and_prepare_data(self, file_path):
        """
        Load and prepare the movie dataset for recommendation
        """
        print("Loading and preparing movie dataset...")
        
        # Load the cleaned dataset
        self.df = quick_clean_tmdb_data(file_path)
        
        # Add taglines if not present (for demonstration)
        if 'tagline' not in self.df.columns:
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
            self.df['tagline'] = taglines[:len(self.df)]
        
        # Create movie index mapping
        self.movie_indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        
        print(f"Dataset loaded: {len(self.df)} movies")
        return self.df
    
    def create_content_features(self):
        """
        Create combined content features from overview, genres, and taglines
        """
        print("Creating content features...")
        
        # Convert genres list to string
        self.df['genres_str'] = self.df['genres_list'].apply(lambda x: ' '.join(x) if x else '')
        
        # Create combined features
        # Weight different features differently
        self.df['content_features'] = (
            self.df['overview'].fillna('') + ' ' +
            self.df['genres_str'] + ' ' + self.df['genres_str'] + ' ' +  # Give genres more weight
            self.df['tagline'].fillna('') + ' ' +
            self.df['director'].fillna('') + ' ' +
            self.df['language'].fillna('')
        )
        
        # Clean the content features
        self.df['content_features'] = self.df['content_features'].str.lower()
        
        print("Content features created successfully!")
        
    def build_tfidf_matrix(self):
        """
        Build TF-IDF matrix from content features
        """
        print("Building TF-IDF matrix...")
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform the content features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['content_features'])
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print("TF-IDF matrix built successfully!")
        
    def compute_similarity_matrix(self):
        """
        Compute cosine similarity matrix
        """
        print("Computing cosine similarity matrix...")
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"Similarity matrix shape: {self.cosine_sim.shape}")
        print("Cosine similarity matrix computed successfully!")
        
    def get_recommendations(self, title, num_recommendations=5):
        """
        Get movie recommendations based on content similarity
        """
        try:
            # Get movie index
            if title not in self.movie_indices:
                print(f"Movie '{title}' not found in dataset!")
                print("Available movies:")
                for i, movie in enumerate(self.df['title'].head(10)):
                    print(f"  {i+1}. {movie}")
                return None
            
            idx = self.movie_indices[title]
            
            # Get similarity scores for all movies
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Sort movies by similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies (excluding the input movie itself)
            sim_scores = sim_scores[1:num_recommendations+1]
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Create recommendations dataframe
            recommendations = self.df.iloc[movie_indices][['title', 'director', 'genres_str', 
                                                          'imdb_rating', 'box_office_collection', 
                                                          'release_year', 'language']].copy()
            
            # Add similarity scores
            recommendations['similarity_score'] = [score[1] for score in sim_scores]
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return None
    
    def display_recommendations(self, title, recommendations):
        """
        Display recommendations in a formatted way
        """
        if recommendations is None:
            return
            
        print(f"\n🎬 TOP 5 MOVIES SIMILAR TO '{title.upper()}':")
        print("=" * 80)
        
        for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{i}. 🎭 {movie['title']} ({movie['release_year']})")
            print(f"   🎬 Director: {movie['director']}")
            print(f"   🎪 Genres: {movie['genres_str']}")
            print(f"   🌍 Language: {movie['language']}")
            print(f"   ⭐ IMDB Rating: {movie['imdb_rating']}/10")
            print(f"   💰 Box Office: ₹{movie['box_office_collection']:.0f} Cr")
            print(f"   📊 Similarity Score: {movie['similarity_score']:.3f}")
            print("-" * 60)
    
    def analyze_recommendations(self, title):
        """
        Analyze and visualize recommendations
        """
        recommendations = self.get_recommendations(title)
        
        if recommendations is None:
            return
        
        # Display recommendations
        self.display_recommendations(title, recommendations)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Recommendation Analysis for "{title}"', fontsize=16, fontweight='bold')
        
        # 1. Similarity Scores
        axes[0,0].barh(range(len(recommendations)), recommendations['similarity_score'], 
                       color='skyblue', alpha=0.8)
        axes[0,0].set_yticks(range(len(recommendations)))
        axes[0,0].set_yticklabels(recommendations['title'], fontsize=10)
        axes[0,0].set_xlabel('Similarity Score')
        axes[0,0].set_title('Content Similarity Scores')
        axes[0,0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(recommendations['similarity_score']):
            axes[0,0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        
        # 2. IMDB Ratings Comparison
        movies_to_compare = [title] + recommendations['title'].tolist()
        ratings_to_compare = [self.df[self.df['title'] == title]['imdb_rating'].iloc[0]] + \
                            recommendations['imdb_rating'].tolist()
        
        colors = ['red'] + ['lightgreen'] * len(recommendations)
        bars = axes[0,1].bar(range(len(movies_to_compare)), ratings_to_compare, 
                            color=colors, alpha=0.8)
        axes[0,1].set_xticks(range(len(movies_to_compare)))
        axes[0,1].set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in movies_to_compare], 
                                 rotation=45, ha='right')
        axes[0,1].set_ylabel('IMDB Rating')
        axes[0,1].set_title('IMDB Ratings Comparison')
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, rating in zip(bars, ratings_to_compare):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                          f'{rating:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Box Office Comparison
        collections_to_compare = [self.df[self.df['title'] == title]['box_office_collection'].iloc[0]] + \
                                recommendations['box_office_collection'].tolist()
        
        bars = axes[1,0].bar(range(len(movies_to_compare)), collections_to_compare, 
                            color=colors, alpha=0.8)
        axes[1,0].set_xticks(range(len(movies_to_compare)))
        axes[1,0].set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in movies_to_compare], 
                                 rotation=45, ha='right')
        axes[1,0].set_ylabel('Box Office Collection (₹ Cr)')
        axes[1,0].set_title('Box Office Collection Comparison')
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, collection in zip(bars, collections_to_compare):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                          f'₹{collection:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Genre Distribution
        all_genres = []
        for genres_str in recommendations['genres_str']:
            all_genres.extend(genres_str.split())
        
        genre_counts = pd.Series(all_genres).value_counts().head(8)
        
        axes[1,1].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', 
                     startangle=90)
        axes[1,1].set_title('Genre Distribution in Recommendations')
        
        plt.tight_layout()
        plt.savefig(f'recommendations_for_{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return recommendations
    
    def get_feature_importance(self, title):
        """
        Analyze which features contribute most to recommendations
        """
        if title not in self.movie_indices:
            print(f"Movie '{title}' not found!")
            return
        
        idx = self.movie_indices[title]
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for the movie
        movie_tfidf = self.tfidf_matrix[idx].toarray()[0]
        
        # Get top features
        top_features_idx = movie_tfidf.argsort()[-20:][::-1]
        top_features = [(feature_names[i], movie_tfidf[i]) for i in top_features_idx if movie_tfidf[i] > 0]
        
        print(f"\n🔍 TOP CONTENT FEATURES FOR '{title.upper()}':")
        print("=" * 50)
        for i, (feature, score) in enumerate(top_features[:10], 1):
            print(f"{i:2d}. {feature:<20} (Score: {score:.4f})")
        
        return top_features
    
    def evaluate_recommendations(self, test_movies=None):
        """
        Evaluate the recommendation system
        """
        if test_movies is None:
            test_movies = ['Dangal', '3 Idiots', 'Baahubali 2', 'PK', 'RRR']
        
        print("\n🧪 RECOMMENDATION SYSTEM EVALUATION:")
        print("=" * 60)
        
        for movie in test_movies:
            if movie in self.movie_indices:
                recommendations = self.get_recommendations(movie, 3)
                if recommendations is not None:
                    print(f"\n🎬 {movie}:")
                    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                        print(f"  {i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
    
    def build_complete_system(self, file_path):
        """
        Build the complete recommendation system
        """
        print("🚀 Building Content-Based Movie Recommendation System...")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(file_path)
        
        # Step 2: Create content features
        self.create_content_features()
        
        # Step 3: Build TF-IDF matrix
        self.build_tfidf_matrix()
        
        # Step 4: Compute similarity matrix
        self.compute_similarity_matrix()
        
        print("\n✅ Recommendation System Built Successfully!")
        print(f"📊 Ready to recommend from {len(self.df)} movies")
        print("🎯 Use get_recommendations() or analyze_recommendations() methods")
        
        return self

def main():
    """
    Main function to demonstrate the recommendation system
    """
    try:
        # Initialize recommender
        recommender = ContentBasedRecommender()
        
        # Build the system
        recommender.build_complete_system('bollywood_movies.csv')
        
        # Demo recommendations
        print("\n" + "="*80)
        print("🎬 MOVIE RECOMMENDATION SYSTEM DEMO")
        print("="*80)
        
        # Test with different movies
        test_movies = ['Dangal', '3 Idiots', 'Baahubali 2']
        
        for movie in test_movies:
            print(f"\n{'='*60}")
            print(f"🎭 ANALYZING RECOMMENDATIONS FOR: {movie}")
            print(f"{'='*60}")
            
            # Get and display recommendations
            recommendations = recommender.analyze_recommendations(movie)
            
            # Show feature importance
            recommender.get_feature_importance(movie)
            
            print("\n" + "="*60)
        
        # Evaluate system
        recommender.evaluate_recommendations()
        
        print(f"\n✅ Demo Complete!")
        print(f"📁 Recommendation visualizations saved as PNG files")
        
        return recommender
        
    except FileNotFoundError:
        print("❌ Error: bollywood_movies.csv not found!")
        print("Please run quick_clean_example.py first to create the dataset.")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the recommendation system
    recommender_system = main()
