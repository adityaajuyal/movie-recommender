import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from quick_clean_example import quick_clean_tmdb_data
import warnings
warnings.filterwarnings('ignore')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_eda_plots(df):
    """
    Create comprehensive EDA plots for movie dataset
    """
    print("Creating EDA visualizations...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribution of Movie Ratings
    plt.subplot(2, 3, 1)
    plt.hist(df['imdb_rating'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of IMDB Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('IMDB Rating')
    plt.ylabel('Number of Movies')
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_rating = df['imdb_rating'].mean()
    plt.axvline(mean_rating, color='red', linestyle='--', label=f'Mean: {mean_rating:.2f}')
    plt.legend()
    
    # 2. Most Popular Genres (Bar Plot)
    plt.subplot(2, 3, 2)
    all_genres = [genre for sublist in df['genres_list'] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    
    # Create bar plot
    bars = plt.bar(range(len(genre_counts)), genre_counts.values, color='lightcoral', alpha=0.8)
    plt.title('Most Popular Genres', fontsize=14, fontweight='bold')
    plt.xlabel('Genres')
    plt.ylabel('Number of Movies')
    plt.xticks(range(len(genre_counts)), genre_counts.index, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, genre_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # 3. Average Box Office Collection per Genre
    plt.subplot(2, 3, 3)
    
    # Create genre-revenue mapping
    genre_revenue = {}
    for idx, row in df.iterrows():
        for genre in row['genres_list']:
            if genre not in genre_revenue:
                genre_revenue[genre] = []
            genre_revenue[genre].append(row['box_office_collection'])
    
    # Calculate average revenue per genre
    avg_revenue = {genre: np.mean(revenues) for genre, revenues in genre_revenue.items()}
    avg_revenue_series = pd.Series(avg_revenue).sort_values(ascending=False)
    
    bars = plt.bar(range(len(avg_revenue_series)), avg_revenue_series.values, 
                   color='lightgreen', alpha=0.8)
    plt.title('Average Box Office Collection per Genre', fontsize=14, fontweight='bold')
    plt.xlabel('Genres')
    plt.ylabel('Average Collection (₹ Crores)')
    plt.xticks(range(len(avg_revenue_series)), avg_revenue_series.index, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_revenue_series.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'₹{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Correlation Heatmap
    plt.subplot(2, 3, 4)
    
    # Select numeric columns for correlation
    numeric_cols = ['runtime', 'box_office_collection', 'imdb_rating', 'release_year']
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 5. Box Office vs IMDB Rating Scatter Plot
    plt.subplot(2, 3, 5)
    scatter = plt.scatter(df['imdb_rating'], df['box_office_collection'], 
                         c=df['runtime'], cmap='viridis', alpha=0.7, s=100)
    plt.colorbar(scatter, label='Runtime (minutes)')
    plt.title('Box Office vs IMDB Rating', fontsize=14, fontweight='bold')
    plt.xlabel('IMDB Rating')
    plt.ylabel('Box Office Collection (₹ Crores)')
    plt.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['imdb_rating'], df['box_office_collection'], 1)
    p = np.poly1d(z)
    plt.plot(df['imdb_rating'], p(df['imdb_rating']), "r--", alpha=0.8, linewidth=2)
    
    # 6. Movies Released Over Years
    plt.subplot(2, 3, 6)
    year_counts = df['release_year'].value_counts().sort_index()
    plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6)
    plt.title('Movies Released Over Years', fontsize=14, fontweight='bold')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Movies')
    plt.grid(alpha=0.3)
    
    # Fill area under curve
    plt.fill_between(year_counts.index, year_counts.values, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bollywood_eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def create_detailed_plots(df):
    """
    Create additional detailed plots
    """
    print("\nCreating detailed analysis plots...")
    
    # Create second figure with additional plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top Directors by Average Rating
    director_stats = df.groupby('director').agg({
        'imdb_rating': 'mean',
        'box_office_collection': 'mean',
        'title': 'count'
    }).round(2)
    director_stats.columns = ['Avg_Rating', 'Avg_Collection', 'Movie_Count']
    
    # Filter directors with more than 1 movie
    top_directors = director_stats[director_stats['Movie_Count'] >= 1].sort_values('Avg_Rating', ascending=False).head(10)
    
    top_directors['Avg_Rating'].plot(kind='barh', ax=axes[0,0], color='orange', alpha=0.8)
    axes[0,0].set_title('Top Directors by Average IMDB Rating', fontweight='bold')
    axes[0,0].set_xlabel('Average IMDB Rating')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # 2. Language Distribution
    language_counts = df['language'].value_counts()
    axes[0,1].pie(language_counts.values, labels=language_counts.index, autopct='%1.1f%%', 
                  startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0,1].set_title('Movies by Language', fontweight='bold')
    
    # 3. Runtime Distribution
    axes[1,0].hist(df['runtime'], bins=12, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title('Distribution of Movie Runtime', fontweight='bold')
    axes[1,0].set_xlabel('Runtime (minutes)')
    axes[1,0].set_ylabel('Number of Movies')
    axes[1,0].axvline(df['runtime'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["runtime"].mean():.0f} min')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. Top Grossing Movies
    top_grossers = df.nlargest(10, 'box_office_collection')
    bars = axes[1,1].barh(range(len(top_grossers)), top_grossers['box_office_collection'], 
                          color='gold', alpha=0.8)
    axes[1,1].set_yticks(range(len(top_grossers)))
    axes[1,1].set_yticklabels(top_grossers['title'], fontsize=9)
    axes[1,1].set_title('Top 10 Grossing Movies', fontweight='bold')
    axes[1,1].set_xlabel('Box Office Collection (₹ Crores)')
    axes[1,1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_grossers['box_office_collection'])):
        axes[1,1].text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, 
                       f'₹{value:.0f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('bollywood_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_dataset_summary(df):
    """
    Print comprehensive dataset summary
    """
    print("\n" + "="*60)
    print("BOLLYWOOD MOVIE DATASET - COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total Movies: {len(df)}")
    print(f"   • Time Period: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")
    print(f"   • Dataset Shape: {df.shape}")
    
    print(f"\n🎭 GENRE ANALYSIS:")
    all_genres = [genre for sublist in df['genres_list'] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    print(f"   • Total Unique Genres: {len(genre_counts)}")
    print(f"   • Most Popular Genre: {genre_counts.index[0]} ({genre_counts.iloc[0]} movies)")
    print(f"   • Top 5 Genres: {', '.join(genre_counts.head().index.tolist())}")
    
    print(f"\n⭐ RATING STATISTICS:")
    print(f"   • Average IMDB Rating: {df['imdb_rating'].mean():.2f}")
    print(f"   • Highest Rated: {df.loc[df['imdb_rating'].idxmax(), 'title']} ({df['imdb_rating'].max():.1f})")
    print(f"   • Lowest Rated: {df.loc[df['imdb_rating'].idxmin(), 'title']} ({df['imdb_rating'].min():.1f})")
    
    print(f"\n💰 BOX OFFICE ANALYSIS:")
    print(f"   • Total Collection: ₹{df['box_office_collection'].sum():.0f} Crores")
    print(f"   • Average Collection: ₹{df['box_office_collection'].mean():.0f} Crores")
    print(f"   • Highest Grosser: {df.loc[df['box_office_collection'].idxmax(), 'title']} (₹{df['box_office_collection'].max():.0f} Cr)")
    
    print(f"\n🎬 LANGUAGE DISTRIBUTION:")
    lang_dist = df['language'].value_counts()
    for lang, count in lang_dist.items():
        print(f"   • {lang}: {count} movies ({count/len(df)*100:.1f}%)")
    
    print(f"\n🎯 RUNTIME STATISTICS:")
    print(f"   • Average Runtime: {df['runtime'].mean():.0f} minutes")
    print(f"   • Shortest Movie: {df.loc[df['runtime'].idxmin(), 'title']} ({df['runtime'].min():.0f} min)")
    print(f"   • Longest Movie: {df.loc[df['runtime'].idxmax(), 'title']} ({df['runtime'].max():.0f} min)")
    
    print(f"\n🎪 TOP DIRECTORS:")
    director_stats = df.groupby('director').agg({
        'imdb_rating': 'mean',
        'box_office_collection': 'mean',
        'title': 'count'
    }).round(2)
    director_stats.columns = ['Avg_Rating', 'Avg_Collection', 'Movie_Count']
    top_directors = director_stats.sort_values('Avg_Rating', ascending=False).head(5)
    
    for director, stats in top_directors.iterrows():
        print(f"   • {director}: {stats['Movie_Count']} movies, Avg Rating: {stats['Avg_Rating']:.1f}")

def main():
    """
    Main function to run EDA analysis
    """
    try:
        # Load the dataset
        print("Loading Bollywood movie dataset...")
        df = quick_clean_tmdb_data('bollywood_movies.csv')
        
        # Print dataset summary
        print_dataset_summary(df)
        
        # Create visualizations
        create_eda_plots(df)
        create_detailed_plots(df)
        
        print(f"\n✅ EDA Analysis Complete!")
        print(f"📊 Visualizations saved as:")
        print(f"   • bollywood_eda_analysis.png")
        print(f"   • bollywood_detailed_analysis.png")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: bollywood_movies.csv not found!")
        print("Please run quick_clean_example.py first to create the dataset.")
        return None
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the analysis
    df = main()
