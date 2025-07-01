import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime

def load_and_clean_tmdb_data(movies_file_path, credits_file_path=None):
    """
    Load and clean TMDB 5000 movie dataset
    
    Parameters:
    movies_file_path (str): Path to the movies CSV file
    credits_file_path (str): Path to the credits CSV file (optional)
    
    Returns:
    pd.DataFrame: Cleaned movies dataframe
    """
    
    # Load the movies dataset
    print("Loading movies dataset...")
    movies_df = pd.read_csv(movies_file_path)
    
    print(f"Original dataset shape: {movies_df.shape}")
    print(f"Columns: {list(movies_df.columns)}")
    
    # Display basic info about missing values
    print("\nMissing values in original dataset:")
    print(movies_df.isnull().sum())
    
    # 1. Handle missing/null values
    print("\n1. Handling missing/null values...")
    
    # Drop rows where critical columns are missing
    critical_columns = ['title', 'release_date']
    initial_count = len(movies_df)
    movies_df = movies_df.dropna(subset=critical_columns)
    print(f"Dropped {initial_count - len(movies_df)} rows with missing critical data")
    
    # Fill missing values for other columns
    fill_values = {
        'overview': 'No overview available',
        'tagline': 'No tagline available',
        'homepage': '',
        'runtime': movies_df['runtime'].median(),
        'budget': 0,
        'revenue': 0,
        'vote_average': movies_df['vote_average'].mean(),
        'vote_count': 0,
        'popularity': movies_df['popularity'].median()
    }
    
    for column, fill_value in fill_values.items():
        if column in movies_df.columns:
            movies_df[column] = movies_df[column].fillna(fill_value)
    
    # 2. Extract genres as lists
    print("\n2. Extracting genres as lists...")
    
    def extract_genres(genres_str):
        """
        Extract genre names from JSON string format
        Example: '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        Returns: ['Action', 'Adventure']
        """
        if pd.isna(genres_str) or genres_str == '':
            return []
        
        try:
            # Parse the JSON string
            genres_list = ast.literal_eval(genres_str)
            # Extract only the 'name' field from each genre dictionary
            return [genre['name'] for genre in genres_list if isinstance(genre, dict) and 'name' in genre]
        except (ValueError, SyntaxError, TypeError):
            # If parsing fails, return empty list
            return []
    
    # Apply genre extraction
    if 'genres' in movies_df.columns:
        movies_df['genres_list'] = movies_df['genres'].apply(extract_genres)
        movies_df['genre_count'] = movies_df['genres_list'].apply(len)
        
        # Create a string version for easier analysis
        movies_df['genres_str'] = movies_df['genres_list'].apply(lambda x: ', '.join(x) if x else 'No genres')
        
        print(f"Sample genres: {movies_df['genres_list'].iloc[0]}")
    
    # 3. Extract release year from release_date column
    print("\n3. Extracting release year from release_date...")
    
    def extract_year(date_str):
        """
        Extract year from date string
        Handles various date formats
        """
        if pd.isna(date_str) or date_str == '':
            return np.nan
        
        try:
            # Try parsing with pandas
            date_obj = pd.to_datetime(date_str)
            return date_obj.year
        except:
            # If that fails, try to extract year manually
            try:
                # Look for 4-digit year pattern
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
                if year_match:
                    return int(year_match.group())
                else:
                    return np.nan
            except:
                return np.nan
    
    # Extract release year
    if 'release_date' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_date'].apply(extract_year)
        
        # Create decade column for additional analysis
        movies_df['decade'] = (movies_df['release_year'] // 10) * 10
        
        print(f"Year range: {movies_df['release_year'].min()} - {movies_df['release_year'].max()}")
    
    # Additional cleaning steps
    print("\n4. Additional data cleaning...")
    
    # Clean up text columns
    text_columns = ['title', 'overview', 'tagline']
    for col in text_columns:
        if col in movies_df.columns:
            movies_df[col] = movies_df[col].astype(str).str.strip()
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
    for col in numeric_columns:
        if col in movies_df.columns:
            movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce').fillna(0)
    
    # Create additional useful columns
    if 'budget' in movies_df.columns and 'revenue' in movies_df.columns:
        movies_df['profit'] = movies_df['revenue'] - movies_df['budget']
        movies_df['roi'] = np.where(movies_df['budget'] > 0, 
                                   movies_df['profit'] / movies_df['budget'], 0)
    
    # Sort by release date
    if 'release_date' in movies_df.columns:
        movies_df = movies_df.sort_values('release_date').reset_index(drop=True)
    
    print(f"\nFinal dataset shape: {movies_df.shape}")
    print(f"Missing values after cleaning:")
    print(movies_df.isnull().sum())
    
    return movies_df

def analyze_cleaned_data(movies_df):
    """
    Perform basic analysis on the cleaned dataset
    """
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"Total movies: {len(movies_df)}")
    print(f"Year range: {movies_df['release_year'].min():.0f} - {movies_df['release_year'].max():.0f}")
    print(f"Average rating: {movies_df['vote_average'].mean():.2f}")
    print(f"Total budget: ${movies_df['budget'].sum():,.0f}")
    print(f"Total revenue: ${movies_df['revenue'].sum():,.0f}")
    
    # Top genres
    if 'genres_list' in movies_df.columns:
        all_genres = []
        for genres in movies_df['genres_list']:
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts()
        print(f"\nTop 10 Genres:")
        print(genre_counts.head(10))
    
    # Movies by decade
    if 'decade' in movies_df.columns:
        decade_counts = movies_df['decade'].value_counts().sort_index()
        print(f"\nMovies by Decade:")
        print(decade_counts)
    
    # Top rated movies
    print(f"\nTop 10 Highest Rated Movies:")
    top_rated = movies_df.nlargest(10, 'vote_average')[['title', 'release_year', 'vote_average', 'genres_str']]
    print(top_rated.to_string(index=False))

def main():
    """
    Main function to demonstrate usage
    """
    # Example usage - replace with your actual file paths
    movies_file = "tmdb_5000_movies.csv"  # Replace with actual path
    
    print("TMDB 5000 Movie Dataset Cleaning Script")
    print("="*50)
    
    try:
        # Load and clean the data
        cleaned_movies = load_and_clean_tmdb_data(movies_file)
        
        # Analyze the cleaned data
        analyze_cleaned_data(cleaned_movies)
        
        # Save the cleaned dataset
        output_file = "cleaned_tmdb_movies.csv"
        cleaned_movies.to_csv(output_file, index=False)
        print(f"\nCleaned dataset saved to: {output_file}")
        
        # Show sample of cleaned data
        print(f"\nSample of cleaned data:")
        sample_columns = ['title', 'release_year', 'genres_str', 'vote_average', 'runtime']
        available_columns = [col for col in sample_columns if col in cleaned_movies.columns]
        print(cleaned_movies[available_columns].head().to_string(index=False))
        
        return cleaned_movies
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{movies_file}'")
        print("Please make sure the TMDB dataset file is in the correct location.")
        print("\nTo download the dataset:")
        print("1. Go to https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("2. Download 'tmdb_5000_movies.csv'")
        print("3. Place it in the same directory as this script")
        return None

if __name__ == "__main__":
    cleaned_data = main()
