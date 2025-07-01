import pandas as pd
import numpy as np
import ast

# Quick example of how to use the cleaning functions
def quick_clean_tmdb_data(file_path):
    """
    Simplified version for quick data cleaning - works with both TMDB and Bollywood datasets
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # 1. Handle missing values
    df = df.dropna(subset=['title', 'release_date'])
    df['overview'] = df['overview'].fillna('No overview')
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # Handle Bollywood-specific columns if they exist
    if 'director' in df.columns:
        df['director'] = df['director'].fillna('Unknown Director')
    if 'language' in df.columns:
        df['language'] = df['language'].fillna('Unknown Language')
    if 'box_office_collection' in df.columns:
        df['box_office_collection'] = df['box_office_collection'].fillna(0)
    if 'imdb_rating' in df.columns:
        df['imdb_rating'] = df['imdb_rating'].fillna(df['imdb_rating'].mean())
    
    # 2. Extract genres as lists
    def parse_genres(genres_str):
        if pd.isna(genres_str):
            return []
        try:
            genres = ast.literal_eval(genres_str)
            return [g['name'] for g in genres]
        except:
            return []
    
    df['genres_list'] = df['genres'].apply(parse_genres)
    
    # 3. Extract release year
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year
    
    return df

# Example usage:
if __name__ == "__main__":
    import os
    
    # Check if the dataset file exists
    dataset_file = 'bollywood_movies.csv'
    
    if os.path.exists(dataset_file):
        print("Loading and cleaning TMDB dataset...")
        df = quick_clean_tmdb_data(dataset_file)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df[['title', 'release_year', 'genres_list']].head())
        
        print(f"\nDataset info:")
        print(f"- Total movies: {len(df)}")
        print(f"- Year range: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")
        print(f"- Unique genres: {len(set([g for sublist in df['genres_list'] for g in sublist]))}")
        
    else:
        print(f"Dataset file '{dataset_file}' not found!")
        print("\nCreating a sample Bollywood movies dataset...")
        
        # Create a comprehensive Bollywood movie dataset
        bollywood_data = {
            'title': [
                'Dangal', '3 Idiots', 'Baahubali 2', 'PK', 'Bajrangi Bhaijaan', 
                'Sultan', 'Sanju', 'Padmaavat', 'Tiger Zinda Hai', 'War',
                'Kabir Singh', 'Uri: The Surgical Strike', 'Mission Mangal', 
                'Housefull 4', 'Good Newwz', 'Tanhaji', 'Sooryavanshi', 
                'RRR', 'KGF Chapter 2', 'Pushpa'
            ],
            'release_date': [
                '2016-12-23', '2009-12-25', '2017-04-28', '2014-12-19', '2015-07-17',
                '2016-07-06', '2018-06-29', '2018-01-25', '2017-12-22', '2019-10-02',
                '2019-06-21', '2019-01-11', '2019-08-15', '2019-10-25', '2019-12-27',
                '2020-01-10', '2021-11-05', '2022-03-25', '2022-04-14', '2021-12-17'
            ],
            'genres': [
                '[{"name": "Drama"}, {"name": "Sport"}]',
                '[{"name": "Comedy"}, {"name": "Drama"}]',
                '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Drama"}]',
                '[{"name": "Comedy"}, {"name": "Drama"}, {"name": "Science Fiction"}]',
                '[{"name": "Drama"}, {"name": "Family"}]',
                '[{"name": "Action"}, {"name": "Drama"}, {"name": "Romance"}]',
                '[{"name": "Biography"}, {"name": "Drama"}]',
                '[{"name": "Drama"}, {"name": "History"}, {"name": "Romance"}]',
                '[{"name": "Action"}, {"name": "Thriller"}]',
                '[{"name": "Action"}, {"name": "Thriller"}]',
                '[{"name": "Drama"}, {"name": "Romance"}]',
                '[{"name": "Action"}, {"name": "Drama"}, {"name": "War"}]',
                '[{"name": "Drama"}, {"name": "Science Fiction"}]',
                '[{"name": "Comedy"}, {"name": "Horror"}]',
                '[{"name": "Comedy"}, {"name": "Drama"}]',
                '[{"name": "Action"}, {"name": "Biography"}, {"name": "Drama"}]',
                '[{"name": "Action"}, {"name": "Crime"}, {"name": "Thriller"}]',
                '[{"name": "Action"}, {"name": "Drama"}]',
                '[{"name": "Action"}, {"name": "Crime"}, {"name": "Thriller"}]',
                '[{"name": "Action"}, {"name": "Crime"}, {"name": "Thriller"}]'
            ],
            'overview': [
                'Former wrestler Mahavir Singh Phogat trains his daughters to become world-class wrestlers.',
                'Two friends are searching for their lost companion. They revisit their college days.',
                'When Shiva, the son of Bahubali, learns about his heritage, he begins to look for answers.',
                'An alien on Earth loses the only device he can use to communicate with his spaceship.',
                'A young mute girl from Pakistan loses her way and crosses over to India.',
                'Sultan is a classic underdog tale about a wrestlers journey.',
                'Biopic of the controversial life of actor Sanjay Dutt.',
                'Queen Padmavati is known for her exceptional beauty along with a keen strategic mind.',
                'Tiger is alive and kicking in this action-packed thriller.',
                'The story of Indian soldiers who fought in the Kargil war.',
                'A short-tempered house surgeon gets used to drugs and drinks when his girlfriend is forced to marry another person.',
                'Indian army special forces execute a covert operation.',
                'Based on true events of the Indian Space Research Organisation.',
                'Three couples who get separated from each other due to an evil ploy.',
                'Two couples with the same surnames pursue in-vitro fertilization.',
                'The story of Tanhaji Malusare, a military chieftain in the army of Chhatrapati Shivaji Maharaj.',
                'A fearless cop takes on a dangerous gang to save his city.',
                'A tale of two legendary revolutionaries and their journey far away from home.',
                'Rocky, whose name strikes fear in the heart of his foes.',
                'The rise of a truck driver from the forests of Seshachalam.'
            ],
            'runtime': [
                161, 170, 167, 153, 163, 170, 155, 164, 158, 154,
                182, 138, 130, 146, 134, 134, 145, 187, 168, 179
            ],
            'director': [
                'Nitesh Tiwari', 'Rajkumar Hirani', 'S.S. Rajamouli', 'Rajkumar Hirani', 'Kabir Khan',
                'Ali Abbas Zafar', 'Rajkumar Hirani', 'Sanjay Leela Bhansali', 'Ali Abbas Zafar', 'Siddharth Anand',
                'Sandeep Reddy Vanga', 'Aditya Dhar', 'Jagan Shakti', 'Farhad Samji', 'Raj Mehta',
                'Om Raut', 'Rohit Shetty', 'S.S. Rajamouli', 'Prashanth Neel', 'Sukumar'
            ],
            'language': [
                'Hindi', 'Hindi', 'Telugu', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi',
                'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Hindi', 'Telugu', 'Kannada', 'Telugu'
            ],
            'box_office_collection': [
                2000, 2024, 1810, 832, 969, 623, 586, 585, 565, 475,
                379, 342, 290, 282, 220, 368, 296, 1387, 1215, 373
            ],
            'imdb_rating': [
                8.4, 8.4, 8.7, 8.1, 8.1, 7.0, 7.6, 7.0, 7.0, 6.5,
                7.0, 8.3, 6.5, 4.1, 6.7, 7.6, 6.0, 7.9, 8.2, 7.6
            ],
            'poster_url': [
                'https://imgs.search.brave.com/GGwni8iMmZIB3_TAhsYPy_oVlF2o1fOurMZW0hcEll8/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMud2lraWEubm9j/b29raWUubmV0L2Rp/c25leS9pbWFnZXMv/OS85OS9EYW5nYWxf/UG9zdGVyLmpwZy9y/ZXZpc2lvbi9sYXRl/c3Qvc2NhbGUtdG8t/d2lkdGgtZG93bi81/MTY_Y2I9MjAxNjA3/MTAwNDUyMTc',  # Dangal
                'https://imgs.search.brave.com/qQEEMy7w4AuBvCDSudLwlv_TOcV-4oIoyaK3tfIrgag/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9waWMu/YnN0YXJzdGF0aWMu/Y29tL3VnYy82ZGI2/YTdkZTUyMDE2NjFl/NmE5ZTMzOGY1MTZh/YWZlYjUwNDZhYmUz/LnBuZ0AzMjB3XzE4/MGhfMWVfMWNfOTBx',  # 3 Idiots
                'https://imgs.search.brave.com/NlA_Cf9Jp0x4xLAFY6M840YbYArZCxf7varlDTvKPtw/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9tdW1i/YWltaXJyb3IuaW5k/aWF0aW1lcy5jb20v/cGhvdG8vNTgzOTQx/NjguY21z',  # Baahubali 2
                'https://imgs.search.brave.com/ZFIz_z6apeOZEXSujDy_q3nyQcOdcRwMRK2BAyeQJn0/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL00v/TVY1Qk5URTVabUkx/T1RZdE16bGhOUzAw/TmpBMExUaGlNRFl0/Tm1ZMFlXVmhZVFk0/WWpkalhrRXlYa0Zx/Y0djQC5qcGc',  # PK
                'https://imgs.search.brave.com/8ZFOVpSGwix-JtBym38YqZv9-NL67YXkn4ZXBCQ7ZBU/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9waWMu/YnN0YXJzdGF0aWMu/Y29tL3VnYy9hNjQy/NzZlM2RmNmU2NjQ3/MGRlMDhkZTBmMGZm/MWE4Ni5qcGdAMzIw/d18xODBoXzFlXzFj/XzkwcQ',  # Bajrangi Bhaijaan
                'https://imgs.search.brave.com/7fUm3apfiiCnjsotCl9mOea0zMuhnzOkne62peUsPJU/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJjYXZlLmNv/bS93cC93cDgzNTA4/OTcuanBn',  # Sultan
                'https://imgs.search.brave.com/-hAuU--1iYQYGsuiCkKopizHue6wInps8-xRBrbWkIg/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudG9paW1nLmNv/bS9pbWcvNjM4OTMw/NjQvTWFzdGVyLmpw/Zz9pbWdzaXplPTEz/NDAzOA',  # Sanju
                'https://imgs.search.brave.com/71mEZCP3Nrk1n-z73021-43GYv6EDsJhH3KZ76qoskc/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/cm9nZXJlYmVydC5j/b20vd3AtY29udGVu/dC91cGxvYWRzLzIw/MjQvMDgvUGFkbWFh/dmF0LmpwZw',  # Padmaavat
                'https://imgs.search.brave.com/70HGjapbc3DMriaqVIM-8i1u7ADvY23PduLCALYU19Q/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMuaW51dGguY29t/LzIwMTcvMDUvMUth/dHJpbmEtS2FpZi1h/bmQtU2FsbWFuLUto/YW4tb24tdGhlLWFj/dGlvbi1wYWNrZWQt/cG9zdGVyLW9mLVRp/Z2VyLVppbmRhLUhh/aS5qcGc',  # Tiger Zinda Hai
                'https://imgs.search.brave.com/r1fpie0tSjkKfm8NIl3kuoHc81eojbfO7VQvTjZjl5w/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/bGl2ZWhpbmR1c3Rh/bi5jb20vbGgtaW1n/L3VwbG9hZGltYWdl/L2xpYnJhcnkvMjAx/OS8wOC8xMi8xNl85/LzE2XzlfNi93YXJf/MTU2NTU5OTQ2My5q/cGc',  # War
                'https://imgs.search.brave.com/Fd411SLG4CyEipuNQ_CRp3S6sp6A06igQCe7OZoI_7A/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/c2FhbWFuYS5jb20v/d3AtY29udGVudC91/cGxvYWRzLzIwMTkv/MTIva2FiaXItc2lu/Z2gtbW92aWUtcG9z/dGVyLmpwZw',  # Kabir Singh
                'https://imgs.search.brave.com/mJ01xUhMvCkWZgV5RUwom9UsYm1XArQ0y-H-skZGYIA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJjYXZlLmNv/bS93cC93cDQ1Njk5/MTIuanBn',  # Uri
                'https://imgs.search.brave.com/-CnRbq6F5YJKhKewwQU_J00YFOlifIil_m56tHA8C5I/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMubW92aWVjcm93/LmNvbS9tYXJxdWVl/L21pc3Npb24tbWFu/Z2FsLWRheS02LWJv/eC1vZmZpY2UtYWtz/aGF5LWt1bWFycy1m/aWxtLXN0YXlzLXN0/cm9uZy1vbi13ZWVr/ZGF5cy8xNjcyMzZf/dGh1bWJfNTY1Lmpw/Zw',  # Mission Mangal
                'https://imgs.search.brave.com/UT6iFQN6HfY7-MyQR7HG5YBTnhZiUldjmdHLkUeDGFA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9pbWcu/ZXRpbWcuY29tL3Ro/dW1iL21zaWQtNzE3/NzI4OTUsd2lkdGgt/NjUwLGhlaWdodC00/ODgsaW1nc2l6ZS0z/MDE1MDMscmVzaXpl/bW9kZS03NS9ob3Vz/ZWZ1bGwtNC5qcGc',  # Housefull 4
                'https://imgs.search.brave.com/L4GGm2gGyWNEsVI44na5gtYrp2fogYODznlmmdNtuSI/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudG9paW1nLmNv/bS90aHVtYi9pbWdz/aXplLTExMzM0OSxt/c2lkLTcyMzI5NTQ3/LHdpZHRoLTQwMCxy/ZXNpemVtb2RlLTQv/NzIzMjk1NDcuanBn',  # Good Newwz
                'https://imgs.search.brave.com/kqND1EXZml-bhOuTTLsAieYVioECc0IiYzuC-OkSrfY/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93aWtp/YmlvLmluL3dwLWNv/bnRlbnQvdXBsb2Fk/cy8yMDE5LzEwL1Rh/bmhhamktVGhlLVVu/c3VuZy1XYXJyaW9y/LmpwZw',  # Tanhaji
                'https://imgs.search.brave.com/BaIv1WmBLzs2V-uXQFg3S-YxcICUk3-79tprTfQjNhY/rs:fit:0:180:1:0/g:ce/aHR0cHM6Ly93d3cu/Ym9sbHl3b29kaHVu/Z2FtYS5jb20vd3At/Y29udGVudC91cGxv/YWRzLzIwMTgvMTIv/U29vcnlhdmFuc2hp/LWJhbm5lci5qcGc',  # Sooryavanshi
                'https://imgs.search.brave.com/p9LRZ4NKO3NqDpFXB5VimghKYly3f6t-JjOWrj3DIVA/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJzLmNvbS9p/bWFnZXMvaGQvcnJy/LXRyYWluLXNjZW5l/LWlldnIzZHE1azNq/cmJoeDEuanBn',  # RRR
                'https://imgs.search.brave.com/xBG2EqfH1N1aPcNxsLRnzkV6NKEuAIFU_J5MCZevBcc/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMuZmlsbWliZWF0/LmNvbS9waC1iaWcv/MjAyMi8wMy9rZ2Yt/Y2hhcHRlci0yXzE2/NDc4NDMwNDAxODAu/anBn',  # KGF Chapter 2
                'https://image.tmdb.org/t/p/w500/cHRzRf0cTkQJ6r6L1FvSmKDfTJX.jpg'   # Pushpa (keeping original)
            ]
        }
        
        import pandas as pd
        sample_df = pd.DataFrame(bollywood_data)
        sample_df.to_csv('bollywood_movies.csv', index=False)
        
        print(f"\nCreated Bollywood dataset 'bollywood_movies.csv' with {len(sample_df)} movies!")
        print("Columns created:", sample_df.columns.tolist())
        print("Testing with Bollywood data:")
        
        df = quick_clean_tmdb_data('bollywood_movies.csv')
        print(f"\nDataset loaded successfully! Shape: {df.shape}")
        print(f"\nFirst 5 Bollywood movies:")
        print(df[['title', 'release_year', 'genres_list', 'director', 'language']].head())
        
        print(f"\nBollywood Dataset Statistics:")
        print(f"- Total movies: {len(df)}")
        print(f"- Year range: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")
        print(f"- Languages: {', '.join(df['language'].unique())}")
        print(f"- Unique genres: {len(set([g for sublist in df['genres_list'] for g in sublist]))}")
        print(f"- Average IMDB rating: {df['imdb_rating'].mean():.1f}")
        print(f"- Top grossing movie: {df.loc[df['box_office_collection'].idxmax(), 'title']} (₹{df['box_office_collection'].max()} Cr)")
        
        # Show genre distribution
        all_genres = [genre for sublist in df['genres_list'] for genre in sublist]
        genre_counts = pd.Series(all_genres).value_counts()
        print(f"\nTop genres in Bollywood:")
        print(genre_counts.head())
