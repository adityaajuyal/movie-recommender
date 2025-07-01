# 🎬 Bollywood Movie Recommendation System

A comprehensive movie recommendation and analysis system built with Python and Streamlit, featuring content-based filtering and interactive visualizations for Bollywood movies.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Content--Based%20Filtering-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

This project implements an intelligent movie recommendation system that suggests Bollywood movies based on content similarity. Using advanced machine learning techniques like TF-IDF vectorization and cosine similarity, the system analyzes movie features such as genres, directors, overviews, and more to provide personalized recommendations.

### 🎯 Key Highlights

- **20 Curated Bollywood Movies** with high-quality poster images
- **Content-Based Filtering** using TF-IDF and cosine similarity
- **Interactive Web Interface** built with Streamlit
- **Advanced Filtering Options** by language, genre, and release year
- **Data Visualizations** with interactive charts and graphs
- **Export Functionality** to download recommendations as CSV

## ✨ Features

### Core Functionality
- 🎭 **Movie Selection**: Choose from 20 popular Bollywood movies
- 🔍 **Smart Recommendations**: Get 3-10 similar movies based on content analysis
- 📊 **Similarity Scoring**: See how closely recommended movies match your selection
- 🎨 **Visual Movie Cards**: Beautiful cards displaying posters, ratings, and details

### Advanced Features
- 🌍 **Language Filtering**: Filter by Hindi, Telugu, or Kannada
- 🎬 **Genre Filtering**: Filter by Action, Drama, Comedy, Romance, etc.
- 📅 **Year Range Selection**: Filter movies by release year (2009-2022)
- 📈 **Interactive Charts**: IMDB ratings, box office collections, and similarity analysis
- 🖼️ **Poster Gallery**: Clean grid view of recommended movies
- 📥 **Export Options**: Download recommendations as CSV files

### User Interface
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Modern UI**: Clean, professional interface with custom CSS
- 🚀 **Fast Performance**: Cached data loading and processing
- 🎯 **Intuitive Navigation**: Easy-to-use sidebar controls

## 🛠️ Tech Stack

### Backend & ML
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms (TF-IDF, Cosine Similarity)
- **AST**: JSON parsing for movie data

### Frontend & Visualization
- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations
- **HTML/CSS**: Custom styling and layout
- **Markdown**: Documentation and content formatting

### Data Processing
- **TF-IDF Vectorization**: Text analysis and feature extraction
- **Cosine Similarity**: Content similarity calculation
- **Data Cleaning**: Automated data preprocessing and validation

## 📊 Dataset

The system uses a curated dataset of 20 popular Bollywood movies spanning from 2009 to 2022, including:

### Movie Features
- **Basic Info**: Title, Director, Release Year, Language, Runtime
- **Ratings**: IMDB ratings and user scores
- **Financial**: Box office collections in Crores (₹)
- **Content**: Movie overviews, genres, and taglines
- **Visual**: High-quality poster image URLs

### Sample Movies
- Dangal (2016) - Sports Drama
- 3 Idiots (2009) - Comedy Drama  
- Baahubali 2 (2017) - Action Epic
- PK (2014) - Comedy Sci-Fi
- Bajrangi Bhaijaan (2015) - Family Drama
- And 15 more blockbuster movies...

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8 or higher installed on your system.

```bash
python --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bollywood-movie-recommender.git
cd bollywood-movie-recommender
```

2. **Install required packages**
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

3. **Generate the dataset**
```bash
python quick_clean_example.py
```

4. **Run the application**
```bash
streamlit run movie_recommendation_app.py
```

5. **Open your browser**
The app will automatically open at `http://localhost:8501`

### Alternative Installation

You can also install dependencies using a requirements file:

```bash
pip install -r requirements.txt
```

## 💻 How to Use

### Basic Usage

1. **Select a Movie**: Choose from the dropdown in the sidebar
2. **Set Preferences**: Adjust the number of recommendations (3-10)
3. **Apply Filters**: Filter by language, genre, or release year
4. **Choose Display**: Select between detailed cards or poster gallery
5. **Explore Results**: View recommendations with similarity scores

### Advanced Features

- **Visual Analysis**: Scroll down to see interactive charts comparing ratings, collections, and similarity scores
- **Export Data**: Download your recommendations as a CSV file
- **Filter Refinement**: Use multiple filters simultaneously for precise results

## 📸 Screenshots

### Main Interface
![Main Interface](screenshots/main_interface.png)
*The main application interface showing movie selection and recommendations*

### Movie Selection
![Movie Selection](screenshots/movie_selection.png)
*Sidebar with movie selection and filter options*

### Detailed Cards View
![Detailed Cards](screenshots/detailed_cards.png)
*Detailed movie cards showing posters, ratings, and information*

### Poster Gallery
![Poster Gallery](screenshots/poster_gallery.png)
*Clean grid layout showing recommended movies*

### Interactive Charts
![Charts](screenshots/interactive_charts.png)
*Data visualization dashboard with multiple chart types*

## 🧮 Algorithm Details

### Content-Based Filtering

The recommendation system uses the following approach:

1. **Feature Engineering**: Combines movie overviews, genres, taglines, directors, and languages
2. **Text Processing**: Converts text to lowercase and removes noise
3. **TF-IDF Vectorization**: Creates numerical representations of movie features
4. **Cosine Similarity**: Calculates similarity between movies (0-100%)
5. **Ranking**: Sorts recommendations by similarity score

### Technical Parameters

```python
TfidfVectorizer(
    max_features=5000,      # Maximum vocabulary size
    stop_words='english',   # Remove common English words
    ngram_range=(1, 2),     # Use 1-2 word combinations
    min_df=1,              # Minimum document frequency
    max_df=0.8             # Maximum document frequency
)
```

## 📁 Project Structure

```
bollywood-movie-recommender/
│
├── movie_recommendation_app.py    # Main Streamlit application
├── quick_clean_example.py         # Dataset generation and cleaning
├── bollywood_movies.csv          # Generated movie dataset
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── screenshots/                  # Application screenshots
│   ├── main_interface.png
│   ├── movie_selection.png
│   ├── detailed_cards.png
│   ├── poster_gallery.png
│   └── interactive_charts.png
│
└── assets/                       # Additional resources
    └── movie_posters/            # Backup poster images
```

## 🔧 Customization

### Adding New Movies

1. **Edit Dataset**: Modify `quick_clean_example.py` to include new movie data
2. **Add Posters**: Include poster URLs for new movies
3. **Regenerate**: Run the script to create an updated CSV file
4. **Restart App**: Relaunch the Streamlit application

### Modifying Recommendations

- **Change Algorithm**: Modify the TF-IDF parameters in `build_recommendation_system()`
- **Add Features**: Include new movie attributes in the feature combination
- **Adjust Weights**: Give more importance to specific features (genres, directors, etc.)

### UI Customization

- **Colors**: Modify the CSS gradients and color schemes
- **Layout**: Adjust column ratios and component positioning
- **Styling**: Update the custom CSS in the Streamlit markdown

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution

- 🎬 **More Movies**: Expand the dataset with additional Bollywood movies
- 🔍 **Better Algorithms**: Implement collaborative filtering or hybrid methods
- 🎨 **UI Improvements**: Enhance the user interface and user experience
- 📊 **New Features**: Add user ratings, reviews, or social features
- 🐛 **Bug Fixes**: Report and fix any issues you find

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Movie Data**: Curated from various public sources
- **Poster Images**: High-quality images from movie databases
- **Streamlit Community**: For the amazing web framework
- **Scikit-learn**: For machine learning algorithms
- **Plotly**: For interactive visualizations

## 📞 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/bollywood-movie-recommender](https://github.com/yourusername/bollywood-movie-recommender)

---

⭐ **Star this repository if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/bollywood-movie-recommender.svg?style=social&label=Star)](https://github.com/yourusername/bollywood-movie-recommender)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/bollywood-movie-recommender.svg?style=social&label=Fork)](https://github.com/yourusername/bollywood-movie-recommender/fork)

## Cleaning Steps

1. **Missing Values**:
   - Drops rows missing 'title' or 'release_date'
   - Fills missing 'overview' with 'No overview available'
   - Fills missing numeric values with median/mean/0 as appropriate

2. **Genre Extraction**:
   - Parses JSON-formatted genre strings
   - Extracts genre names into Python lists
   - Creates additional genre-related columns

3. **Date Processing**:
   - Extracts year from release_date
   - Creates decade column
   - Handles various date formats

4. **Additional Features**:
   - Calculates profit and ROI
   - Ensures proper data types
   - Creates analysis-ready columns

## Output

The cleaned dataset includes these key columns:
- `genres_list`: List of genre names
- `release_year`: Extracted year
- `decade`: Decade of release
- `profit`: Revenue - Budget
- `roi`: Return on Investment
- `genres_str`: Comma-separated genre string

## Example Output

```
    title  release_year              genres_list  vote_average  runtime
0  Avatar          2009  [Action, Adventure, Fantasy]           7.2      162
1  Pirates of the Caribbean: At World's End  2007  [Adventure, Fantasy, Action]  6.9  169
```
