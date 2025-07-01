import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from quick_clean_example import quick_clean_tmdb_data
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class MovieSentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def textblob_sentiment(self, text):
        """
        Analyze sentiment using TextBlob
        Returns polarity (-1 to 1) and subjectivity (0 to 1)
        """
        if pd.isna(text) or text == '':
            return 0, 0
        
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def vader_sentiment(self, text):
        """
        Analyze sentiment using VADER
        Returns compound score (-1 to 1)
        """
        if pd.isna(text) or text == '':
            return 0
        
        scores = self.vader_analyzer.polarity_scores(str(text))
        return scores['compound']
    
    def classify_sentiment(self, score, method='textblob'):
        """
        Classify sentiment score into Positive, Negative, or Neutral
        """
        if method == 'textblob':
            if score > 0.1:
                return 'Positive'
            elif score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        else:  # VADER
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
    
    def analyze_movie_sentiments(self, df):
        """
        Perform sentiment analysis on movie overviews
        """
        print("Analyzing sentiment of movie overviews...")
        
        # TextBlob Analysis
        print("• Running TextBlob sentiment analysis...")
        textblob_results = df['overview'].apply(self.textblob_sentiment)
        df['textblob_polarity'] = [result[0] for result in textblob_results]
        df['textblob_subjectivity'] = [result[1] for result in textblob_results]
        df['textblob_sentiment'] = df['textblob_polarity'].apply(
            lambda x: self.classify_sentiment(x, 'textblob'))
        
        # VADER Analysis
        print("• Running VADER sentiment analysis...")
        df['vader_compound'] = df['overview'].apply(self.vader_sentiment)
        df['vader_sentiment'] = df['vader_compound'].apply(
            lambda x: self.classify_sentiment(x, 'vader'))
        
        return df
    
    def create_sentiment_visualizations(self, df):
        """
        Create comprehensive sentiment visualizations
        """
        print("Creating sentiment visualizations...")
        
        # Create main figure
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Movie Overview Sentiment Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. TextBlob Sentiment Distribution (Pie Chart)
        textblob_counts = df['textblob_sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        explode = (0.05, 0.05, 0.05)
        
        axes[0,0].pie(textblob_counts.values, labels=textblob_counts.index, 
                      autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
        axes[0,0].set_title('TextBlob Sentiment Distribution', fontweight='bold', pad=20)
        
        # 2. VADER Sentiment Distribution (Pie Chart)
        vader_counts = df['vader_sentiment'].value_counts()
        axes[0,1].pie(vader_counts.values, labels=vader_counts.index, 
                      autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
        axes[0,1].set_title('VADER Sentiment Distribution', fontweight='bold', pad=20)
        
        # 3. Sentiment Score Distribution (Histogram)
        axes[1,0].hist(df['textblob_polarity'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].axvline(df['textblob_polarity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["textblob_polarity"].mean():.3f}')
        axes[1,0].set_title('TextBlob Polarity Score Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Polarity Score (-1 to 1)')
        axes[1,0].set_ylabel('Number of Movies')
        axes[1,0].legend()
        axes[1,0].grid(axis='y', alpha=0.3)
        
        # 4. VADER Score Distribution (Histogram)
        axes[1,1].hist(df['vader_compound'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1,1].axvline(df['vader_compound'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["vader_compound"].mean():.3f}')
        axes[1,1].set_title('VADER Compound Score Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Compound Score (-1 to 1)')
        axes[1,1].set_ylabel('Number of Movies')
        axes[1,1].legend()
        axes[1,1].grid(axis='y', alpha=0.3)
        
        # 5. Sentiment vs IMDB Rating
        sentiment_colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        
        for sentiment in df['textblob_sentiment'].unique():
            data = df[df['textblob_sentiment'] == sentiment]
            axes[2,0].scatter(data['textblob_polarity'], data['imdb_rating'], 
                             label=sentiment, alpha=0.7, s=100, 
                             color=sentiment_colors[sentiment])
        
        axes[2,0].set_title('Sentiment vs IMDB Rating (TextBlob)', fontweight='bold')
        axes[2,0].set_xlabel('TextBlob Polarity Score')
        axes[2,0].set_ylabel('IMDB Rating')
        axes[2,0].legend()
        axes[2,0].grid(alpha=0.3)
        
        # 6. Box Office vs Sentiment
        sentiments = ['Positive', 'Negative', 'Neutral']
        box_office_by_sentiment = [df[df['textblob_sentiment'] == s]['box_office_collection'].values 
                                  for s in sentiments]
        
        bp = axes[2,1].boxplot(box_office_by_sentiment, labels=sentiments, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[2,1].set_title('Box Office Collection by Sentiment', fontweight='bold')
        axes[2,1].set_xlabel('Sentiment Category')
        axes[2,1].set_ylabel('Box Office Collection (₹ Crores)')
        axes[2,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('movie_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sentiment_comparison(self, df):
        """
        Create sentiment comparison between TextBlob and VADER
        """
        print("Creating sentiment method comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Correlation between TextBlob and VADER scores
        axes[0].scatter(df['textblob_polarity'], df['vader_compound'], 
                       alpha=0.7, s=100, color='purple')
        
        # Add correlation coefficient
        correlation = np.corrcoef(df['textblob_polarity'], df['vader_compound'])[0,1]
        axes[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[0].transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[0].set_title('TextBlob vs VADER Sentiment Scores', fontweight='bold')
        axes[0].set_xlabel('TextBlob Polarity')
        axes[0].set_ylabel('VADER Compound Score')
        axes[0].grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['textblob_polarity'], df['vader_compound'], 1)
        p = np.poly1d(z)
        axes[0].plot(df['textblob_polarity'], p(df['textblob_polarity']), "r--", alpha=0.8)
        
        # 2. Agreement between methods
        agreement_data = []
        for tb_sent, vader_sent in zip(df['textblob_sentiment'], df['vader_sentiment']):
            if tb_sent == vader_sent:
                agreement_data.append('Agree')
            else:
                agreement_data.append('Disagree')
        
        agreement_counts = pd.Series(agreement_data).value_counts()
        colors = ['#2ecc71', '#e74c3c']
        
        axes[1].pie(agreement_counts.values, labels=agreement_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('TextBlob vs VADER Agreement', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sentiment_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_word_clouds(self, df):
        """
        Create word clouds for different sentiment categories
        """
        print("Creating sentiment-based word clouds...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sentiments = ['Positive', 'Negative', 'Neutral']
        colors = ['Greens', 'Reds', 'Greys']
        
        for i, (sentiment, colormap) in enumerate(zip(sentiments, colors)):
            # Get overviews for this sentiment
            sentiment_overviews = df[df['textblob_sentiment'] == sentiment]['overview']
            text = ' '.join(sentiment_overviews.astype(str))
            
            if text.strip():
                # Create word cloud
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    colormap=colormap,
                                    max_words=100).generate(text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment} Sentiment Keywords', fontweight='bold', pad=20)
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sentiment} movies', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{sentiment} Sentiment Keywords', fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_sentiment_analysis_report(self, df):
        """
        Print comprehensive sentiment analysis report
        """
        print("\n" + "="*70)
        print("MOVIE SENTIMENT ANALYSIS REPORT")
        print("="*70)
        
        # Overall statistics
        print(f"\n📊 OVERALL SENTIMENT STATISTICS:")
        print(f"   • Total Movies Analyzed: {len(df)}")
        print(f"   • Average TextBlob Polarity: {df['textblob_polarity'].mean():.3f}")
        print(f"   • Average VADER Compound: {df['vader_compound'].mean():.3f}")
        print(f"   • Average Subjectivity: {df['textblob_subjectivity'].mean():.3f}")
        
        # TextBlob results
        print(f"\n🎭 TEXTBLOB SENTIMENT DISTRIBUTION:")
        textblob_counts = df['textblob_sentiment'].value_counts()
        for sentiment, count in textblob_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   • {sentiment}: {count} movies ({percentage:.1f}%)")
        
        # VADER results
        print(f"\n⚡ VADER SENTIMENT DISTRIBUTION:")
        vader_counts = df['vader_sentiment'].value_counts()
        for sentiment, count in vader_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   • {sentiment}: {count} movies ({percentage:.1f}%)")
        
        # Agreement analysis
        agreement = sum(df['textblob_sentiment'] == df['vader_sentiment'])
        agreement_percentage = (agreement / len(df)) * 100
        print(f"\n🤝 METHOD AGREEMENT:")
        print(f"   • Agreement: {agreement}/{len(df)} movies ({agreement_percentage:.1f}%)")
        
        # Correlation
        correlation = np.corrcoef(df['textblob_polarity'], df['vader_compound'])[0,1]
        print(f"   • Score Correlation: {correlation:.3f}")
        
        # Most positive and negative movies
        print(f"\n🌟 MOST POSITIVE MOVIES (TextBlob):")
        most_positive = df.nlargest(3, 'textblob_polarity')[['title', 'textblob_polarity', 'imdb_rating']]
        for _, movie in most_positive.iterrows():
            print(f"   • {movie['title']}: {movie['textblob_polarity']:.3f} (IMDB: {movie['imdb_rating']})")
        
        print(f"\n🌧️ MOST NEGATIVE MOVIES (TextBlob):")
        most_negative = df.nsmallest(3, 'textblob_polarity')[['title', 'textblob_polarity', 'imdb_rating']]
        for _, movie in most_negative.iterrows():
            print(f"   • {movie['title']}: {movie['textblob_polarity']:.3f} (IMDB: {movie['imdb_rating']})")
        
        # Sentiment vs Performance
        print(f"\n📈 SENTIMENT VS PERFORMANCE:")
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_data = df[df['textblob_sentiment'] == sentiment]
            if len(sentiment_data) > 0:
                avg_rating = sentiment_data['imdb_rating'].mean()
                avg_collection = sentiment_data['box_office_collection'].mean()
                print(f"   • {sentiment} Movies:")
                print(f"     - Average IMDB Rating: {avg_rating:.2f}")
                print(f"     - Average Collection: ₹{avg_collection:.0f} Cr")

def main():
    """
    Main function to run sentiment analysis
    """
    try:
        # Load the dataset
        print("Loading Bollywood movie dataset...")
        df = quick_clean_tmdb_data('bollywood_movies.csv')
        
        # Initialize sentiment analyzer
        analyzer = MovieSentimentAnalyzer()
        
        # Perform sentiment analysis
        df = analyzer.analyze_movie_sentiments(df)
        
        # Create visualizations
        analyzer.create_sentiment_visualizations(df)
        analyzer.create_sentiment_comparison(df)
        analyzer.create_word_clouds(df)
        
        # Print detailed report
        analyzer.print_sentiment_analysis_report(df)
        
        # Save results
        df.to_csv('bollywood_movies_with_sentiment.csv', index=False)
        
        print(f"\n✅ Sentiment Analysis Complete!")
        print(f"📊 Visualizations saved:")
        print(f"   • movie_sentiment_analysis.png")
        print(f"   • sentiment_method_comparison.png")
        print(f"   • sentiment_wordclouds.png")
        print(f"📁 Dataset with sentiment scores saved as: bollywood_movies_with_sentiment.csv")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: bollywood_movies.csv not found!")
        print("Please run quick_clean_example.py first to create the dataset.")
        return None
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Run sentiment analysis
    df_with_sentiment = main()
