#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import optional packages with fallbacks
try:
    from googleapiclient.discovery import build
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    st.error("Google API client not available. Please install: pip install google-api-python-client")
    YOUTUBE_API_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    # Download required NLTK data quietly
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    st.warning("NLTK not available. Using basic sentiment analysis.")
    NLTK_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    st.warning("WordCloud not available. Word clouds will be skipped.")
    WORDCLOUD_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    st.warning("Scikit-learn not available. Advanced spam detection will be disabled.")
    SKLEARN_AVAILABLE = False

# Initialize Streamlit App
st.set_page_config(page_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# Configuration
API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your own YouTube API Key

# Initialize YouTube API if available
if YOUTUBE_API_AVAILABLE:
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        YOUTUBE_API_AVAILABLE = False

# Function Definitions
def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'v=([0-9A-Za-z_-]{11})',
        r'\/([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_comments(video_id):
    """Fetch YouTube Comments with Username"""
    if not YOUTUBE_API_AVAILABLE:
        st.error("YouTube API not available")
        return pd.DataFrame()
    
    comments = []
    timestamps = []
    users = []
    
    try:
        request = youtube.commentThreads().list(
            part='snippet', 
            videoId=video_id,
            textFormat='plainText',
            maxResults=100
        )
        
        response = request.execute()
        
        for item in response['items']:
            comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            timestamps.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])
            users.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])
            
        return pd.DataFrame({
            'User': users, 
            'Comment': comments, 
            'Timestamp': pd.to_datetime(timestamps)
        })
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return pd.DataFrame()

def preprocess_text(text):
    """Basic text preprocessing"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    
    # Basic stopword removal if NLTK is available
    if NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english'))
            text = ' '.join(word for word in text.split() if word not in stop_words)
        except:
            pass  # If stopwords fail, continue without them
    
    return text

def detect_sentiment_basic(comment):
    """Basic sentiment analysis without NLTK"""
    positive_words = ['good', 'great', 'awesome', 'amazing', 'excellent', 'love', 'like', 'best', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid', 'boring', 'sucks']
    
    comment_lower = comment.lower()
    positive_count = sum(1 for word in positive_words if word in comment_lower)
    negative_count = sum(1 for word in negative_words if word in comment_lower)
    
    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
        return 'Negative'
    else:
        return 'Neutral'

def detect_sentiment(comment):
    """Sentiment analysis with fallback"""
    if NLTK_AVAILABLE:
        try:
            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(comment)
            if score['compound'] > 0.05:
                return 'Positive'
            elif score['compound'] < -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            pass
    
    # Fallback to basic sentiment analysis
    return detect_sentiment_basic(comment)

def detect_spam_basic(comment):
    """Basic spam detection without machine learning"""
    spam_indicators = [
        'subscribe', 'follow me', 'check out my', 'visit my channel', 
        'click here', 'free money', 'make money', 'earn money',
        'http', 'www.', '.com', 'link in bio'
    ]
    
    comment_lower = comment.lower()
    spam_count = sum(1 for indicator in spam_indicators if indicator in comment_lower)
    
    # Simple heuristic: if comment has multiple spam indicators or is very short with spam words
    if spam_count >= 2 or (len(comment.split()) <= 5 and spam_count >= 1):
        return 'Spam'
    return 'Not Spam'

# Main App Interface
st.sidebar.header("Configuration")
if not YOUTUBE_API_AVAILABLE:
    st.sidebar.error("YouTube API not configured properly")
else:
    st.sidebar.success("YouTube API ready")

# User Input
video_url = st.text_input('Enter YouTube Video URL:', placeholder='https://www.youtube.com/watch?v=...')

# Only show analysis if URL is provided
if video_url:
    if st.button("Analyze Comments", type="primary"):
        video_id = extract_video_id(video_url)
        
        if video_id:
            st.success(f'✅ Video ID extracted: {video_id}')
            
            # Fetch comments
            with st.spinner('Fetching comments from YouTube...'):
                df = get_youtube_comments(video_id)
            
            if not df.empty:
                st.success(f"📊 Successfully fetched {len(df)} comments!")
                
                # Preprocess comments
                with st.spinner('Processing comments...'):
                    df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
                    df['Sentiment'] = df['Processed_Comment'].apply(detect_sentiment)
                    df['Spam'] = df['Comment'].apply(detect_spam_basic)
                
                # Display results
                st.header("📈 Analysis Results")
                
                # Sentiment Overview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df['Sentiment'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Green, Red, Teal
                    sentiment_counts.plot(kind='bar', ax=ax, color=colors)
                    plt.title('Comment Sentiments')
                    plt.xlabel('Sentiment')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Sentiment Pie Chart")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                           autopct='%1.1f%%', colors=colors)
                    plt.title('Sentiment Distribution')
                    st.pyplot(fig)
                
                # Comments Analysis
                st.subheader("📝 Comment Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Positive Comments:**")
                    positive_comments = df[df['Sentiment'] == 'Positive']['Comment'].head(5)
                    for i, comment in enumerate(positive_comments, 1):
                        st.write(f"{i}. {comment[:100]}...")
                
                with col2:
                    st.write("**Top Negative Comments:**")
                    negative_comments = df[df['Sentiment'] == 'Negative']['Comment'].head(5)
                    for i, comment in enumerate(negative_comments, 1):
                        st.write(f"{i}. {comment[:100]}...")
                
                # Word Analysis
                st.subheader("🔤 Most Common Words")
                all_words = ' '.join(df['Processed_Comment']).split()
                word_freq = Counter(all_words).most_common(20)
                
                if word_freq:
                    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=word_df.head(10), x='Frequency', y='Word', palette='viridis')
                    plt.title('Top 10 Most Common Words')
                    st.pyplot(fig)
                
                # Spam Detection Results
                st.subheader("🚩 Spam Detection")
                spam_counts = df['Spam'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    spam_counts.plot(kind='bar', ax=ax, color=['#90EE90', '#FFB6C1'])
                    plt.title('Spam vs Non-Spam Comments')
                    plt.xlabel('Comment Type')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    spam_comments = df[df['Spam'] == 'Spam']
                    if not spam_comments.empty:
                        st.write("**Detected Spam Comments:**")
                        for comment in spam_comments['Comment'].head(3):
                            st.write(f"• {comment[:80]}...")
                    else:
                        st.success("🎉 No spam comments detected!")
                
                # Time Series Analysis
                if 'Timestamp' in df.columns:
                    st.subheader("📅 Timeline Analysis")
                    df['Date'] = df['Timestamp'].dt.date
                    daily_sentiments = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
                    
                    if not daily_sentiments.empty:
                        st.line_chart(daily_sentiments)
                
                # Summary Statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Comments", len(df))
                
                with col2:
                    positive_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
                    st.metric("Positive %", f"{positive_pct:.1f}%")
                
                with col3:
                    negative_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100
                    st.metric("Negative %", f"{negative_pct:.1f}%")
                
                with col4:
                    spam_pct = (spam_counts.get('Spam', 0) / len(df)) * 100
                    st.metric("Spam %", f"{spam_pct:.1f}%")
                
                # Raw Data
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df[['User', 'Comment', 'Sentiment', 'Spam', 'Timestamp']])
                
            else:
                st.error("❌ No comments found for this video. The video might be private, have comments disabled, or the API quota might be exceeded.")
        else:
            st.error('❌ Invalid YouTube URL. Please check the URL format.')
else:
    st.info("👆 Please enter a YouTube URL above to start the analysis.")

# Footer
st.markdown("---")
st.markdown("**Note:** This app uses basic sentiment analysis and spam detection. For more accurate results, ensure all required packages are installed.")
