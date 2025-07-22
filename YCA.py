#!/usr/bin/env python
# coding: utf-8

import sys
import subprocess
from googleapiclient.discovery import build
import streamlit as st
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
import joblib
import os
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Page Configuration
st.set_page_config(
    page_title='Vibes Pie - YouTube Sentiment Analysis',
    page_icon='🎭',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class YouTubeSentimentAnalyzer:
    def __init__(self):
        self.API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your API key
        self.youtube = build('youtube', 'v3', developerKey=self.API_KEY)
        self.sia = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.spam_detector_model = None
        self.load_spam_model()
    
    def extract_video_id(self, url):
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
    
    def get_youtube_comments(self, video_id, max_results=100):
        """Fetch YouTube comments with error handling"""
        try:
            comments = []
            timestamps = []
            users = []
            
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=max_results
            )
            
            response = request.execute()
            
            for item in response['items']:
                comment_data = item['snippet']['topLevelComment']['snippet']
                comments.append(comment_data['textDisplay'])
                timestamps.append(comment_data['publishedAt'])
                users.append(comment_data['authorDisplayName'])
            
            return pd.DataFrame({
                'User': users,
                'Comment': comments,
                'Timestamp': pd.to_datetime(timestamps)
            })
            
        except Exception as e:
            st.error(f"Error fetching comments: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        try:
            stop_words = set(stopwords.words('english'))
            text = ' '.join(word for word in text.split() if word not in stop_words)
        except:
            pass
        return text
    
    def detect_sentiment(self, comment):
        """Detect sentiment using VADER"""
        score = self.sia.polarity_scores(comment)
        if score['compound'] > 0.05:
            return 'Positive'
        elif score['compound'] < -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    @st.cache_resource
    def load_spam_model(_self):
        """Load or train spam detection model"""
        try:
            if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("spam_detector_model.pkl"):
                vectorizer = joblib.load("tfidf_vectorizer.pkl")
                model = joblib.load("spam_detector_model.pkl")
                return vectorizer, model
            else:
                # Create a simple spam detector for demo purposes
                from sklearn.naive_bayes import MultinomialNB
                
                # Sample data for training
                sample_data = {
                    'text': [
                        'Check out my channel', 'Subscribe to my channel', 'Like and subscribe',
                        'Great video!', 'Thanks for sharing', 'Very informative',
                        'First!', 'Early!', 'Nice content'
                    ],
                    'label': [1, 1, 1, 0, 0, 0, 1, 1, 0]  # 1 = spam, 0 = not spam
                }
                
                df_sample = pd.DataFrame(sample_data)
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform(df_sample['text'])
                
                model = MultinomialNB()
                model.fit(X, df_sample['label'])
                
                # Save models
                joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                joblib.dump(model, "spam_detector_model.pkl")
                
                return vectorizer, model
                
        except Exception as e:
            st.warning(f"Could not load spam model: {e}")
            return None, None
    
    def detect_spam(self, comment):
        """Detect if comment is spam"""
        if self.tfidf_vectorizer and self.spam_detector_model:
            try:
                comment_transformed = self.tfidf_vectorizer.transform([comment])
                prediction = self.spam_detector_model.predict(comment_transformed)
                return 'Spam' if prediction[0] == 1 else 'Not Spam'
            except:
                return 'Not Spam'
        return 'Not Spam'

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Vibes Pie</h1>
        <h3>YouTube Sentiment Analysis Dashboard</h3>
        <p>Unmasking the true sentiments through comments!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = YouTubeSentimentAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Analysis Settings")
        
        # Video URL input
        video_url = st.text_input(
            "📺 YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the full YouTube video URL here"
        )
        
        # Max comments slider
        max_comments = st.slider(
            "📊 Maximum Comments to Analyze:",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="More comments = more accurate analysis but slower processing"
        )
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Start Analysis",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📋 How to use:
        1. Paste a YouTube video URL
        2. Adjust the number of comments
        3. Click 'Start Analysis'
        4. Explore the results below!
        """)
    
    # Main content area
    if analyze_button and video_url:
        video_id = analyzer.extract_video_id(video_url)
        
        if not video_id:
            st.markdown("""
            <div class="error-message">
                ❌ <strong>Invalid YouTube URL</strong><br>
                Please make sure you've entered a valid YouTube video URL.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch comments
        status_text.text("🔍 Fetching comments...")
        progress_bar.progress(20)
        
        df = analyzer.get_youtube_comments(video_id, max_comments)
        
        if df.empty:
            st.markdown("""
            <div class="error-message">
                ❌ <strong>No comments found</strong><br>
                This video might have comments disabled or no comments yet.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Process comments
        status_text.text("🔄 Processing comments...")
        progress_bar.progress(40)
        
        df['Processed_Comment'] = df['Comment'].apply(analyzer.preprocess_text)
        
        # Sentiment analysis
        status_text.text("🎭 Analyzing sentiments...")
        progress_bar.progress(60)
        
        df['Sentiment'] = df['Processed_Comment'].apply(analyzer.detect_sentiment)
        
        # Spam detection
        status_text.text("🚫 Detecting spam...")
        progress_bar.progress(80)
        
        if analyzer.tfidf_vectorizer and analyzer.spam_detector_model:
            df['Spam'] = df['Comment'].apply(analyzer.detect_spam)
        else:
            df['Spam'] = 'Not Spam'
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(df, analyzer)
    
    elif not video_url:
        # Welcome message
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to Vibes Pie!</h3>
            <p>Get started by entering a YouTube video URL in the sidebar and clicking "Start Analysis".</p>
            <p><strong>What you'll get:</strong></p>
            <ul>
                <li>📊 Sentiment distribution of comments</li>
                <li>☁️ Word clouds for positive and negative sentiments</li>
                <li>📈 Time-series analysis of comment sentiments</li>
                <li>🚫 Spam detection and analysis</li>
                <li>📋 Detailed metrics and insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_results(df, analyzer):
    """Display analysis results with improved UI"""
    
    # Key metrics
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_comments = len(df)
    sentiment_counts = df['Sentiment'].value_counts()
    spam_count = len(df[df['Spam'] == 'Spam'])
    
    with col1:
        st.metric(
            label="Total Comments",
            value=total_comments,
            delta=None
        )
    
    with col2:
        positive_pct = (sentiment_counts.get('Positive', 0) / total_comments * 100)
        st.metric(
            label="Positive Sentiment",
            value=f"{positive_pct:.1f}%",
            delta=f"{sentiment_counts.get('Positive', 0)} comments"
        )
    
    with col3:
        negative_pct = (sentiment_counts.get('Negative', 0) / total_comments * 100)
        st.metric(
            label="Negative Sentiment",
            value=f"{negative_pct:.1f}%",
            delta=f"{sentiment_counts.get('Negative', 0)} comments"
        )
    
    with col4:
        spam_pct = (spam_count / total_comments * 100)
        st.metric(
            label="Spam Detection",
            value=f"{spam_pct:.1f}%",
            delta=f"{spam_count} spam comments"
        )
    
    st.markdown("---")
    
    # Sentiment Analysis Overview
    st.header("🎭 Sentiment Analysis Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive pie chart
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#f39c12'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Interactive bar chart
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Comments by Sentiment",
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#f39c12'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Top Comments
    st.header("💬 Top Comments by Sentiment")
    
    tab1, tab2, tab3 = st.tabs(["😊 Positive", "😞 Negative", "😐 Neutral"])
    
    with tab1:
        positive_comments = df[df['Sentiment'] == 'Positive'].head(10)
        if not positive_comments.empty:
            for idx, row in positive_comments.iterrows():
                st.markdown(f"""
                <div style="background: #e8f5e8; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; border-left: 4px solid #2ecc71;">
                    <strong>👤 {row['User']}</strong><br>
                    {row['Comment'][:200]}{'...' if len(row['Comment']) > 200 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No positive comments found.")
    
    with tab2:
        negative_comments = df[df['Sentiment'] == 'Negative'].head(10)
        if not negative_comments.empty:
            for idx, row in negative_comments.iterrows():
                st.markdown(f"""
                <div style="background: #fdeaea; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; border-left: 4px solid #e74c3c;">
                    <strong>👤 {row['User']}</strong><br>
                    {row['Comment'][:200]}{'...' if len(row['Comment']) > 200 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No negative comments found.")
    
    with tab3:
        neutral_comments = df[df['Sentiment'] == 'Neutral'].head(10)
        if not neutral_comments.empty:
            for idx, row in neutral_comments.iterrows():
                st.markdown(f"""
                <div style="background: #fef9e7; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; border-left: 4px solid #f39c12;">
                    <strong>👤 {row['User']}</strong><br>
                    {row['Comment'][:200]}{'...' if len(row['Comment']) > 200 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No neutral comments found.")
    
    # Word Analysis
    st.header("☁️ Word Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Common Words")
        all_words = ' '.join(df['Processed_Comment']).split()
        word_freq = Counter(all_words).most_common(20)
        
        if word_freq:
            words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            fig_words = px.bar(
                words_df.head(10),
                x='Frequency',
                y='Word',
                orientation='h',
                title="Top 10 Most Common Words"
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
    
    with col2:
        st.subheader("Word Cloud")
        if not df['Processed_Comment'].empty:
            try:
                all_text = ' '.join(df['Processed_Comment'])
                if all_text.strip():
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color='white',
                        colormap='viridis'
                    ).generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Not enough text data for word cloud.")
            except Exception as e:
                st.error(f"Could not generate word cloud: {e}")
    
    # Time Series Analysis
    if 'Timestamp' in df.columns:
        st.header("📈 Time Series Analysis")
        
        df['Date'] = df['Timestamp'].dt.date
        time_series = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
        
        if not time_series.empty:
            fig_time = px.line(
                time_series,
                title="Sentiment Trends Over Time",
                labels={'value': 'Number of Comments', 'Date': 'Date'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
    
    # Spam Analysis
    st.header("🚫 Spam Analysis")
    
    spam_comments = df[df['Spam'] == 'Spam']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not spam_comments.empty:
            st.subheader("Detected Spam Comments")
            spam_display = spam_comments[['User', 'Comment']].head(10)
            st.dataframe(spam_display, use_container_width=True)
        else:
            st.success("🎉 No spam comments detected!")
    
    with col2:
        if not spam_comments.empty:
            st.subheader("Top Spam Users")
            top_spammers = spam_comments['User'].value_counts().head(10)
            if not top_spammers.empty:
                fig_spam = px.bar(
                    x=top_spammers.values,
                    y=top_spammers.index,
                    orientation='h',
                    title="Users with Most Spam Comments"
                )
                fig_spam.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_spam, use_container_width=True)
    
    # Download data
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Analysis (CSV)",
            data=csv,
            file_name="youtube_sentiment_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        summary_data = {
            'Metric': ['Total Comments', 'Positive %', 'Negative %', 'Neutral %', 'Spam %'],
            'Value': [
                total_comments,
                f"{positive_pct:.1f}%",
                f"{negative_pct:.1f}%",
                f"{(sentiment_counts.get('Neutral', 0) / total_comments * 100):.1f}%",
                f"{spam_pct:.1f}%"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary (CSV)",
            data=summary_csv,
            file_name="youtube_analysis_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
