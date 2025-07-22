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
nltk.download('stopwords', quiet=True)
import re
import joblib
import os
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# Initialize Streamlit App
st.set_page_config(page_title='Vibes Pie - YouTube Sentiment Analysis', layout='wide')
st.title('Vibes Pie - YouTube Sentiment Analysis Dashboard')
st.write('Unmasking the true sentiments through comments!')

# YouTube API Key and Configurations
API_KEY = 'AIzaSyD5-RtE9nM-wgOXCSnQsmz6CuN4dnDJ7bE'  # Replace with your own YouTube API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Function Definitions (moved to top)
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
    """Data Preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

@st.cache_resource
def load_spam_model():
    """Load the pre-trained spam detection model and vectorizer"""
    try:
        # Check if the files exist in the current directory
        if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("spam_detector_model.pkl"):
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            model = joblib.load("spam_detector_model.pkl")
            return vectorizer, model
        else:
            # Train a new model if pre-trained models don't exist
            st.info("Pre-trained models not found. Training a new model...")
            
            # Sample Spam Dataset
            data = pd.read_csv(
                "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv",
                sep='\t',
                header=None
            )
            data.columns = ['Label', 'Message']
            
            # Vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(data['Message'])
            y = data['Label'].map({'ham': 0, 'spam': 1})
            
            # Train-Test Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Model Training
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            # Save the model for future use
            joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
            joblib.dump(model, "spam_detector_model.pkl")
            
            # Model Evaluation
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.success(f"Spam Detection Model trained with an accuracy of: {accuracy:.2f}")
            
            return vectorizer, model
            
    except Exception as e:
        st.error(f"🚫 Error loading models: {e}")
        return None, None

def detect_spam(comment, vectorizer, model):
    """Use the pre-trained model to detect if a comment is spam or not"""
    if vectorizer is None or model is None:
        return 'Unknown'
    
    comment_transformed = vectorizer.transform([comment])
    prediction = model.predict(comment_transformed)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

def detect_sentiment(comment):
    """Function to determine sentiment using VADER"""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(comment)
    if score['compound'] > 0.05:
        return 'Positive'
    elif score['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# User Input for YouTube Video URL
video_url = st.text_input('Enter YouTube Video URL:', '')

# Only show the button if there's input
if video_url:
    # Button to fetch comments
    if st.button("Fetch Comments"):
        video_id = extract_video_id(video_url)
        
        if video_id:
            st.success(f'Video ID extracted: {video_id}')
            
            # Fetch comments
            with st.spinner('Fetching comments...'):
                df = get_youtube_comments(video_id)
            
            if not df.empty:
                st.success("Comments fetched successfully!")
                
                # Preprocess comments
                df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
                
                # Apply sentiment detection
                with st.spinner('Analyzing sentiments...'):
                    df['Sentiment'] = df['Processed_Comment'].apply(detect_sentiment)
                
                # Load spam detection model
                tfidf_vectorizer, spam_detector_model = load_spam_model()
                
                # Apply spam detection
                if tfidf_vectorizer is not None and spam_detector_model is not None:
                    with st.spinner('Detecting spam comments...'):
                        df['Spam'] = df['Comment'].apply(lambda x: detect_spam(x, tfidf_vectorizer, spam_detector_model))
                
                # Display sentiment distribution
                st.subheader('Sentiment Analysis Overview')
                sentiment_counts = df['Sentiment'].value_counts()
                
                # Creating Columns for Side by Side Display
                col1, col2 = st.columns(2)
                
                # Sentiment Distribution Bar Chart
                with col1:
                    plt.figure(figsize=(4, 4))
                    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
                    plt.title("Number of Comments per Sentiment")
                    plt.ylabel('Count')
                    plt.xlabel('Sentiment')
                    st.pyplot(plt)
                
                # Sentiment Split Pie Chart
                with col2:
                    plt.figure(figsize=(4, 4))
                    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                            colors=['#66b3ff', '#99ff99', '#ff9999'])
                    plt.gca().set_aspect('equal')
                    st.pyplot(plt)
                
                # Top 10 Positive and Negative Comments
                positive_comments = df[df['Sentiment'] == 'Positive']['Comment'].head(10).reset_index(drop=True)
                negative_comments = df[df['Sentiment'] == 'Negative']['Comment'].head(10).reset_index(drop=True)
                
                # Combine into a DataFrame for display
                max_len = max(len(positive_comments), len(negative_comments))
                positive_comments = positive_comments.reindex(range(max_len)).fillna('')
                negative_comments = negative_comments.reindex(range(max_len)).fillna('')
                
                comments_df = pd.DataFrame({
                    "Top 10 Positive Comments": positive_comments,
                    "Top 10 Negative Comments": negative_comments
                })
                
                # Creating Columns for Side by Side Display
                col1, col2 = st.columns(2)
                
                # Display in respective columns
                with col1:
                    st.markdown("### Top 10 Positive Comments")
                    st.dataframe(comments_df[['Top 10 Positive Comments']], use_container_width=True)
                
                with col2:
                    st.markdown("### Top 10 Negative Comments")
                    st.dataframe(comments_df[['Top 10 Negative Comments']], use_container_width=True)
                
                # Most Common Words
                st.subheader('Most Common Words')
                common_words = Counter(' '.join(df['Processed_Comment']).split()).most_common(20)
                common_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
                st.write(common_df)
                
                # Time-Series Analysis
                st.subheader('Time-Series Analysis of Sentiments')
                df['Date'] = df['Timestamp'].dt.date
                time_series_data = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
                st.line_chart(time_series_data)
                
                # Confusion Matrix Analysis
                y_pred = df['Sentiment']
                y_true = ['Positive' if i % 3 == 0 else 'Negative' if i % 3 == 1 else 'Neutral' for i in range(len(df))]
                
                cm = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral'])
                total = cm.sum() if cm.size > 0 else 1
                
                # Create two columns for display
                col1, col2 = st.columns(2)
                
                # Display Confusion Matrix
                with col1:
                    st.subheader('Confusion Matrix')
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=['Positive', 'Negative', 'Neutral'],
                               yticklabels=['Positive', 'Negative', 'Neutral'])
                    plt.title('Confusion Matrix')
                    st.pyplot(plt)
                
                # Calculate metrics
                TP = [cm[i, i] for i in range(3)]
                FP = [cm[:, i].sum() - cm[i, i] for i in range(3)]
                FN = [cm[i, :].sum() - cm[i, i] for i in range(3)]
                TN = [total - (TP[i] + FP[i] + FN[i]) for i in range(3)]
                
                TP_pct = [round((x / total) * 100, 2) for x in TP]
                FP_pct = [round((x / total) * 100, 2) for x in FP]
                FN_pct = [round((x / total) * 100, 2) for x in FN]
                TN_pct = [round((x / total) * 100, 2) for x in TN]
                
                metrics_data = {
                    'Class': ['Positive', 'Negative', 'Neutral'],
                    'True Positive (TP)': TP,
                    'False Positive (FP)': FP,
                    'False Negative (FN)': FN,
                    'True Negative (TN)': TN,
                    'TP %': TP_pct,
                    'FP %': FP_pct,
                    'FN %': FN_pct,
                    'TN %': TN_pct
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                
                with col2:
                    st.subheader('Confusion Matrix Breakdown')
                    plt.figure(figsize=(5, 4))
                    
                    bar_width = 0.2
                    positions = np.arange(len(metrics_df['Class']))
                    
                    plt.bar(positions, metrics_df['True Positive (TP)'], width=bar_width, label='TP', color='green')
                    plt.bar(positions + bar_width, metrics_df['False Positive (FP)'], width=bar_width, label='FP', color='red')
                    plt.bar(positions + bar_width * 2, metrics_df['False Negative (FN)'], width=bar_width, label='FN', color='orange')
                    plt.bar(positions + bar_width * 3, metrics_df['True Negative (TN)'], width=bar_width, label='TN', color='blue')
                    
                    for i, pos in enumerate(positions):
                        plt.text(pos, metrics_df['True Positive (TP)'][i] + 1, f"{metrics_df['TP %'][i]}%", ha='center')
                        plt.text(pos + bar_width, metrics_df['False Positive (FP)'][i] + 1, f"{metrics_df['FP %'][i]}%", ha='center')
                        plt.text(pos + bar_width * 2, metrics_df['False Negative (FN)'][i] + 1, f"{metrics_df['FN %'][i]}%", ha='center')
                        plt.text(pos + bar_width * 3, metrics_df['True Negative (TN)'][i] + 1, f"{metrics_df['TN %'][i]}%", ha='center')
                    
                    plt.xticks(positions + bar_width * 1.5, metrics_df['Class'])
                    plt.legend()
                    plt.title('TP, FP, FN, TN Breakdown by Class')
                    st.pyplot(plt)
                
                # Display Metrics DataFrame
                st.subheader('Metrics DataFrame Preview')
                st.dataframe(metrics_df, use_container_width=True)
                
                # WordClouds Side by Side
                st.subheader('Word Clouds of Positive and Negative Comments')
                
                col1, col2 = st.columns(2)
                
                # Positive Word Cloud
                with col1:
                    st.markdown("### Positive Comments")
                    positive_words = ' '.join(df[df['Sentiment'] == 'Positive']['Processed_Comment'])
                    if positive_words.strip():
                        wordcloud = WordCloud(width=600, height=400).generate(positive_words)
                        plt.figure(figsize=(6, 4))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)
                    else:
                        st.info("No positive comments to generate word cloud")
                
                # Negative Word Cloud
                with col2:
                    st.markdown("### Negative Comments")
                    negative_words = ' '.join(df[df['Sentiment'] == 'Negative']['Processed_Comment'])
                    if negative_words.strip():
                        wordcloud = WordCloud(width=600, height=400).generate(negative_words)
                        plt.figure(figsize=(6, 4))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)
                    else:
                        st.info("No negative comments to generate word cloud")
                
                # Spam Detection Results (only show if spam detection was successful)
                if 'Spam' in df.columns:
                    st.subheader('🚩 Detected Spam Comments')
                    
                    col1, col2 = st.columns(2)
                    
                    # Display Spam Comments in Column 1
                    with col1:
                        spam_comments = df[df['Spam'] == 'Spam']
                        
                        if not spam_comments.empty:
                            st.markdown("### 🚫 Spam Comments and Usernames")
                            st.dataframe(spam_comments[['User', 'Comment']], use_container_width=True)
                        else:
                            st.success("No spam comments detected! 🎉")
                    
                    # Display Top Spam Commenters in Column 2
                    with col2:
                        st.subheader('🏆 Top Spam Commenters')
                        
                        if not spam_comments.empty:
                            top_spammers = spam_comments['User'].value_counts().head(10).reset_index()
                            top_spammers.columns = ['Username', 'Spam Count']
                            st.dataframe(top_spammers, use_container_width=True)
                        else:
                            st.success("No spammers found! 🎉")
                    
                    # Spam Detection Visualization
                    st.markdown("---")
                    vis_col1, vis_col2 = st.columns(2)
                    
                    with vis_col1:
                        st.markdown("### 📊 Spam Detection Overview")
                        spam_counts = df['Spam'].value_counts()
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.barplot(x=spam_counts.index, y=spam_counts.values, palette='Reds')
                        plt.title("Spam Detection Overview")
                        plt.ylabel('Number of Comments')
                        plt.xlabel('Comment Type')
                        st.pyplot(fig)
                
            else:
                st.error("No comments found for this video.")
        else:
            st.error('Invalid YouTube URL')
else:
    st.info("Please enter a YouTube URL above to get started.")
