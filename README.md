# 🎭 Vibes Pie - YouTube Sentiment Analysis Dashboard

A modern, interactive dashboard for analyzing YouTube video comments sentiment, spam detection, and engagement insights.

## Features

- **🎭 Sentiment Analysis**: Analyze positive, negative, and neutral sentiments in comments
- **🚫 Spam Detection**: Identify and filter spam comments automatically  
- **☁️ Word Clouds**: Visual representation of most common words
- **📈 Time Series Analysis**: Track sentiment trends over time
- **📊 Interactive Visualizations**: Modern charts and graphs using Plotly
- **💾 Data Export**: Download analysis results as CSV files
- **📱 Responsive Design**: Works on desktop and mobile devices

## Setup

1. Install dependencies:
bash
pip install -r requirements.txt


2. Get a YouTube Data API key from Google Cloud Console

3. Replace the API_KEY in the code with your key

4. Run the application:
bash
streamlit run app.py

## Usage

1. Enter a YouTube video URL in the sidebar
2. Adjust the number of comments to analyze
3. Click "Start Analysis" 
4. Explore the interactive results and insights
5. Download the data for further analysis

## Technologies Used

- **Streamlit**: Web application framework
- **YouTube Data API**: Fetching video comments
- **NLTK**: Natural language processing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning for spam detection
- **Pandas**: Data manipulation and analysis

