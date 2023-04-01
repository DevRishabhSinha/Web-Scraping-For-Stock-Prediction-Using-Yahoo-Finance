import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from fbprophet import Prophet

# Define a function to scrape news articles from a web page
def scrape_news(url):
    # Make a GET request to the URL and get the response content
    response = requests.get(url)
    content = response.content
    
    # Use BeautifulSoup to parse the HTML content and extract the news articles
    soup = BeautifulSoup(content, 'html.parser')
    articles = soup.find_all('article')
    
    # Create a list to store the article titles and text
    news = []
    
    # Loop through each article and extract the title and text
    for article in articles:
        title = article.find('h2').get_text()
        text = article.find('p').get_text()
        
        # Remove any leading/trailing whitespace and newlines
        title = title.strip().replace('\n', '')
        text = text.strip().replace('\n', '')
        
        # Append the title and text to the news list
        news.append((title, text))
    
    # Return the scraped news
    return news

# Define a function to analyze the sentiment of news articles
def analyze_sentiment(news):
    # Create a dictionary to store the sentiment scores for each article
    sentiment_scores = {}
    
    # Loop through each article and analyze the sentiment using TextBlob
    for title, text in news:
        # Create a TextBlob object with the article text
        blob = TextBlob(text)
        
        # Get the polarity score of the text
        # The polarity score is a float between -1 and 1, where negative values indicate negative sentiment, 
        # positive values indicate positive sentiment, and 0 indicates neutral sentiment
        sentiment_score = blob.sentiment.polarity
        
        # Store the sentiment score in the dictionary
        sentiment_scores[title] = sentiment_score
    
    # Return the sentiment scores
    return sentiment_scores

# Define the URL of the web page to scrape
url = "https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAACFzAXy0FKh4b-q90pU_aHawOcWGsDqXww7tKtvJioRD23Ugd1GE_h0MQS2KMbbhEqsE9f1BahSXWwxJ2klbtw86aL2aT4kOKGUOk2oh6Ygg1lmHVsi6I02R7QuvuNz-pm33p7jM02JnaIVxsyb9HzHT-AtnfqGf78P3cMq4o0Dq"

# Scrape the news articles from the web page
news = scrape_news(url)

# Analyze the sentiment of the news articles
sentiment_scores = analyze_sentiment(news)

# Create a DataFrame with the sentiment scores
df = pd.DataFrame(list(sentiment_scores.items()), columns=['ds', 'y'])

# Initialize a Prophet model and fit it to the sentiment scores data
model = Prophet()
model.fit(df)

# Create a future DataFrame with dates to predict
future = model.make_future_dataframe(periods=30)

# Make predictions for the future dates
forecast = model.predict(future)

# Print the predicted sentiment trend and stock price trend
sentiment_trend = forecast[['ds', 'yhat']].tail(30)
stock_trend = forecast[['ds', 'yhat_upper']].tail(30)
print("Predicted sentiment trend:\n", sentiment_trend)
print("\nPredicted stock price trend:\n", stock_trend)
