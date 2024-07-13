import pandas as pd #for data manipulation
import re #regular expression for text processing such as sorting,searching and replacing text patterns
from nltk.corpus import stopwords #common words used out for text processing
from nltk.tokenize import word_tokenize # spliting for text in words
from textblob import TextBlob #NLP library for processing textual data
import matplotlib.pyplot as plt # for ploting graph
import nltk #natural language toolkit

# Download necessary NLTK data
nltk.download('stopwords') #common words used out for text processing
nltk.download('punkt') #punkt is pre-trained model for tokenizating text

# Simulated data with consistent lengths
data = {
    'platform': ['Instagram', 'Twitter', 'Facebook', 'Snapchat', 'YouTube',
                 'WhatsApp', 'Quora', 'LinkedIn', 'Telegram', 'TikTok',
                 'Reddit', 'Pinterest', 'Tumblr', 'Flickr', 'VKontakte',
                 'WeChat', 'Line', 'Weibo', 'Viber', 'Discord',
                 'Signal', 'Skype', 'Mixi', 'Sina Weibo', 'Qzone'],
    'user_id': [101, 102, 103, 104, 105,
                106, 107, 108, 109, 110,
                111, 112, 113, 114, 115,
                116, 117, 118, 119, 120,
                121, 122, 123, 124, 125],
    'post_text': [
        'Feeling really down today. Can’t seem to shake it off. 😔 #depression',
        'Just finished a stressful week at work. Need to relax now. #stress',
        'Excited about my new job! Feeling positive about the future. 😊',
        'Snapchat post about a fun outing with friends.',
        'Uploaded a video about travel tips on YouTube!',
        'WhatsApp message about upcoming event.',
        'Answering a question on Quora about mental health.',
        'Updated LinkedIn profile with new achievements.',
        'Telegram message to friends about weekend plans.',
        'Funny TikTok video to cheer everyone up!',
        'Reddit post discussing anxiety management tips.',
        'Pin board on Pinterest with self-care ideas.',
        'Tumblr post about mindfulness and meditation.',
        'Flickr photo album of nature walks.',
        'VKontakte post sharing a music playlist.',
        'WeChat message about a new movie release.',
        'Line sticker set for expressing emotions.',
        'Weibo post discussing mental health awareness.',
        'Viber group chat about stress at work.',
        'Discord server discussing gaming and mental health.',
        'Signal message to friends about a meetup.',
        'Skype call with family members.',
        'Mixi blog post about hobbies and mental health.',
        'Sina Weibo post about positive thinking.',
        'Qzone diary entry about daily challenges.',
       
    ],
    'timestamp': [
        '2024-06-30 10:00:00', '2024-06-29 15:30:00', '2024-06-28 08:45:00',
        '2024-06-27 12:00:00', '2024-06-26 18:20:00',
        '2024-06-25 09:00:00', '2024-06-24 14:30:00', '2024-06-23 11:15:00',
        '2024-06-22 17:45:00', '2024-06-21 20:00:00',
        '2024-06-20 16:30:00', '2024-06-19 13:45:00', '2024-06-18 19:00:00',
        '2024-06-17 10:30:00', '2024-06-16 08:00:00',
        '2024-06-15 21:45:00', '2024-06-14 22:20:00', '2024-06-13 12:15:00',
        '2024-06-12 18:30:00', '2024-06-11 14:00:00',
        '2024-06-10 11:30:00', '2024-06-09 09:45:00', '2024-06-08 07:00:00',
        '2024-06-07 15:20:00', '2024-06-06 16:45:00', 
    ]
}

# Ensure all arrays have the same length
num_records = len(data['platform'])
data['user_id'] = data['user_id'][:num_records]
data['post_text'] = data['post_text'][:num_records]
data['timestamp'] = data['timestamp'][:num_records]

# Create DataFrame
df = pd.DataFrame(data)  #convert dictionary into dataframe

# Preprocess text function(combining re ,tokenization,stopwords to clean and prepare data for analysis)
stop_words = set(stopwords.words('english')) #retrieve the list of english stopwords from nltk

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]  # Remove stopwords and lowercase
    return tokens

# Apply preprocessing to post_text column
df['processed_text'] = df['post_text'].apply(preprocess_text)

# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity #analysis of polarity of text 

# Apply sentiment analysis to post_text column
df['sentiment'] = df['post_text'].apply(get_sentiment)

# Simulated ground truth sentiment for demonstration (0 for negative, 1 for positive)
# Here we are making a simple assumption that posts with certain words are positive or negative
df['ground_truth_sentiment'] = df['post_text'].apply(lambda x: 1 if 'positive' in x or '😊' in x else 0)

# Convert predicted sentiment to binary (0 for negative, 1 for positive)
df['predicted_sentiment'] = df['sentiment'].apply(lambda x: 1 if x > 0 else 0)

# Calculate accuracy by comparing ground truth sentiment and predicted sentiment
accuracy = (df['ground_truth_sentiment'] == df['predicted_sentiment']).mean()

# Display sentiment analysis results and accuracy
print(df[['post_text', 'sentiment', 'ground_truth_sentiment', 'predicted_sentiment']])
print(f"Accuracy: {accuracy:.2f}")

# Plotting sentiment analysis results
plt.figure(figsize=(12, 8))
plt.bar(df['platform'], df['sentiment'], color='skyblue')
plt.xlabel('Platform')
plt.ylabel('Sentiment Polarity')
plt.title('Sentiment Analysis of Posts Across Platforms')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
