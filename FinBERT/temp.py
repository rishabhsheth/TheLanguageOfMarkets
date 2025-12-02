import pandas as pd

# Define the path to your .pkl file
file_path = '/Users/kuttan/CS439 - Final Project/TheLanguageOfMarkets/FinBERT/motley-fool-data.pkl' 

# Load the data from the .pkl file into a pandas DataFrame
df = pd.read_pickle(file_path)

# You can now work with the DataFrame

print(df.iloc[850]['transcript'])

'''
def split_sections(text):
    text_lower = text.lower()
    if "questions & answers:" in text_lower:
        parts = text_lower.split("questions & answers")
        prepared = parts[0]
        qa = parts[1]
    else:
        prepared = text_lower
        qa = ""
    return prepared, qa

'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


finbert_sentiment = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

text = "We are extremely confident about our earnings this quarter."
result = finbert_sentiment(text)

print(result)

text = "We are very confident about our earnings this quarter."

score = vader.polarity_scores(text)
print(score)
