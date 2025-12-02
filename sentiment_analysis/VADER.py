from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transcripts import split_sections

vader = SentimentIntensityAnalyzer()

score = vader.polarity_scores(text)

print(score)
