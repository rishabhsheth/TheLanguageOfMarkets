import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import os

file_path = "data/processed_data_sampled_with_prices_ultimate.pkl"
df = pd.read_pickle(file_path)

df.dropna(subset=['adj_close', 'adj_close_1d_before',
       'adj_close_3d_before', 'adj_close_5d_before', 'adj_close_1d_after',
       'adj_close_3d_after', 'adj_close_5d_after', 'ret_1d_after',
       'ret_3d_after', 'ret_5d_after', 'ret_1d_before', 'ret_3d_before',
       'ret_5d_before'], inplace=True)

vader = SentimentIntensityAnalyzer()

def vader_dominant(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.NA, pd.NA
    scores = vader.polarity_scores(text)
    label_scores = {k: v for k, v in scores.items() if k in ["pos", "neu", "neg"]}
    dominant_label = max(label_scores, key=label_scores.get)
    dominant_score = label_scores[dominant_label]
    return dominant_label, dominant_score

for col in ['transcript', 'prepared', 'qa']:
    print(f"Processing {col}...")
    tqdm.pandas(desc=f"Scoring {col}")
    df[[f'{col}_vader_label', f'{col}_vader_score']] = df[col].progress_apply(
        lambda x: pd.Series(vader_dominant(x))
    )

os.makedirs("data", exist_ok=True)
df.to_pickle("data/processed_data_sampled_with_vader_fast.pkl")
print("Done! VADER labels and scores added for all columns.")