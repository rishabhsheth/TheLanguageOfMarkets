import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import os

# --- Load sampled data ---
file_path = "data/processed_data_sampled_with_prices_ultimate.pkl"
df = pd.read_pickle(file_path)

df.dropna(subset=['adj_close', 'adj_close_1d_before',
       'adj_close_3d_before', 'adj_close_5d_before', 'adj_close_1d_after',
       'adj_close_3d_after', 'adj_close_5d_after', 'ret_1d_after',
       'ret_3d_after', 'ret_5d_after', 'ret_1d_before', 'ret_3d_before',
       'ret_5d_before'], inplace=True)


# --- Initialize VADER ---
vader = SentimentIntensityAnalyzer()

# --- Function to get compound score ---
def vader_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return float('nan')
    return vader.polarity_scores(text)['compound']

# --- Apply VADER with progress bar ---
for col in ['transcript', 'prepared', 'qa']:
    print(f"Processing {col}...")
    tqdm.pandas(desc=f"Scoring {col}")
    df[f'{col}_score'] = df[col].progress_apply(vader_score)

# --- Save processed dataframe ---
os.makedirs("data", exist_ok=True)
df.to_pickle("data/processed_data_sampled_with_vader_fast.pkl")
print("Done! VADER sentiment scores added for all columns.")
