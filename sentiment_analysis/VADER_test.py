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

# Only take first 3 rows for testing
df_test = df.head(5).copy()

vader = SentimentIntensityAnalyzer()

def vader_full_scores(text):
    if not isinstance(text, str) or text.strip() == "":
        return {'neg': pd.NA, 'neu': pd.NA, 'pos': pd.NA, 'compound': pd.NA}
    return vader.polarity_scores(text)

# Apply VADER and store full score dicts
for col in ['transcript', 'prepared', 'qa']:
    tqdm.pandas(desc=f"Scoring {col}")
    df_test[f'{col}_vader_scores'] = df_test[col].progress_apply(vader_full_scores)

# Print each row's full VADER scoring
for idx, row in df_test.iterrows():
    row_dict = {col: row[f'{col}_vader_scores'] for col in ['transcript', 'prepared', 'qa']}
    print(f"\nRow {idx} full VADER scores:\n", row_dict)

# Optional: make a DataFrame where each column is expanded
expanded_rows = []
for idx, row in df_test.iterrows():
    expanded_row = {}
    for col in ['transcript', 'prepared', 'qa']:
        scores = row[f'{col}_vader_scores']
        for score_label, score_val in scores.items():
            expanded_row[f"{col}_{score_label}"] = score_val
    expanded_rows.append(expanded_row)

vader_df = pd.DataFrame(expanded_rows)
print("\nExpanded DataFrame with full VADER scores:")
print(vader_df)
