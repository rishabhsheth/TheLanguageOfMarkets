import pandas as pd
import os


# df = pd.read_pickle("data/processed_data_sampled")

# # Extract ONLY the first token (before any space)
# df["exchange"] = df["exchange"].str.split().str[0]

# # Remove parentheses and everything inside them
# df["exchange"] = df["exchange"].str.replace(r"\(.*?\)", "", regex=True)

# # Remove trailing punctuation like : or :
# df["exchange"] = df["exchange"].str.replace(r"[:;,\-]+$", "", regex=True)

# # Strip whitespace
# df["exchange"] = df["exchange"].str.strip()

# print(df.value_counts("exchange"))

# successful_tickers = pd.read_pickle("data/price_cache.pkl")
# print(len(successful_tickers))

# failed_tickers = pd.read_pickle("data/failed_tickers.pkl")
# print(len(failed_tickers))

df = pd.read_pickle("data\processed_data_sampled_with_finbert_fast.pkl")

# df = pd.read_pickle("data/processed_data_sampled_with_vader_fast.pkl")

print(df.columns)

# print(df[['ticker', 'transcript_finbert_label', 'transcript_finbert_score',
#        'prepared_finbert_label', 'prepared_finbert_score', 'qa_finbert_label',
#        'qa_finbert_score']][15:30])

print(df['transcript_finbert_label'].value_counts())
print(df['prepared_finbert_label'].value_counts())
print(df['qa_finbert_label'].value_counts())

# print(df['transcript_score'].value_counts())
# print(df['prepared_score'].value_counts())
# print(df['qa_score'].value_counts())