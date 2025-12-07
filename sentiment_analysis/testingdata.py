import pandas as pd
import os

df = pd.read_pickle("data/processed_data")

# Extract ONLY the first token (before any space)
df["exchange"] = df["exchange"].str.split().str[0]

# Remove parentheses and everything inside them
df["exchange"] = df["exchange"].str.replace(r"\(.*?\)", "", regex=True)

# Remove trailing punctuation like : or :
df["exchange"] = df["exchange"].str.replace(r"[:;,\-]+$", "", regex=True)

# Strip whitespace
df["exchange"] = df["exchange"].str.strip()

print(df.value_counts("exchange"))
