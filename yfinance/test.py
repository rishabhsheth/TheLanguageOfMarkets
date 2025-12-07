import pandas as pd

# Load the pickle
df = pd.read_pickle("data/processed_data_sampled_with_prices_fast.pkl")  # or your pickle path

# Print all column names
print(df.columns)

# Optionally, get as a list
columns_list = df.columns.tolist()
print(columns_list)


print(df.iloc[445])