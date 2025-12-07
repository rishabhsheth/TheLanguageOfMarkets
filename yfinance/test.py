import pandas as pd

# Load the pickle
df = pd.read_pickle("data/processed_data_sampled_with_prices_ultimate.pkl")  # or your pickle path

# Print all column names
print(df.columns)

print(df.shape)

# # Optionally, get as a list
# columns_list = df.columns.tolist()
# print(columns_list)


# print(df.iloc[445])


df.dropna(subset=['adj_close', 'adj_close_1d_before',
       'adj_close_3d_before', 'adj_close_5d_before', 'adj_close_1d_after',
       'adj_close_3d_after', 'adj_close_5d_after', 'ret_1d_after',
       'ret_3d_after', 'ret_5d_after', 'ret_1d_before', 'ret_3d_before',
       'ret_5d_before'], inplace=True)

print(df.shape)