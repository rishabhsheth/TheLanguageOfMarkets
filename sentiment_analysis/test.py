import pandas as pd

pd.set_option("display.max_rows", None)  # show all rows
pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)  # auto width
pd.set_option("display.max_colwidth", None)  # do not truncate cell contents


df = pd.read_pickle("data/processed_with_finbert.pkl")

exclude_cols = ["transcript", "prepared", "qa"]

df_subset = df.drop(columns=exclude_cols)

print(df_subset)