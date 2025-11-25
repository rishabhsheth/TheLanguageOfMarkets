import pandas as pd

# Define the path to your .pkl file
file_path = '/Users/kuttan/CS439 - Final Project/TheLanguageOfMarkets/FinBERT/motley-fool-data.pkl' 

# Load the data from the .pkl file into a pandas DataFrame
df = pd.read_pickle(file_path)

# You can now work with the DataFrame
print(df.head()) 

print(df.info)

