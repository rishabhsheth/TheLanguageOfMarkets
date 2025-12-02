import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Define the path to your .pkl file
file_path = os.getenv("PICKLE_PATH")

# Load the data from the .pkl file into a pandas DataFrame
df = pd.read_pickle(file_path)

# You can now work with the DataFrame
print(df.head()) 

print(df.info)

