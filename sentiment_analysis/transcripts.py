import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Define the path to your .pkl file
file_path = os.getenv("PICKLE_PATH")

# Load the data from the .pkl file into a pandas DataFrame
df = pd.read_pickle(file_path)

# You can now work with the DataFrame
# print(df.head()) 

# print(df.info)


print(df.iloc[850]['transcript'])

def split_sections(text):
    text_lower = text.lower()
    prepared, qa = text_lower.split("questions and answers:", 1)
    # if "questions and answers:" in text_lower:
    #     parts = text_lower.split("questions and answers")
    #     prepared = parts[0]
    #     qa = parts[1]
    # else:
    #     prepared = text_lower
    #     qa = ""
    return prepared, qa

prepared, qa = split_sections(df.iloc[850]['transcript'])

print("1. Prepared: ", prepared)

print("2. QA: ", qa)
