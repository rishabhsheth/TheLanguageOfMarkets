import pandas as pd
import os
from dotenv import load_dotenv
import re

load_dotenv()

file_path = os.getenv("PICKLE_PATH")
df = pd.read_pickle(file_path)

TEXT_COL = "transcript"

def split_sections(text: str):
    if not isinstance(text, str):
        return "", ""
    
    text_lower = text.lower()
    
    pattern = r"questions\s*(?:and|&)\s*answers:"
    match = re.split(pattern, text_lower, maxsplit=1)
    
    if len(match) == 2:
        prepared, qa = match
        return prepared.strip(), qa.strip()
    else:
        # No match found
        pattern = r"(questions\s*(?:and|&)\s*answers:?)|(q\s*(?:and|&)\s*a:?)"

        parts = re.split(pattern, text_lower, maxsplit=1)
    
        if len(parts) >= 4 and (parts[1] or parts[2]):
            prepared = parts[0].strip()
            qa = parts[-1].strip()
            return prepared, qa
        
        return text_lower.strip(), ""

# Apply function and create two new columns
df[["prepared", "qa"]] = df[TEXT_COL].apply(lambda x: pd.Series(split_sections(x)))

# Preview results
print(df.head())


problem_rows = df[(df["prepared"] == "") | (df["qa"] == "")]
print(f"Number of rows with empty prepared & qa: {len(problem_rows)}")
print(problem_rows.head())
print(df.columns)

df.to_pickle("data/processed_data")

print("Done! Processed data and added prepared and qa columns.")
