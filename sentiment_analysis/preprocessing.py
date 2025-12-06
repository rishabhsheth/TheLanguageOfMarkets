import pandas as pd
import os
from dotenv import load_dotenv
import re
import numpy as np

# Load environment variables
load_dotenv()

file_path = os.getenv("PICKLE_PATH")
df = pd.read_pickle(file_path)

TEXT_COL = "transcript"

# --- Transcript splitting ---
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
        # fallback patterns
        pattern = r"(questions\s*(?:and|&)\s*answers:?)|(q\s*(?:and|&)\s*a:?)"
        parts = re.split(pattern, text_lower, maxsplit=1)
    
        if len(parts) >= 4 and (parts[1] or parts[2]):
            prepared = parts[0].strip()
            qa = parts[-1].strip()
            return prepared, qa
        
        return text_lower.strip(), ""

df[["prepared", "qa"]] = df[TEXT_COL].apply(lambda x: pd.Series(split_sections(x)))

# --- Date cleaning & parsing ---
def clean_and_parse_date(value):
    """
    Cleans messy strings and parses into pd.Timestamp.
    Handles:
      - Extra text before/after the date
      - Quarter mentions like Q4 2018
      - "ET" timezone
      - "July. 31, 2019" style
      - Lists/arrays in a single cell
    """
    # If value is a list, Series, Index, or ndarray, try each element recursively
    if isinstance(value, (list, pd.Series, pd.Index, np.ndarray)):
        for v in value:
            dt = clean_and_parse_date(v)
            if pd.notna(dt):
                return dt
        return pd.NaT

    # Now value is scalar — safe to check for NaN
    if pd.isna(value):
        return pd.NaT

    s = str(value)
    s = re.sub(r'\s*\.?\s*ET\.?$', '', s, flags=re.I)
    s = s.replace("\xa0", " ").replace("\u200b", "").strip()

    # Quarter handling: Q1-Q4 YYYY -> first day of quarter
    q = re.search(r"Q([1-4])\s*[,]?\s*(\d{4})", s, flags=re.I)
    if q:
        qn, year = int(q.group(1)), int(q.group(2))
        month = 1 + (qn - 1) * 3
        return pd.Timestamp(year=year, month=month, day=1)

    # Extract month-day-year (+ optional time)
    m = re.search(
        r"([A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4}(?:,\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?)?)",
        s
    )
    if m:
        s = m.group(1).replace(".", "")

    try:
        dt = pd.to_datetime(s, errors='coerce')
    except Exception:
        dt = pd.NaT

    return dt

df['date_parsed'] = df['date'].apply(clean_and_parse_date)

# Report any failed rows
failed_dates = df['date_parsed'].isna().sum()
print(f"Still failing after cleaning: {failed_dates}")

# --- Save full processed data ---
os.makedirs("data", exist_ok=True)
df.to_pickle("data/processed_data")
print("Done! Processed data with prepared, qa, and date_parsed columns.")

# --- Sample 25% for testing ---
sampled_df = df.sample(frac=0.25, random_state=123)
sampled_df.to_pickle("data/processed_data_sampled")
print("Done! Created sampled version (25% of rows).")

# # Rows where parsing failed
# failed_rows = df[df['date_parsed'].isna()]

# # Print the original 'date' column and optionally other columns
# print(failed_rows[['date']])

# # If you want to see the full row(s)
# print(failed_rows)
