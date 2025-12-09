import pandas as pd
import numpy as np
import os
import re
from dotenv import load_dotenv

load_dotenv()

file_path = os.getenv("PICKLE_PATH")
df = pd.read_pickle(file_path)

TEXT_COL = "transcript"

print(f"Initial data shape: {df.shape}")
print(df.head())

US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}

if "exchange" in df.columns:
    pattern = "|".join(US_EXCHANGES)   # -> "NYSE|NASDAQ|AMEX"
    df = df[df["exchange"].str.contains(pattern, case=False, na=False)]

print(f"Data shape after filtering exchanges: {df.shape}")

df = df[df["date"].notna()]

print(f"Data shape after removing missing dates: {df.shape}")

def clean_and_parse_date(value):
    """Attempts to extract and parse a valid timestamp from messy date formats."""

    if isinstance(value, (list, pd.Series, pd.Index, np.ndarray)):
        for v in value:
            dt = clean_and_parse_date(v)
            if pd.notna(dt):
                return dt
        return pd.NaT

    if pd.isna(value):
        return pd.NaT

    s = str(value)

    s = re.sub(r'\s*\.?\s*ET\.?$', '', s, flags=re.I)
    s = s.replace("\xa0", " ").replace("\u200b", "").strip()

    q = re.search(r"Q([1-4])\s*[,]?\s*(\d{4})", s, flags=re.I)
    if q:
        quarter = int(q.group(1))
        year = int(q.group(2))
        month = 1 + (quarter - 1) * 3
        return pd.Timestamp(year=year, month=month, day=1)

    m = re.search(
        r"([A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4}(?:,\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?)?)",
        s
    )
    if m:
        s = m.group(1).replace(".", "")

    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


df["date_parsed"] = df["date"].apply(clean_and_parse_date)

df = df[df["date_parsed"].notna()]

print(f"Data shape after date parsing: {df.shape}")

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
        pattern = r"(questions\s*(?:and|&)\s*answers:?)|(q\s*(?:and|&)\s*a:?)"
        parts = re.split(pattern, text_lower, maxsplit=1)
    
        if len(parts) >= 4 and (parts[1] or parts[2]):
            prepared = parts[0].strip()
            qa = parts[-1].strip()
            return prepared, qa
        
        return text_lower.strip(), ""

df[["prepared", "qa"]] = df[TEXT_COL].apply(lambda x: pd.Series(split_sections(x)))

os.makedirs("data", exist_ok=True)
df.to_pickle("data/processed_data")

sampled = df.sample(frac=0.25, random_state=123)
sampled.to_pickle("data/processed_data_sampled")

print("Done! Full dataset and 25% sampled dataset generated successfully.")
