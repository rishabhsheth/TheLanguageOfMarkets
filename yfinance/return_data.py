import pandas as pd
import yfinance as yf
import time
import os
from tqdm import tqdm

CACHE_FILE = "data/price_cache.pkl"
FAILED_TICKERS_FILE = "data/failed_tickers.pkl"

def clean_ticker(t):
    if not isinstance(t, str):
        return None
    t = t.strip().upper()
    t = t.replace(")", "").replace("(", "")
    t = t.replace(".B", "-B").replace(".A", "-A")
    return t


def safe_download(ticker, start, end, max_retries=5):
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                threads=False,
                auto_adjust=False
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if "YFTzMissingError" in str(e):
                return "YFTzMissingError"
            pass

        time.sleep(0.8 + attempt * 0.4)

    return None

if os.path.exists(CACHE_FILE):
    price_cache = pd.read_pickle(CACHE_FILE)
else:
    price_cache = {}

if os.path.exists(FAILED_TICKERS_FILE):
    failed_tickers = pd.read_pickle(FAILED_TICKERS_FILE)
else:
    failed_tickers = {}


sampled_df = pd.read_pickle("data/processed_data_sampled")
sampled_df = sampled_df.dropna(subset=["ticker", "date_parsed"]).copy()
sampled_df["ticker_clean"] = sampled_df["ticker"].apply(clean_ticker)

tickers = sorted(sampled_df["ticker_clean"].dropna().unique())

min_date = sampled_df["date_parsed"].min() - pd.Timedelta(days=10)
max_date = sampled_df["date_parsed"].max() + pd.Timedelta(days=10)
start = min_date.strftime("%Y-%m-%d")
end = max_date.strftime("%Y-%m-%d")


for t in tqdm(tickers, desc="Downloading tickers"):
    if t in price_cache and isinstance(price_cache[t], pd.Series):
        continue

    df = safe_download(t, start, end)

    if isinstance(df, str) and df == "YFTzMissingError":
        price_cache[t] = pd.Series(dtype=float)
        failed_tickers[t] = "YFTzMissingError (possibly delisted)"
    elif df is not None and not df.empty:
        if "Adj Close" in df.columns:
            price_cache[t] = df["Adj Close"].copy()
        elif "Close" in df.columns:
            price_cache[t] = df["Close"].copy()
        else:
            price_cache[t] = pd.Series(dtype=float)
            failed_tickers[t] = "No Close or Adj Close"
    else:
        price_cache[t] = pd.Series(dtype=float)
        failed_tickers[t] = "Empty / Delisted / Missing data"

    time.sleep(0.35)

    if len(price_cache) % 10 == 0:
        pd.to_pickle(price_cache, CACHE_FILE)
        pd.to_pickle(failed_tickers, FAILED_TICKERS_FILE)

# final save
pd.to_pickle(price_cache, CACHE_FILE)
pd.to_pickle(failed_tickers, FAILED_TICKERS_FILE)


def nearest_price(series, date, direction="backward"):
    date = pd.Timestamp(date).normalize()
    if date in series.index:
        return series.loc[date]

    if direction == "backward":
        subset = series.index[series.index <= date]
        return series.loc[subset.max()] if len(subset) else pd.NA
    else:
        subset = series.index[series.index >= date]
        return series.loc[subset.min()] if len(subset) else pd.NA


def ret(p_after, p0):
    # Convert Series to scalar if needed
    if isinstance(p_after, pd.Series):
        p_after = p_after.iloc[0] if not p_after.empty else pd.NA
    if isinstance(p0, pd.Series):
        p0 = p0.iloc[0] if not p0.empty else pd.NA

    if pd.isna(p_after) or pd.isna(p0) or p0 == 0:
        return pd.NA

    return (p_after - p0) / p0



results = []

for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Building price features"):
    t = row["ticker_clean"]
    call_date = row["date_parsed"]

    series = price_cache.get(t, pd.Series(dtype=float))

    if series.empty:
        results.append({
            "index": idx,
            "adj_close": pd.NA,
            "adj_close_1d_before": pd.NA,
            "adj_close_3d_before": pd.NA,
            "adj_close_5d_before": pd.NA,
            "adj_close_1d_after": pd.NA,
            "adj_close_3d_after": pd.NA,
            "adj_close_5d_after": pd.NA,
            "ret_1d_after": pd.NA,
            "ret_3d_after": pd.NA,
            "ret_5d_after": pd.NA,
            "ret_1d_before": pd.NA,
            "ret_3d_before": pd.NA,
            "ret_5d_before": pd.NA,
        })
        continue

    # Precompute dates
    dates = {
        "0": call_date,
        "1d_before": call_date - pd.Timedelta(days=1),
        "3d_before": call_date - pd.Timedelta(days=3),
        "5d_before": call_date - pd.Timedelta(days=5),
        "1d_after": call_date + pd.Timedelta(days=1),
        "3d_after": call_date + pd.Timedelta(days=3),
        "5d_after": call_date + pd.Timedelta(days=5),
    }

    prices = {}
    for key, dt in dates.items():
        direction = "forward" if "after" in key else "backward"
        prices[key] = nearest_price(series, dt, direction)

    results.append({
        "index": idx,
        "adj_close": prices["0"],
        "adj_close_1d_before": prices["1d_before"],
        "adj_close_3d_before": prices["3d_before"],
        "adj_close_5d_before": prices["5d_before"],
        "adj_close_1d_after": prices["1d_after"],
        "adj_close_3d_after": prices["3d_after"],
        "adj_close_5d_after": prices["5d_after"],

        "ret_1d_after": ret(prices["1d_after"], prices["0"]),
        "ret_3d_after": ret(prices["3d_after"], prices["0"]),
        "ret_5d_after": ret(prices["5d_after"], prices["0"]),
        "ret_1d_before": ret(prices["0"], prices["1d_before"]),
        "ret_3d_before": ret(prices["0"], prices["3d_before"]),
        "ret_5d_before": ret(prices["0"], prices["5d_before"]),
    })

prices_df = pd.DataFrame(results).set_index("index")


# ============================================================
# 9) --- Merge Back Into Your Data ---
# ============================================================
merged = sampled_df.merge(prices_df, left_index=True, right_index=True, how="left")

merged.to_pickle("data/processed_data_sampled_with_prices_ultimate.pkl")
merged.to_csv("data/processed_data_sampled_with_prices_ultimate.csv", index=False)

print("\n✅ Finished: Ultimate price dataset created.")
print("📌 Cached tickers saved to:", CACHE_FILE)
print("⚠️ Failed tickers saved to:", FAILED_TICKERS_FILE)
