import pandas as pd
import statsmodels.api as sm
import os

# --- Paths ---
data_path = "data/processed_data_sampled_with_finbert_fast.pkl"
output_folder = "analysis"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "regression_by_label.txt")

# --- Load data ---
df = pd.read_pickle(data_path)
df["prepared_finbert_score"] = pd.to_numeric(df["prepared_finbert_score"], errors="coerce")
df["ret_1d_after"] = pd.to_numeric(df["ret_1d_after"], errors="coerce")
df = df.dropna(subset=["prepared_finbert_score", "ret_1d_after"])

# --- Regression by label ---
labels = df["prepared_finbert_label"].unique()
with open(output_file, "w") as f:
    f.write("=== Regression by Sentiment Label ===\n\n")
    for label in labels:
        subset = df[df["prepared_finbert_label"] == label]
        X = sm.add_constant(subset["prepared_finbert_score"])  # intercept
        y = subset["ret_1d_after"]
        
        model = sm.OLS(y, X).fit()
        
        alpha = model.params['const']
        beta = model.params['prepared_finbert_score']
        r2 = model.rsquared
        t_stat = model.tvalues['prepared_finbert_score']
        p_val = model.pvalues['prepared_finbert_score']
        
        f.write(f"{label}:\n")
        f.write(f"  Intercept: {alpha:.5f}, Slope: {beta:.5f}\n")
        f.write(f"  R-squared: {r2:.4f}\n")
        f.write(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}\n\n")

print(f"Regression evaluation completed. Results saved to {output_file}")
