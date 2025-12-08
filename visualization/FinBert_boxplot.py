import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- LOAD DATA ---
df = pd.read_pickle("data/processed_data_sampled_with_finbert_fast.pkl")

# output directory
os.makedirs("visualization/FinBERT", exist_ok=True)

# --- CONFIG ---
sections = {
    "transcript": "transcript_finbert_label",
    "prepared": "prepared_finbert_label",
    "qa": "qa_finbert_label"
}

valid_labels = ["positive", "negative", "neutral"]

ret_cols = {
    "ret_1d_after": "1D After",
    "ret_3d_after": "3D After",
    "ret_5d_after": "5D After"
}

# --- LOOP OVER SECTIONS ---
for section_name, label_col in sections.items():

    print(f"Generating boxplot for: {section_name}")

    # Filter valid rows
    df_filtered = df[df[label_col].isin(valid_labels)]

    # Melt returns
    df_melt = df_filtered.melt(
        id_vars=[label_col],
        value_vars=list(ret_cols.keys()),
        var_name="horizon",
        value_name="return"
    )

    df_melt["horizon"] = df_melt["horizon"].map(ret_cols)

    # --- PLOT ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_melt,
        x="horizon",
        y="return",
        hue=label_col
    )

    plt.title(f"Returns After Earnings Call — Grouped by {section_name.capitalize()} Sentiment")
    plt.xlabel("Return Horizon")
    plt.ylabel("Return")
    plt.legend(title=f"{section_name.capitalize()} Sentiment")
    plt.tight_layout()

    # save
    save_path = f"visualization/FinBERT/FinBERT_boxplot_{section_name}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")
