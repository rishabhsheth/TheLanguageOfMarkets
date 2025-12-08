import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD DATA ---
df = pd.read_pickle("data/processed_data_sampled_with_vader_fast.pkl")

# keep only valid labels
valid_labels = ["pos", "neg", "neu"]
df_filtered = df[df["prepared_vader_label"].isin(valid_labels)]

# melt the return columns
ret_cols = {
    "ret_1d_after": "1D After",
    "ret_3d_after": "3D After",
    "ret_5d_after": "5D After"
}

df_melt = df_filtered.melt(
    id_vars=["prepared_vader_label"],
    value_vars=list(ret_cols.keys()),
    var_name="horizon",
    value_name="return"
)

# map cleaner names
df_melt["horizon"] = df_melt["horizon"].map(ret_cols)

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_melt,
    x="horizon",
    y="return",
    hue="prepared_vader_label"
)

plt.title("Returns After Earnings Call — Grouped by VADER Label")
plt.xlabel("Return Horizon")
plt.ylabel("Return")
plt.legend(title="Prepared Sentiment")
plt.tight_layout()

plt.savefig("visualization/VADER_boxplot.png")

plt.show()
