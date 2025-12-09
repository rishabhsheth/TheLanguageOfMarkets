import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_pickle("data/processed_data_sampled_with_finbert_fast.pkl")

os.makedirs("visualization/FinBERT", exist_ok=True)

df["prepared_finbert_score"] = pd.to_numeric(df["prepared_finbert_score"], errors="coerce")
df["ret_1d_after"] = pd.to_numeric(df["ret_1d_after"], errors="coerce")
df = df.dropna(subset=["prepared_finbert_score", "ret_1d_after"])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

sns.scatterplot(
    data=df,
    x="prepared_finbert_score",
    y="ret_1d_after",
    hue="prepared_finbert_label",
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title("FinBERT Sentiment vs 1-Day Return")
axes[0].set_xlabel("FinBERT Sentiment Score")
axes[0].set_ylabel("1-Day Return")

sns.scatterplot(
    data=df,
    x="prepared_finbert_score",
    y="ret_1d_after",
    hue="prepared_finbert_label",
    alpha=0.6,
    ax=axes[1]
)

labels = df["prepared_finbert_label"].unique()
for label in labels:
    sns.regplot(
        data=df[df["prepared_finbert_label"] == label],
        x="prepared_finbert_score",
        y="ret_1d_after",
        scatter=False,
        label=f"{label} trend",
        ci=None,
        ax=axes[1]
    )

axes[1].set_title("With Regression Trendlines")
axes[1].set_xlabel("FinBERT Sentiment Score")
axes[1].set_ylabel("1-Day Return")

plt.tight_layout()
plt.savefig("visualization/FinBERT/FinBERT_scatterplot.png")
plt.show()
