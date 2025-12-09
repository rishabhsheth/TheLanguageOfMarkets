import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_pickle("data/processed_data_sampled_with_finbert_fast.pkl")

os.makedirs("visualization/FinBERT", exist_ok=True)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="prepared_finbert_score",
    y="ret_1d_after",
    hue="prepared_finbert_label",
    alpha=0.6
)

plt.title("FinBERT Sentiment Score vs 1-Day Return")
plt.xlabel("FinBERT Sentiment Score (Prepared)")
plt.ylabel("1-Day Return")
plt.tight_layout()

plt.savefig("visualization/FinBERT/FinBERT_scatterplot.png")

plt.show()
