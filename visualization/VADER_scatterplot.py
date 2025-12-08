import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- LOAD DATA ---
df = pd.read_pickle("data/processed_data_sampled_with_vader_fast.pkl")

# output directory
os.makedirs("visualization/VADER", exist_ok=True)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="qa_vader_score",
    y="ret_1d_after",
    hue="qa_vader_label",
    alpha=0.6
)

plt.title("VADER Sentiment Score vs 1-Day Return")
plt.xlabel("VADER Sentiment Score (Q&A)")
plt.ylabel("1-Day Return")
plt.tight_layout()

plt.savefig("visualization/VADER/VADER_scatterplot.png")

plt.show()
