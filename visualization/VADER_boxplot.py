import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_pickle("data/processed_data_sampled_with_vader_fast.pkl")

os.makedirs("visualization/VADER", exist_ok=True)

sections = {
    "transcript": "transcript_vader_label",
    "prepared": "prepared_vader_label",
    "qa": "qa_vader_label"
}

valid_labels = ["pos", "neg", "neu"]

ret_cols = {
    "ret_5d_before": "5D Before",
    "ret_3d_before": "3D Before",
    "ret_1d_before": "1D Before",
    "ret_1d_after": "1D After",
    "ret_3d_after": "3D After",
    "ret_5d_after": "5D After"
}

for section_name, label_col in sections.items():

    print(f"Generating boxplot for: {section_name}")

    df_filtered = df[df[label_col].isin(valid_labels)]

    df_melt = df_filtered.melt(
        id_vars=[label_col],
        value_vars=list(ret_cols.keys()),
        var_name="horizon",
        value_name="return"
    )

    df_melt["horizon"] = df_melt["horizon"].map(ret_cols)

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

    save_path = f"visualization/VADER/VADER_boxplot_{section_name}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")
