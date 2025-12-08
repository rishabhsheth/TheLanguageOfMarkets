# 1. KDE 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_pickle("data/processed_data_sampled_with_vader_fast.pkl")

# output directory
os.makedirs("visualization/VADER", exist_ok=True)

plt.figure(figsize=(10,6))
sns.kdeplot(df['transcript_vader_score'].dropna(), label='Transcript', fill=True, alpha=0.3)
sns.kdeplot(df['prepared_vader_score'].dropna(), label='Prepared', fill=True, alpha=0.3)
sns.kdeplot(df['qa_vader_score'].dropna(), label='Q&A', fill=True, alpha=0.3)
plt.title('VADER Sentiment Score Distribution by Section')
plt.xlabel('VADER Compound Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('visualization/VADER/VADER_kernel_density_estimator.png')
plt.show()

#2. Box Plot

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_pickle("data/processed_data_sampled_with_vader_fast.pkl")

sections = ['transcript_vader_score','prepared_vader_score','qa_vader_score']
labels = ['Transcript','Prepared','Q&A']
return_cols = ['ret_1d_after', 'ret_3d_after', 'ret_5d_after']
titles = ['1-Day After', '3-Day After', '5-Day After']

plt.figure(figsize=(18,6))

for i, ret in enumerate(return_cols):
    # melt for each subplot
    tmp = df[sections]
    tmp = tmp.melt(var_name='section', value_name='score')
    
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=tmp, x='section', y='score')
    plt.xticks(range(3), labels, rotation=45)
    plt.title(f"VADER Sentiment — {titles[i]} Return Window")
    plt.xlabel("")
    plt.ylabel("VADER Score")

plt.tight_layout()

plt.savefig("visualization/VADER/VADER_boxplot_per_returns.png")

plt.show()

