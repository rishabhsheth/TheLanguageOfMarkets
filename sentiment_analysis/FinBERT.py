from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from math import ceil
from tqdm import tqdm  # for progress bar

# Load FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
model.eval()  # ensure evaluation mode

# Load your processed data
df = pd.read_pickle("data/processed_data")

# Columns to score
TEXT_COLUMNS = ["transcript", "preprared", "qa"]

# ---- CONFIG ----
BATCH_SIZE = 16  # adjust based on your GPU memory
# ----------------

# Labels for FinBERT
labels = ["negative", "neutral", "positive"]

def batch_finbert(text_list):
    """Process a batch of texts and return labels and scores."""
    # Tokenize batch
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu()  # move back to CPU
    max_idxs = torch.argmax(probs, dim=1)

    batch_labels = [labels[i] for i in max_idxs.tolist()]
    batch_scores = [float(probs[i, idx]) for i, idx in enumerate(max_idxs.tolist())]

    return batch_labels, batch_scores

# Process each column in batches
for col in TEXT_COLUMNS:
    texts = df[col].fillna("").tolist()  # replace NaN with empty string

    all_labels = []
    all_scores = []

    num_batches = ceil(len(texts) / BATCH_SIZE)

    for i in tqdm(range(num_batches), desc=f"Processing {col}"):
        batch_texts = texts[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        batch_labels, batch_scores = batch_finbert(batch_texts)
        all_labels.extend(batch_labels)
        all_scores.extend(batch_scores)

    df[f"{col}_finbert_label"] = all_labels
    df[f"{col}_finbert_score"] = all_scores

# Save result
df.to_pickle("data/processed_with_finbert.pkl")
print("Done! Added FinBERT sentiment columns (GPU-batched).")
