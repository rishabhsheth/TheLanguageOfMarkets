from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from math import ceil
from tqdm import tqdm  # progress bar

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device} (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple GPU via MPS)")
else:
    device = torch.device("cpu")
    print(f"Using device: {device} (CPU)")

# --- Load FinBERT ---
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()  # evaluation mode

# --- Load your data ---
df = pd.read_pickle("data/processed_data")
TEXT_COLUMNS = ["transcript", "prepared", "qa"]

# --- Config ---
BATCH_SIZE = 16  # adjust for GPU/CPU memory

# --- Labels ---
labels = ["negative", "neutral", "positive"]

# --- Batch inference function ---
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

    # Use autocast for Apple GPU (MPS) or NVIDIA if desired
    if device.type in ["cuda", "mps"]:
        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.autocast(device):
            with torch.no_grad():
                logits = model(**inputs).logits
    else:
        with torch.no_grad():
            logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu()
    max_idxs = torch.argmax(probs, dim=1)

    batch_labels = [labels[i] for i in max_idxs.tolist()]
    batch_scores = [float(probs[i, idx]) for i, idx in enumerate(max_idxs.tolist())]

    return batch_labels, batch_scores

# --- Process each column in batches ---
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

# --- Save the results ---
df.to_pickle("data/processed_with_finbert.pkl")
print("Done! Added FinBERT sentiment columns (GPU/CPU auto-selected).")
