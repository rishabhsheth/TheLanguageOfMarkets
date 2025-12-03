from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load your processed data
df = pd.read_pickle("data/processed_data")

# ---- CONFIG ----
TEXT_COLUMNS = ["transcript", "preprared", "qa"]  # change these to your actual columns
# ----------------

def finbert_score(text: str):
    """Returns FinBERT sentiment (label + score)."""
    if not isinstance(text, str) or not text.strip():
        return {"label": None, "score": None}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).flatten()

    # FinBERT labels from ProsusAI: [0=negative, 1=neutral, 2=positive]
    labels = ["negative", "neutral", "positive"]
    idx = torch.argmax(probs).item()

    return {"label": labels[idx], "score": probs[idx].item()}

# Apply FinBERT to each selected column
for col in TEXT_COLUMNS:
    df[f"{col}_finbert_label"] = df[col].apply(lambda x: finbert_score(x)["label"])
    df[f"{col}_finbert_score"] = df[col].apply(lambda x: finbert_score(x)["score"])

# Save if needed
df.to_pickle("data/processed_with_finbert.pkl")
print("Done! Added FinBERT sentiment columns.")
