import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from math import ceil
import pickle
import os
import numpy as np

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()
LABELS = ["negative", "neutral", "positive"]

# The maximum sequence length for FinBERT (and most BERT models)
MAX_LENGTH = 512
# We set a slight overlap to prevent chopping sentences in half at the boundaries
OVERLAP = 128
# Adjust for your GPU VRAM
BATCH_SIZE = 256 

# --- LOAD DATA ---
# Ensure this path is correct
try:
    df = pd.read_pickle("data/processed_data_sampled_with_prices_ultimate.pkl")
    df.dropna(subset=['adj_close', 'adj_close_1d_before',
       'adj_close_3d_before', 'adj_close_5d_before', 'adj_close_1d_after',
       'adj_close_3d_after', 'adj_close_5d_after', 'ret_1d_after',
       'ret_3d_after', 'ret_5d_after', 'ret_1d_before', 'ret_3d_before',
       'ret_5d_before'], inplace=True)
except FileNotFoundError:
    print("Error: Could not find 'data/processed_data_sampled_with_prices_ultimate.pkl'. Please ensure the file exists.")
    exit()

TEXT_COLUMNS = ["transcript", "prepared", "qa"]

# --- NEW: TOKENIZE FUNCTION WITH SLIDING WINDOW ---
def tokenize_with_sliding_window(texts, max_len=MAX_LENGTH, overlap=OVERLAP):
    """
    Tokenizes a list of long texts using a sliding window approach.
    Returns a list of dictionaries, where each dict represents a document
    and contains all its chunks' input_ids and attention_mask.
    """
    tokenized_documents = []

    for text in tqdm(texts, desc="Tokenizing and Chunking"):
        # Handle NaN/None text
        if pd.isna(text) or text == "":
            tokenized_documents.append({"input_ids": None, "attention_mask": None, "num_chunks": 0})
            continue

        # Basic tokenization without truncation to get token IDs
        # return_offsets_mapping is used to help find split points, but we'll use a simpler
        # direct token slicing method here for efficiency.
        full_tok = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        full_ids = full_tok["input_ids"][0]  # Get the tensor of token IDs

        n_tokens = full_ids.size(0)
        chunks_ids = []
        chunks_attention = []

        # Start and end indices for the window
        start = 0
        end = max_len

        while start < n_tokens:
            # Determine the end of the current chunk
            end = min(start + max_len, n_tokens)
            
            # Slice the token IDs for the current chunk
            chunk_ids = full_ids[start:end]
            
            # Create attention mask (all ones since it's a full chunk)
            chunk_attention_mask = torch.ones_like(chunk_ids)

            chunks_ids.append(chunk_ids)
            chunks_attention.append(chunk_attention_mask)

            # Move the window forward by (max_len - overlap)
            start += (max_len - overlap)
            
            # Break if the last slice was the full text
            if end == n_tokens:
                break
        
        # We need to pad all chunks to the same length (MAX_LENGTH) for batch processing later
        # We use a custom function here because the standard tokenizer call operates on lists of texts.
        padded_ids = []
        padded_attention = []
        for c_ids, c_att in zip(chunks_ids, chunks_attention):
            padding_needed = max_len - c_ids.size(0)
            
            # Pad with 0 (padding token ID) and 0 (for attention mask)
            padded_ids.append(torch.cat([c_ids, torch.full((padding_needed,), tokenizer.pad_token_id)], dim=0))
            padded_attention.append(torch.cat([c_att, torch.zeros(padding_needed)], dim=0))


        # Convert list of tensors to a single tensor for the document
        doc_input_ids = torch.stack(padded_ids) if padded_ids else None
        doc_attention_mask = torch.stack(padded_attention) if padded_attention else None
        
        tokenized_documents.append({
            "input_ids": doc_input_ids,
            "attention_mask": doc_attention_mask,
            "num_chunks": len(chunks_ids)
        })

    return tokenized_documents

# --- AUTO-LOAD OR CREATE TOKENIZED DATA ---
tokenized_path = "data/tokenized_finbert_chunks.pkl"

if os.path.exists(tokenized_path):
    print(f"Loading pre-tokenized chunked data from {tokenized_path} ...")
    with open(tokenized_path, "rb") as f:
        chunked_tokenized_data = pickle.load(f)
else:
    print("Chunked tokenized file not found — performing CPU tokenization...")
    chunked_tokenized_data = {}
    for col in TEXT_COLUMNS:
        texts = df[col].fillna("").tolist()
        print(f"\nTokenizing and chunking column: {col}")
        # The new function returns a list of document objects
        chunked_tokenized_data[col] = tokenize_with_sliding_window(texts, max_len=MAX_LENGTH, overlap=OVERLAP)
        print(f"{col}: tokenized and chunked {len(texts)} items")

    print(f"\nSaving chunked tokenized tensors to {tokenized_path} ...")
    with open(tokenized_path, "wb") as f:
        pickle.dump(chunked_tokenized_data, f)
    print("Chunked tokenized data saved.")

# --- NEW: GPU INFERENCE AND AGGREGATION FUNCTION ---
def run_batches_and_aggregate(tokenized_docs_list):
    """
    Runs GPU inference on all chunks across all documents and aggregates the results.
    """
    all_doc_labels = []
    all_doc_scores = []
    
    # 1. Collect ALL chunks from ALL documents into one massive list for efficient batching
    all_chunks_ids = []
    all_chunks_mask = []
    
    # Map from chunk index back to the original document index
    chunk_to_doc_idx = [] 
    
    for i, doc in enumerate(tokenized_docs_list):
        if doc["num_chunks"] > 0:
            all_chunks_ids.append(doc["input_ids"])
            all_chunks_mask.append(doc["attention_mask"])
            chunk_to_doc_idx.extend([i] * doc["num_chunks"])

    if not all_chunks_ids:
        print("No valid chunks to process.")
        return [None] * len(tokenized_docs_list), [None] * len(tokenized_docs_list)

    # Convert the list of [chunks_per_doc, 512] tensors into one massive [total_chunks, 512] tensor
    full_input_ids = torch.cat(all_chunks_ids, dim=0)
    full_attention_mask = torch.cat(all_chunks_mask, dim=0)

    # 2. Run batched GPU inference on the massive tensor
    n_chunks = full_input_ids.size(0)
    num_batches = ceil(n_chunks / BATCH_SIZE)

    all_chunk_probs = []

    for b in tqdm(range(num_batches), desc="GPU inference on chunks"):
        ids = full_input_ids[b*BATCH_SIZE:(b+1)*BATCH_SIZE].to(device)
        mask = full_attention_mask[b*BATCH_SIZE:(b+1)*BATCH_SIZE].to(device)

        with torch.no_grad():
            logits = model(ids, attention_mask=mask).logits

        # Convert logits to probabilities (Negative, Neutral, Positive)
        probs = torch.softmax(logits, dim=1).cpu() 
        all_chunk_probs.append(probs)

    # Concatenate all probability tensors
    full_probs_tensor = torch.cat(all_chunk_probs, dim=0)

    # 3. Aggregate results per document
    
    # Initialize lists to store the final aggregated results
    final_doc_labels = [None] * len(tokenized_docs_list)
    final_doc_scores = [None] * len(tokenized_docs_list)
    
    # Loop through the original list of documents
    current_chunk_idx = 0
    for doc_idx, doc in enumerate(tokenized_docs_list):
        n_chunks = doc["num_chunks"]
        
        if n_chunks == 0:
            # Keep as None if there was no text
            continue

        # Get the probabilities for all chunks belonging to this document
        doc_probs = full_probs_tensor[current_chunk_idx : current_chunk_idx + n_chunks]
        
        # AGGREGATION METHOD: Average the probabilities across all chunks.
        # This is generally a robust method for long document sentiment.
        mean_probs = doc_probs.mean(dim=0)
        
        # The final label is the one with the highest average probability
        max_idx = mean_probs.argmax().item()
        
        final_doc_labels[doc_idx] = LABELS[max_idx]
        final_doc_scores[doc_idx] = mean_probs[max_idx].item()

        current_chunk_idx += n_chunks

    return final_doc_labels, final_doc_scores

# --- PROCESS EACH COLUMN ---
for col in TEXT_COLUMNS:
    print(f"\n=== Processing {col} (Chunk Aggregation) ===")
    
    # tokenized_docs_list is the output from tokenize_with_sliding_window
    labels, scores = run_batches_and_aggregate(chunked_tokenized_data[col])
    
    # Store the results in the DataFrame
    df[f"{col}_finbert_label"] = labels
    df[f"{col}_finbert_score"] = scores

# --- SAVE RESULTS ---
df.to_pickle("data/processed_data_sampled_with_finbert_fast.pkl")
print("\n DONE — full FinBERT sentiment analysis with chunking complete.")
print(" GPU optimized with batching on all chunks")
print(" Chunked data auto-loaded next time")