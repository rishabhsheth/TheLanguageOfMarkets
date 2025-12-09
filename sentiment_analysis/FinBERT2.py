import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from math import ceil
import pickle
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()
LABELS = ["negative", "neutral", "positive"]

MAX_LENGTH = 512
OVERLAP = 128
BATCH_SIZE = 256 

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

def tokenize_with_sliding_window(texts, max_len=MAX_LENGTH, overlap=OVERLAP):
    """
    Tokenizes a list of long texts using a sliding window approach.
    Returns a list of dictionaries, where each dict represents a document
    and contains all its chunks' input_ids and attention_mask.
    """
    tokenized_documents = []

    for text in tqdm(texts, desc="Tokenizing and Chunking"):
        if pd.isna(text) or text == "":
            tokenized_documents.append({"input_ids": None, "attention_mask": None, "num_chunks": 0})
            continue

        full_tok = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        full_ids = full_tok["input_ids"][0]  # Get the tensor of token IDs

        n_tokens = full_ids.size(0)
        chunks_ids = []
        chunks_attention = []

        start = 0
        end = max_len

        while start < n_tokens:
            end = min(start + max_len, n_tokens)
            
            chunk_ids = full_ids[start:end]
            
            chunk_attention_mask = torch.ones_like(chunk_ids)

            chunks_ids.append(chunk_ids)
            chunks_attention.append(chunk_attention_mask)

            start += (max_len - overlap)
            
            if end == n_tokens:
                break

        padded_ids = []
        padded_attention = []
        for c_ids, c_att in zip(chunks_ids, chunks_attention):
            padding_needed = max_len - c_ids.size(0)
            
            padded_ids.append(torch.cat([c_ids, torch.full((padding_needed,), tokenizer.pad_token_id)], dim=0))
            padded_attention.append(torch.cat([c_att, torch.zeros(padding_needed)], dim=0))


        doc_input_ids = torch.stack(padded_ids) if padded_ids else None
        doc_attention_mask = torch.stack(padded_attention) if padded_attention else None
        
        tokenized_documents.append({
            "input_ids": doc_input_ids,
            "attention_mask": doc_attention_mask,
            "num_chunks": len(chunks_ids)
        })

    return tokenized_documents

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
        chunked_tokenized_data[col] = tokenize_with_sliding_window(texts, max_len=MAX_LENGTH, overlap=OVERLAP)
        print(f"{col}: tokenized and chunked {len(texts)} items")

    print(f"\nSaving chunked tokenized tensors to {tokenized_path} ...")
    with open(tokenized_path, "wb") as f:
        pickle.dump(chunked_tokenized_data, f)
    print("Chunked tokenized data saved.")

def run_batches_and_aggregate(tokenized_docs_list):
    """
    Runs GPU inference on all chunks across all documents and aggregates the results.
    """
    all_doc_labels = []
    all_doc_scores = []
    
    all_chunks_ids = []
    all_chunks_mask = []
    
    chunk_to_doc_idx = [] 
    
    for i, doc in enumerate(tokenized_docs_list):
        if doc["num_chunks"] > 0:
            all_chunks_ids.append(doc["input_ids"])
            all_chunks_mask.append(doc["attention_mask"])
            chunk_to_doc_idx.extend([i] * doc["num_chunks"])

    if not all_chunks_ids:
        print("No valid chunks to process.")
        return [None] * len(tokenized_docs_list), [None] * len(tokenized_docs_list)

    full_input_ids = torch.cat(all_chunks_ids, dim=0)
    full_attention_mask = torch.cat(all_chunks_mask, dim=0)

    n_chunks = full_input_ids.size(0)
    num_batches = ceil(n_chunks / BATCH_SIZE)

    all_chunk_probs = []

    for b in tqdm(range(num_batches), desc="GPU inference on chunks"):
        ids = full_input_ids[b*BATCH_SIZE:(b+1)*BATCH_SIZE].to(device)
        mask = full_attention_mask[b*BATCH_SIZE:(b+1)*BATCH_SIZE].to(device)

        with torch.no_grad():
            logits = model(ids, attention_mask=mask).logits

        probs = torch.softmax(logits, dim=1).cpu() 
        all_chunk_probs.append(probs)

    full_probs_tensor = torch.cat(all_chunk_probs, dim=0)

    
    final_doc_labels = [None] * len(tokenized_docs_list)
    final_doc_scores = [None] * len(tokenized_docs_list)
    
    current_chunk_idx = 0
    for doc_idx, doc in enumerate(tokenized_docs_list):
        n_chunks = doc["num_chunks"]
        
        if n_chunks == 0:
            continue

        doc_probs = full_probs_tensor[current_chunk_idx : current_chunk_idx + n_chunks]
        
        mean_probs = doc_probs.mean(dim=0)
        
        max_idx = mean_probs.argmax().item()
        
        final_doc_labels[doc_idx] = LABELS[max_idx]
        final_doc_scores[doc_idx] = mean_probs[max_idx].item()

        current_chunk_idx += n_chunks

    return final_doc_labels, final_doc_scores

for col in TEXT_COLUMNS:
    print(f"\n=== Processing {col} (Chunk Aggregation) ===")
    
    labels, scores = run_batches_and_aggregate(chunked_tokenized_data[col])
    
    df[f"{col}_finbert_label"] = labels
    df[f"{col}_finbert_score"] = scores

df.to_pickle("data/processed_data_sampled_with_finbert_fast.pkl")
print("\n DONE — full FinBERT sentiment analysis with chunking complete.")
print(" GPU optimized with batching on all chunks")
print(" Chunked data auto-loaded next time")