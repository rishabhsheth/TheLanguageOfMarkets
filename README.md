# The Language of Markets: Linking Executive Tone to Investor Behavior

## Overview
This project investigates whether the sentiment expressed by company executives during quarterly earnings calls can predict short-term stock performance. By converting unstructured transcript text into quantitative sentiment metrics, we aim to determine whether market reactions in the 1-, 3-, and 5-day window following an earnings call can be systematically explained—or even anticipated—using natural language processing (NLP) techniques.

This work sits at the intersection of **financial analysis**, **text analytics**, and **machine learning**, combining statistical modeling and domain-adapted sentiment models to evaluate how tone and language influence investor behavior.

---

## Team
- **Kaustubh Baskaran (kb1153)**
- **Rishabh Sheth (rs2299)**

---

## Motivation
Investors traditionally rely on financial statements to assess company performance, but quantitative metrics alone often overlook the nuances of human communication. Earnings calls provide rich qualitative insight, including tone, confidence, and sentiment directly from company leadership—signals that can shape investor expectations and drive short-term price action.

Our central research question:

> **Can earnings call sentiment predict short-term stock performance?**

If so, this would demonstrate that NLP-based sentiment analysis can enhance data-driven investment strategies and potentially generate excess market returns.

---

## Background and Related Work
Traditional sentiment dictionaries often mislabel common financial terms (e.g., *depreciation*, *liabilities*, *amortization*) as negative, leading to unreliable sentiment scoring in financial domains. This project builds on the foundational research of:

> **Loughran & McDonald (2011), “When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.”**

Their work led to the development of domain-specific sentiment lexicons and modern financial NLP models such as **FinBERT**, which we use to more accurately classify sentiment in earnings call transcripts.

---

## Dataset

### Earnings Call Transcripts
We use a dataset from Kaggle containing:

- **18,755 earnings call transcripts**
- Scraped from *The Motley Fool*
- Distributed in `.pkl` format
- Includes ticker information and earnings call dates

### Stock Price Data
Daily adjusted closing prices are retrieved through **Yahoo Finance** via the `yfinance` API. Each transcript is matched by ticker and call date, allowing us to compute returns over:

- **1-day post-call window**
- **3-day post-call window**
- **5-day post-call window**

---

## Methods

### 1. Preprocessing
- Load transcripts using Pandas  
- Clean data (remove non-US stock exchanges and rows with missing dates, as they can not be matched to yfinance data)
- Split data into Prepared and Q&A sections (to compare performance across entire transcript, exclusively Prepared, and exclusively Q&A sections)
- Sample the resulting data to reduce the number of rows for computational purposes

### 2. Sentiment Modeling
We compute sentiment using:

- **FinBERT** – BERT adapted for financial text  
- **VADER** – Used as a baseline for comparison

Each transcript receives sentiment scores reflecting tone.

### 3. Stock Return Computation
Using stock price data:

Return = (Price_t+n – Price_t) / Price_t



where *n* ∈ {1,3,5} trading days after the earnings call.

### 4. Regression & Statistical Analysis
We evaluate the relationship between market reaction and sentiment using:

- **Ordinary Least Squares (OLS) regression**
- Potential controls (e.g., earnings surprise, other fundamentals)

### 5. Evaluation Criteria

#### Statistical
- Significance of regression coefficients  
- p-values  
- Magnitude and sign of effect  

#### Predictive
- Whether sentiment provides explanatory power beyond traditional financial metrics  
- Sensibility of effect directions  

#### Financial relevance
- Whether results meaningfully improve investment decision-making on recent calls

---

## Implementation Plan
1. Load and inspect transcript dataset  
2. Clean and preprocess text    
3. Pull price data from Yahoo Finance  
4. Compute sentiment with FinBERT and VADER
5. Merge return data with sentiment features  
6. Run regression and correlation analyses  
7. Interpret results and evaluate predictive value

---

## Tools & Dependencies
- **Python 3.x**  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `nltk`  
- `yfinance`  
- `transformers` (for FinBERT)  
- `statsmodels` (for OLS regression)

---

## Results (To Be Added)
This section will present:

- Visualizations of sentiment vs. return  
- Comparison between FinBERT and VADER performance  
- Insights on financial interpretability  

---

## Citation
Loughran, T., & McDonald, B. (2011).  
*When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.*  
The Journal of Finance, 66(1), 35–65.  

---

## License
This project is for academic and research purposes.
