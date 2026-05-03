# AI Trend Intelligence System

An AI-driven system that analyzes emerging topics using NLP, semantic embeddings, and machine learning.

---

## Overview
This project identifies and analyzes trending topics from web-scraped data by combining natural language processing, semantic similarity, and machine learning techniques. It provides an interactive dashboard to explore trends, related topics, and prediction confidence.

---

## Features
-  Trend analysis using XGBoost classifier  
-  Semantic similarity search using FAISS  
-  Feature engineering (sentiment, frequency, semantic relevance)  
-  Interactive Streamlit dashboard  
-  Fast retrieval using vector embeddings  

---

## How It Works
1. Scrapes trending data from web sources (Hacker News)  
2. Cleans and preprocesses text using NLP techniques  
3. Generates embeddings using Sentence Transformers  
4. Computes trend scores using frequency + semantic relevance  
5. Trains an XGBoost model for classification  
6. Uses FAISS for similarity-based trend retrieval  
7. Displays results via Streamlit UI  

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **NLP:** NLTK, Sentence Transformers  
- **ML:** XGBoost  
- **Vector Search:** FAISS  
- **Frontend:** Streamlit  

---

## Model Evaluation
The model is evaluated using classification metrics such as:
- Precision  
- Recall  
- F1-score  

(Displayed in console / can be integrated into UI)

---
## Live Demo
https://ai-trend-intelligence-6hyaqj7supdnvpdktfddf8.streamlit.app/


