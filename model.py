import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import faiss
from collections import Counter
import joblib

#SETUP
nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

_initialized = False

#  CLEAN 
def clean(text):
    words = text.lower().split()
    return " ".join([w for w in words if w not in stop_words])

# INITIALIZE 
def initialize():
    global df, model, model_emb, index, _initialized

    if _initialized:
        return

    print("Loading & training model...")

#SCRAPING 
    all_titles = []
    for page in range(1, 10): 
        url = f"https://news.ycombinator.com/news?p={page}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        titles = soup.find_all("span", class_="titleline")

        for t in titles:
            all_titles.append(t.text)

    df = pd.DataFrame(all_titles, columns=["title"])
    df["clean_title"] = df["title"].apply(clean)

    # WORD FREQUENCY
    all_words = " ".join(df["clean_title"]).split()
    word_freq = Counter(all_words)

    def trend_score(text):
        return sum(word_freq[w] for w in text.split())

    df["trend_score"] = df["clean_title"].apply(trend_score)

    # EMBEDDINGS
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_emb.encode(df["clean_title"].tolist())

    #SEMANTIC SCORE
    centroid = np.mean(embeddings, axis=0)
    semantic_score = np.dot(embeddings, centroid)

    df["semantic_score"] = semantic_score

    #FINAL SCORE
    df["final_score"] = (
        0.6 * df["trend_score"] +
        0.4 * df["semantic_score"]
    )

    threshold = df["final_score"].quantile(0.7)
    df["label"] = (df["final_score"] >= threshold).astype(int)

    #FEATURES
    df["sentiment"] = df["clean_title"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["length"] = df["clean_title"].apply(len)
    df["word_count"] = df["clean_title"].apply(lambda x: len(x.split()))

    extra = df[["sentiment", "length", "word_count", "semantic_score"]].values
    X = np.hstack((embeddings, extra))
    y = df["label"].values

    #TRAIN
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    #EVALUATION
    print("\nModel Evaluation:")
    print(classification_report(y_test, model.predict(X_test)))

    # SAVE MODEL
    joblib.dump(model, "trend_model.pkl")

    # FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Model ready!")
    _initialized = True


# SEARCH 
def search_similar(query, k=5):
    query_clean = clean(query)
    emb = model_emb.encode([query_clean])

    D, I = index.search(emb, k * 3)

    results = []
    for i in I[0]:
        title = df.iloc[i]["title"]

 # better filtering
        if any(word in title.lower() for word in query_clean.split()):
            results.append(title)

    return results[:k] if results else [df.iloc[i]["title"] for i in I[0][:k]]


# EXPLANATION 
def explain_trend(query, similar):
    return f"""
    The topic '{query}' is gaining attention due to related discussions such as:
    {', '.join(similar[:3])}.
    
    The model identifies this trend using frequency patterns, semantic similarity,
    and contextual relevance in recent data.
    """


# MAIN
def ai_analyst(query, k=5):
    initialize()

    query_clean = clean(query)
    emb = model_emb.encode([query_clean])

    sent = sia.polarity_scores(query_clean)["compound"]
    length = len(query_clean)
    wc = len(query_clean.split())

    # QUERY RELEVANCE
    similarity_scores = np.dot(
        model_emb.encode(df["clean_title"].tolist()),
        emb.T
    ).flatten()

    relevance = np.max(similarity_scores)

    features = np.hstack((emb, [[sent, length, wc, relevance]]))

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    #BOOST COMMON TECH TERMS 
    trending_keywords = ["ai", "chatgpt", "llm", "openai", "machine learning"]

    if any(k in query.lower() for k in trending_keywords):
        prob += 0.2

    prob = min(max(prob, 0.1), 0.95)

    similar = search_similar(query, k)
    explanation = explain_trend(query, similar)

    return pred, prob, similar, explanation