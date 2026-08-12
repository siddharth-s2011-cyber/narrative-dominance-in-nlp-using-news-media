
import re
from urllib.parse import quote
import feedparser
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
NUM_CLUSTER=3
from bs4 import BeautifulSoup

def clean(text):
    if not text:
        return ""
    # Remove HTML first
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def fetch(topic, start_date, end_date):
    encoded_topic = quote(topic)
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries:
        published = entry.get("published", "")
        try:
            pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
        except:
            continue
        if start_date <= pub_date.strftime("%Y-%m-%d") <= end_date:
            articles.append({
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "published": pub_date.strftime("%Y%m%d%H%M%S")
            })
    print(f"Fetched {len(articles)} articles.")
    return articles


def mutation_ratio(t1,t2):
    w1=set(t1.split())
    w2=set(t2.split())
    if not w1 or not w2:
        return 0
    return 1-len(w1.intersection(w2))/max(len(w1),len(w2))

def parse_time(t):
    return datetime.strptime(t,"%Y%m%d%H%M%S")

def main():
    topic=input("Enter topic name:")
    start_date=input("Enter start date in YYYY-MM-DD format:")
    end_date=input("Enter end date in YYYY-MM-DD format:")
    print("\n fetching articles...")
    article=fetch(topic,start_date,end_date)


    if len(article)<10:
        print("Not enough articles found on the subject to run narrative competition")
        return
    texts = [clean(a["title"] + " " + a["description"]) for a in article]
    times=[parse_time(a["published"]) for a in article]


    print("Generated embeddings...")
    embedder=SentenceTransformer("all-MiniLM-L6-v2")
    embedding=embedder.encode(texts)


    print("clustering narratives...")
    kmeans=KMeans(
        n_clusters=NUM_CLUSTER,
        random_state=42
    )
    clusters=kmeans.fit_predict(embedding)


    df=pd.DataFrame({
        "text":texts,
        "time":times,
        "cluster":clusters
    })


    vader=SentimentIntensityAnalyzer()
    df["sentiment"]=df["text"].apply(lambda x: vader.polarity_scores(x)["compound"])

    tfidf=TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )
    tfidf_matrix=tfidf.fit_transform(df["text"])
    feature_names=np.array(tfidf.get_feature_names_out())

    narrative_results=[]

    print("\n Narrative Competition Analysis")

    for c in range(NUM_CLUSTER):
        sub=df[df["cluster"]==c].sort_values("time")
        volume=len(sub)
        if volume<3:
            continue

        duration_days = (sub["time"].iloc[-1] - sub["time"].iloc[0]).days + 1
        growth = volume / duration_days

        base=sub["text"].iloc[0]
        mutation=[mutation_ratio(base,t) for t in sub["text"].iloc[1:]]
        stability=1-np.var(mutation)

        avg_sentiment=sub["sentiment"].mean()
        sentiment_volatility=sub["sentiment"].std()

        idx=sub.index
        cluster_tfidf=tfidf_matrix[idx].mean(axis=0)
        top_words=feature_names[np.argsort(cluster_tfidf.A1)[-5:]]

        nds = volume * growth * stability

        narrative_results.append({
            "Narrative": c,
            "Volume": volume,
            "GrowthRate": round(growth, 3),
            "Stability": round(stability, 3),
            "AvgSentiment": round(avg_sentiment, 3),
            "SentimentVolatility": round(sentiment_volatility, 3),
            "TopKeywords": ", ".join(top_words),
            "NDS": round(nds, 2)
        })

    result_df = pd.DataFrame(narrative_results).sort_values("NDS", ascending=False)
    result_df.index+=1
    result_df["Narrative"]+=1

    print(result_df.to_string())

    dominant = result_df.iloc[0]

    print("\n DOMINANT NARRATIVE ")
    print(dominant)

if __name__ == "__main__":
    main()
