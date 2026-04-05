from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import requests
import re
import os
import io
import zipfile
import joblib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from collections import defaultdict
from typing import Optional
MODEL_PATH     = "./models/all-MiniLM-L6-v2"
SENTIMENT_PATH = "./models/sentiment_gdelt.joblib"
ENCODER_PATH   = "./models/label_encoder_gdelt.joblib"
GDELT_GKG_URL  = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"
GDELT_DATE     = "20200101"

GNEWS_RSS_URL  = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
GNEWS_LOOKBACK_DAYS = 30
GNEWS_MAX_ARTICLES  = 100
model: Optional[SentenceTransformer] = None
sentiment_clf: Optional[SGDClassifier] = None
label_enc: Optional[LabelEncoder] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, sentiment_clf, label_enc

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Sentence-transformer not found at '{MODEL_PATH}'.\n"
            "Run  python save_model.py  first."
        )
    print("Loading sentence transformer...")
    model = SentenceTransformer(MODEL_PATH)
    model.encode(["warmup"], show_progress_bar=False)
    print("Transformer ready.")
    if os.path.exists(SENTIMENT_PATH) and os.path.exists(ENCODER_PATH):
        print("Loading cached GDELT sentiment model...")
        sentiment_clf = joblib.load(SENTIMENT_PATH)
        label_enc     = joblib.load(ENCODER_PATH)
        print("Sentiment model loaded.")
    else:
        print("Training sentiment model on GDELT data...")
        sentiment_clf, label_enc = _train_gdelt_sentiment()
        os.makedirs("./models", exist_ok=True)
        joblib.dump(sentiment_clf, SENTIMENT_PATH)
        joblib.dump(label_enc,     ENCODER_PATH)
        print("GDELT sentiment model trained and saved.")
    yield
    print("Shutting down.")
# gdelt pipeline
def _download_gdelt_gkg(date: str = GDELT_DATE) -> pd.DataFrame:
    url = GDELT_GKG_URL.format(date=date)
    print(f"  [GDELT] Downloading {url} ...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(
            f, sep="\t", header=None,
            on_bad_lines="skip", encoding="utf-8", low_memory=False,
        )
    print(f"  [GDELT] Loaded {len(df):,} rows.")
    return df
def _parse_gdelt_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 8:
        raise ValueError(f"Only {df.shape[1]} columns — need ≥ 8.")

    out = df[[3, 7]].copy()
    out.columns = ["themes", "tone_raw"]
    out = out.dropna(subset=["tone_raw"])

    def parse_tone(t):
        try:
            return float(str(t).split(",")[0])
        except Exception:
            return None

    out["avg_tone"] = out["tone_raw"].apply(parse_tone)
    out = out.dropna(subset=["avg_tone"])
    out["sentiment"] = out["avg_tone"].apply(
        lambda t: "positive" if t > 1.5 else ("negative" if t < -1.5 else "neutral")
    )
    out["text"] = (
        out["themes"].fillna("").astype(str)
        .apply(lambda x: re.sub(r"[^a-zA-Z ]", " ", x.replace(";", " ").lower()).strip())
    )
    out = out[out["text"].str.split().str.len() >= 3]

    counts = out["sentiment"].value_counts()
    min_n  = min(counts.min(), 800)
    out = (
        out.groupby("sentiment", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), min_n), random_state=42))
           .reset_index(drop=True)
    )
    print(f"  [GDELT] Balanced dataset: {len(out)} rows ({min_n} per class).")
    return out[["text", "sentiment"]]


def _seed_dataframe() -> pd.DataFrame:
    positives = [
        "peace agreement reached ceasefire signed nations cooperate",
        "economic recovery growth gdp employment rises significantly",
        "vaccine approved treatment breakthrough medical success",
        "renewable energy solar wind investment expansion globally",
        "diplomatic relations restored cooperation bilateral summit",
        "unemployment falls record low job creation strong growth",
        "trade deal signed exports boost agreement bilateral",
        "environmental progress carbon reduction climate target achieved",
        "education reform literacy rates improve student achievement",
        "infrastructure investment roads bridges completed on schedule",
        "stock market rally investors optimism economic outlook positive",
        "scientific discovery research breakthrough technology advances",
    ]
    negatives = [
        "war conflict military attack casualties killed wounded",
        "earthquake disaster flood hurricane death toll rises thousands",
        "terrorism bombing explosion attack killed injured victims",
        "recession unemployment poverty debt crisis economic collapse",
        "corruption scandal government officials arrested charged",
        "violent protest crackdown demonstration clash police",
        "famine drought crop failure food shortage humanitarian crisis",
        "epidemic disease outbreak spread mortality rising",
        "assassination political leader killed targeted attack violence",
        "sanctions imposed trade war tariffs economic pressure severe",
        "environmental disaster oil spill pollution wildlife destroyed",
        "shooting massacre gun violence victims killed community mourns",
    ]
    neutrals = [
        "parliament session meeting scheduled officials statement released",
        "report published findings researchers annual conference",
        "survey conducted results analysis data released statistics",
        "committee reviewed legislation proposed amendment discussed",
        "central bank meeting interest rates monetary policy discussed",
        "officials confirmed project timeline implementation next phase",
        "delegates arrived summit international conference event",
        "government ministry policy review publicly announced today",
        "statistics bureau published quarterly economic indicators",
        "court hearing scheduled trial proceeding legal case",
        "company announced quarterly results revenue figures disclosed",
        "election campaign candidate policy platform speech delivered",
    ]
    rows = (
        [(t, "positive") for t in positives] +
        [(t, "negative") for t in negatives] +
        [(t, "neutral")  for t in neutrals]
    )
    return pd.DataFrame(rows, columns=["text", "sentiment"])


def _train_gdelt_sentiment():
    global model
    try:
        raw = _download_gdelt_gkg()
        df  = _parse_gdelt_sentiment(raw)
    except Exception as e:
        print(f"  [GDELT] Failed ({e}). Falling back to seed dataset.")
        df = _seed_dataframe()

    print(f"  [Train] Encoding {len(df)} texts...")
    embeddings = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)

    enc = LabelEncoder()
    y   = enc.fit_transform(df["sentiment"])

    clf = SGDClassifier(
        loss="modified_huber",
        alpha=0.0001,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(embeddings, y)
    preds = clf.predict(embeddings)
    print("  [Train] In-sample classification report:")
    print(classification_report(y, preds, target_names=enc.classes_))
    return clf, enc


# text cleaning
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# sentiment score
def sentiment_to_score(sentiment: str, probs: np.ndarray, enc: LabelEncoder) -> float:
    classes = list(enc.classes_)
    score = 0.0
    pole_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    for i, cls in enumerate(classes):
        score += pole_map.get(cls, 0.0) * probs[i]
    return round(float(score), 4)

#nds
def compute_nds(
    articles:   list[dict],
    labels:     np.ndarray,
    embeddings: np.ndarray,
    sentiments: list[str],
    sent_probs: np.ndarray,
) -> dict[int, dict]:
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)

    polarity_map = {"positive": 1.0, "neutral": 0.5, "negative": 0.2}
    nds_scores: dict[int, dict] = {}

    for cid, indices in clusters.items():
        volume = len(indices)

        # growth-rate
        timestamps = []
        for idx in indices:
            try:
                ts = datetime.fromisoformat(
                    articles[idx].get("publishedAt", "").replace("Z", "+00:00")
                )
                timestamps.append(ts)
            except Exception:
                pass
        span_hrs    = max((max(timestamps) - min(timestamps)).total_seconds() / 3600, 0.5) \
                      if len(timestamps) >= 2 else 24.0
        growth_rate = round(volume / span_hrs, 4)

        # stability
        c_embs    = embeddings[indices]
        centroid  = c_embs.mean(axis=0)
        dists     = np.linalg.norm(c_embs - centroid, axis=1)
        stability = round(float(np.clip(1.0 - dists.mean() / 1.2, 0.05, 1.0)), 4)

        # Emotional-weight
        c_sents       = [sentiments[i] for i in indices]
        c_probs       = sent_probs[indices]
        avg_intensity = float(np.mean([c_probs[j].max() for j in range(len(indices))]))
        avg_polarity  = float(np.mean([polarity_map[s] for s in c_sents]))
        pol_vals      = [polarity_map[s] for s in c_sents]
        volatility    = float(np.std(pol_vals)) if len(pol_vals) > 1 else 0.0
        emo_weight    = round(avg_intensity * avg_polarity * (1.0 - volatility * 0.5), 4)

        raw_nds   = volume * growth_rate * stability * emo_weight
        sent_dist = {
            "positive": c_sents.count("positive"),
            "neutral":  c_sents.count("neutral"),
            "negative": c_sents.count("negative"),
        }

        # Dominant sentiment = label with highest weighted count.
        # Tie-break rule: neutral beats positive/negative on a tie
        # (avoids over-stating emotional polarity on ambiguous clusters).
        def _dominant(sd):
            mx = max(sd.values())
            leaders = [k for k, v in sd.items() if v == mx]
            if len(leaders) == 1:
                return leaders[0]
            return "neutral" if "neutral" in leaders else leaders[0]

        nds_scores[cid] = {
            "volume":             volume,
            "growth_rate":        growth_rate,
            "stability":          stability,
            "emotional_weight":   emo_weight,
            "volatility":         round(volatility, 4),
            "raw_nds":            round(raw_nds, 6),
            "sentiment_dist":     sent_dist,
            "dominant_sentiment": _dominant(sent_dist),
        }

    # Normalise to 0–100 with guaranteed spread
    # Use min-max normalisation so scores are always spread across the full range.
    # If all raw scores are identical (edge case), fall back to rank-based scoring.
    raw_vals = [v["raw_nds"] for v in nds_scores.values()]
    min_raw  = min(raw_vals)
    max_raw  = max(raw_vals)
    spread   = max_raw - min_raw

    ranked = sorted(nds_scores, key=lambda c: nds_scores[c]["raw_nds"], reverse=True)

    for rank, cid in enumerate(ranked):
        if spread > 0:
            # Min-max: highest → 100, lowest → 0, others spread between
            normalised = (nds_scores[cid]["raw_nds"] - min_raw) / spread * 100
        else:
            # All identical → assign rank-based scores (100, 75, 50, 25...)
            normalised = max(0.0, 100.0 - rank * (100.0 / max(len(ranked), 1)))

        nds_scores[cid]["nds"]      = round(normalised, 2)
        nds_scores[cid]["rank"]     = rank + 1
        nds_scores[cid]["dominant"] = (rank == 0)

    return nds_scores


# sentiment timeline
def build_sentiment_timeline(
    articles:   list[dict],
    labels:     np.ndarray,
    sent_probs: np.ndarray,
    n_clusters: int,
) -> dict[int, list[dict]]:
    clusters: dict[int, list[dict]] = defaultdict(list)

    for i, lbl in enumerate(labels):
        art   = articles[i]
        probs = sent_probs[i]
        score = sentiment_to_score(art["sentiment"], probs, label_enc)

        raw_ts = art.get("publishedAt", "")
        try:
            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            ts_iso = ts.isoformat()
        except Exception:
            ts_iso = None   # will be filled below

        clusters[int(lbl)].append({
            "timestamp":       ts_iso,
            "sentiment_score": score,
            "sentiment":       art["sentiment"],
            "confidence":      art.get("sentiment_conf", 0.0),
            "title":           art.get("title", "")[:80],
            "source":          art.get("source", ""),
        })

    # Fill missing timestamps synthetically so graph renders
    from datetime import timedelta
    now = datetime.utcnow()

    timeline: dict[int, list[dict]] = {}
    for cid, points in clusters.items():
        missing = [p for p in points if p["timestamp"] is None]
        present = [p for p in points if p["timestamp"] is not None]

        # Space missing timestamps evenly across last 30 days
        if missing:
            for j, p in enumerate(missing):
                offset = timedelta(hours=(j + 1) * (30 * 24 / (len(missing) + 1)))
                p["timestamp"] = (now - timedelta(days=30) + offset).isoformat()

        all_points = present + missing
        all_points.sort(key=lambda p: p["timestamp"])
        timeline[cid] = all_points

    return timeline


# ── App & Routes ──────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan, title="NarrativeMap API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def home() -> dict:
    return {
        "message":  "NarrativeMap API",
        "features": ["Google News RSS (30-day window)", "GDELT-trained sentiment",
                     "NDS scoring", "Temporal sentiment timeline"],
    }


def _fetch_gnews_rss(query: str, max_articles: int = GNEWS_MAX_ARTICLES) -> list[dict]:
    seen_urls: set[str] = set()
    articles:  list[dict] = []

    cutoff = datetime.now(timezone.utc) - timedelta(days=GNEWS_LOOKBACK_DAYS)

    # Build a list of (after, before) date pairs — one per week
    windows: list[tuple[str, str]] = []
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    while start >= cutoff - timedelta(days=7):
        windows.append((
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        ))
        end   = start
        start = start - timedelta(days=7)

    # Queries: plain topic first, then each date-window variant
    base_q = quote_plus(query)
    queries = [GNEWS_RSS_URL.format(query=base_q)]
    for after, before in windows:
        windowed_q = quote_plus(f"{query} after:{after} before:{before}")
        queries.append(GNEWS_RSS_URL.format(query=windowed_q))

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; NarrativeMap/1.0; "
        )
    }

    for url in queries:
        if len(articles) >= max_articles:
            break
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            print(f"  [RSS] Skipping {url[:80]}… — {exc}")
            continue

        for item in root.iter("item"):
            if len(articles) >= max_articles:
                break

            title   = (item.findtext("title")   or "").strip()
            pub_raw = (item.findtext("pubDate")  or "").strip()
            source  = (item.findtext("source")   or "Google News").strip()

            link = ""
            for child in item:
                if child.tag == "link" and child.text and child.text.strip().startswith("http"):
                    link = child.text.strip()
                    break
                if child.tail and child.tail.strip().startswith("http"):
                    link = child.tail.strip()
                    break
            if not title or not link or link in seen_urls:
                continue
            try:
                pub_dt  = parsedate_to_datetime(pub_raw)
                if pub_dt < cutoff:
                    continue
                pub_iso = pub_dt.isoformat()
            except Exception:
                continue   # reject — cannot confirm recency

            seen_urls.add(link)
            articles.append({
                "title":       title,
                "source":      source,
                "publishedAt": pub_iso,
                "url":         link,
            })

    print(f"  [RSS] Collected {len(articles)} unique articles over last {GNEWS_LOOKBACK_DAYS} days.")
    return articles


@app.get("/news")
def fetch_news(topic: str, n_clusters: int = 3) -> dict:
    try:
        # 1 — Fetch live news via Google RSS
        print(f"\n[1/5] Fetching Google News RSS for '{topic}' (last {GNEWS_LOOKBACK_DAYS} days)...")
        raw_articles = _fetch_gnews_rss(topic)
        if not raw_articles:
            return {"error": f"No articles found for '{topic}'. Try a broader search term."}

        # 2 — Clean
        print(f"[2/5] Cleaning {len(raw_articles)} articles...")
        cleaned: list[dict] = []
        for art in raw_articles:
            title = art.get("title") or ""
            # RSS titles from Google News often contain " - Publisher" suffix; strip it
            title_clean = re.sub(r"\s*[-–]\s*[^-–]+$", "", title).strip() or title
            text  = clean_text(title_clean)
            if len(text.split()) >= 4:
                cleaned.append({
                    "title":       title_clean,
                    "text":        text,
                    "source":      art.get("source", "Unknown"),
                    "publishedAt": art.get("publishedAt", ""),
                    "url":         art.get("url", ""),
                })

        if not cleaned:
            return {"error": "Articles found but all too short after cleaning."}

        # 3 — Embed
        print(f"[3/5] Embedding {len(cleaned)} articles...")
        texts      = [a["text"] for a in cleaned]
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=False)

        # 4 — Sentiment (GDELT-trained model)
        print("[4/5] Classifying sentiment with GDELT model...")
        sent_ids   = sentiment_clf.predict(embeddings)
        sent_probs = sentiment_clf.predict_proba(embeddings)

        classes = list(label_enc.classes_)
        pos_idx = classes.index("positive") if "positive" in classes else -1
        neg_idx = classes.index("negative") if "negative" in classes else -1
        neu_idx = classes.index("neutral")  if "neutral"  in classes else -1

        sentiments = []
        for prob_row in sent_probs:
            best_idx   = int(prob_row.argmax())
            best_label = classes[best_idx]
            best_conf  = prob_row[best_idx]

            # If confidence in positive/negative is below threshold → neutral
            if best_label in ("positive", "negative") and best_conf < 0.45:
                sentiments.append("neutral")
            else:
                sentiments.append(best_label)

        for i, art in enumerate(cleaned):
            art["sentiment"]       = sentiments[i]
            art["sentiment_conf"]  = round(float(sent_probs[i].max()), 3)
            art["sentiment_score"] = sentiment_to_score(sentiments[i], sent_probs[i], label_enc)

        # 5 — Cluster + NDS + Timeline
        print("[5/5] Clustering, computing NDS & building timeline...")
        k      = min(n_clusters, len(cleaned))
        labels = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(embeddings)
        for i, art in enumerate(cleaned):
            art["narrative"] = int(labels[i])

        nds      = compute_nds(cleaned, labels, embeddings, sentiments, sent_probs)
        timeline = build_sentiment_timeline(cleaned, labels, sent_probs, k)

        dominant_id = next(cid for cid, v in nds.items() if v["dominant"])

        # Attach NDS score to each timeline point for reference
        for cid in timeline:
            nds_score = nds.get(cid, {}).get("nds", 0)
            for pt in timeline[cid]:
                pt["nds_score"] = nds_score

        print(f"Done — {len(cleaned)} articles · {k} narratives · timeline built.")
        return {
            "topic":              topic,
            "total_articles":     len(cleaned),
            "articles":           cleaned,
            "nds":                nds,
            "dominant_narrative": dominant_id,
            "sentiment_timeline": {str(k): v for k, v in timeline.items()},
        }

    except requests.exceptions.Timeout:
        return {"error": "Google News RSS request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach Google News. Check your internet connection."}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.get("/train-status")
def train_status() -> dict:
    trained = os.path.exists(SENTIMENT_PATH) and os.path.exists(ENCODER_PATH)
    return {"gdelt_model_trained": trained, "model_path": SENTIMENT_PATH if trained else None}