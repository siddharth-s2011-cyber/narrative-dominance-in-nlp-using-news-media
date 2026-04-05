import argparse
import io
import os
import re
import zipfile
import joblib
import pandas as pd
import requests
from pydantic_core.core_schema import to_string_ser_schema
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_PATH    = "./models/all-MiniLM-L6-v2"
SENTIMENT_OUT = "./models/sentiment_gdelt.joblib"
ENCODER_OUT   = "./models/label_encoder_gdelt.joblib"
GDELT_BASE    = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"
def download_gkg(date: str) -> pd.DataFrame:
    url = GDELT_BASE.format(date=date)
    print(f"[GDELT] Downloading {url} ...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(
            f, sep="\t", header=None,
            on_bad_lines="skip", encoding="utf-8", low_memory=False,
        )
    print(f"[GDELT] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    print(df.to_String())
    return df


def load_local(path: str) -> pd.DataFrame:
    print(f"[GDELT] Reading local file: {path}")
    df = pd.read_csv(path, sep="\t", header=None,
                     on_bad_lines="skip", encoding="utf-8", low_memory=False)
    print(f"[GDELT] Loaded {len(df):,} rows.")
    return df

def parse_gkg(df: pd.DataFrame, max_per_class: int = 1000) -> pd.DataFrame:
    if df.shape[1] < 8:
        raise ValueError(f"Only {df.shape[1]} columns found — need at least 8.")

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
    print(f"\n[Parse] Raw class distribution:\n{counts.to_string()}\n")

    min_n = min(counts.min(), max_per_class)
    out = (
        out.groupby("sentiment", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), min_n), random_state=42))
           .reset_index(drop=True)
    )
    print(f"[Parse] Balanced: {len(out)} rows ({min_n} per class).")
    return out[["text", "sentiment"]]
def seed_dataframe():
    return
def train(df: pd.DataFrame) -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Sentence-transformer not found at '{MODEL_PATH}'.\n"
            "Run  python save_model.py  first."
        )

    print(f"\n[Train] Loading sentence transformer from {MODEL_PATH}...")
    st_model = SentenceTransformer(MODEL_PATH)

    print(f"[Train] Encoding {len(df)} texts...")
    embeddings = st_model.encode(
        df["text"].tolist(), batch_size=64, show_progress_bar=True,
    )

    enc = LabelEncoder()
    y   = enc.fit_transform(df["sentiment"])

    X_tr, X_val, y_tr, y_val = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = SGDClassifier(
        loss="modified_huber",
        alpha=0.0001,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)

    print("\n[Train] Validation report:")
    preds = clf.predict(X_val)
    print(classification_report(y_val, preds, target_names=enc.classes_))

    os.makedirs("./models", exist_ok=True)
    joblib.dump(clf, SENTIMENT_OUT)
    joblib.dump(enc, ENCODER_OUT)
    print(f"\nSaved:\n  {SENTIMENT_OUT}\n  {ENCODER_OUT}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GDELT sentiment model")
    parser.add_argument("--date",      default="20200101",
                        help="GDELT GKG date (YYYYMMDD). Default: 20200101")
    parser.add_argument("--local",     default=None,
                        help="Path to a local .gkg.csv file")
    parser.add_argument("--seed-only", action="store_true",
                        help="Skip GDELT download, use seed data only")
    parser.add_argument("--max-per-class", type=int, default=1000,
                        help="Max samples per sentiment class. Default: 1000")
    args = parser.parse_args()

    if args.seed_only:
        df = seed_dataframe()
    else:
        try:
            if args.local:
                raw = load_local(args.local)
            else:
                raw = download_gkg(args.date)
            df = parse_gkg(raw, max_per_class=args.max_per_class)
        except Exception as e:
            print(f"\n[Warning] GDELT load failed: {e}\nFalling back to seed data.")
            df = pd.DataFrame
    train(df)
