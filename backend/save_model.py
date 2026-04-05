import os
from sentence_transformers import SentenceTransformer
MODEL_NAME="all-MiniLM-L6-v2"
SAVE_PATH=f"./models/{MODEL_NAME}"

if os.path.exists(SAVE_PATH):
    print(f"Model already exists at '{SAVE_PATH}'.")
else:
    os.makedirs("./models", exist_ok=True)
    m=SentenceTransformer(MODEL_NAME)
    m.save(SAVE_PATH)
m=SentenceTransformer(SAVE_PATH)
emb=m.encode(["test sentence"])
print(f"Embedding shape: {emb.shape} — model is ready.")
