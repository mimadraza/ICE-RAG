import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import CHUNK_FILES, EMBED_DIR, EMBED_MODEL


def build_embeddings(strategy: str):
    csv_path = CHUNK_FILES[strategy]
    df = pd.read_csv(csv_path)

    texts = df["text"].fillna("").tolist()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    out_dir = EMBED_DIR / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", embeddings)

    metadata = df.to_dict(orient="records")
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved {strategy} embeddings -> {out_dir}")


if __name__ == "__main__":
    for strat in ["fixed", "recursive", "semantic"]:
        build_embeddings(strat)