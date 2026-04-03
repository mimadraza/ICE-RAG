import re
import json
import math
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import tiktoken
except ImportError:
    raise ImportError("Run: pip install tiktoken")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("Run: pip install langchain-text-splitters")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    raise ImportError("Run: pip install sentence-transformers scikit-learn numpy")


BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DIR   = BASE_DIR / "clean_md"
OUTPUT_DIR  = BASE_DIR / "Test" / "chunking"
OUTPUT_CSV_FIXED     = OUTPUT_DIR / "chunks_fixed.csv"
OUTPUT_CSV_RECURSIVE = OUTPUT_DIR / "chunks_recursive.csv"
OUTPUT_CSV_SEMANTIC  = OUTPUT_DIR / "chunks_semantic.csv"
REPORT_FILE = OUTPUT_DIR / "chunking_report.txt"

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"\nCould not find clean_md/ at: {INPUT_DIR}\n"
        f"Put your .md files there first."
    )


CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
SEMANTIC_THRESHOLD = 0.85
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"


TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def strip_markdown(text: str) -> str:
    """
    Light markdown cleanup so chunking works better on .md files.
    Removes code fences, inline code ticks, heading markers, links, images,
    and extra formatting markers while preserving readable text.
    """
    # Remove fenced code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Convert markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove markdown images ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

    # Remove heading markers
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"(\*\*|__|\*|_)", "", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter that handles legal text patterns:
    abbreviations (U.S., No.), numbered lists (1. 2.), and bullet points.
    """
    protected = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|U\.S|No|vs|etc|Art|Sec)\.", r"\1<DOT>", text)
    protected = re.sub(r"(\d+)\.", r"\1<DOT>", protected)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", protected)
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def fixed_size_chunks(text: str) -> list[str]:
    tokens = TOKENIZER.encode(text)
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_toks = tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_toks).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += step

    return chunks


def recursive_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        chunk_size=CHUNK_SIZE * 4,
        chunk_overlap=CHUNK_OVERLAP * 4,
        length_function=len,
        is_separator_regex=False,
    )
    raw_chunks = splitter.split_text(text)
    return [c.strip() for c in raw_chunks if len(c.strip()) > 30]


def semantic_chunks(text: str, model: SentenceTransformer) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)

    chunks = []
    current_sents = [sentences[0]]
    current_tokens = count_tokens(sentences[0])
    current_emb = embeddings[0].reshape(1, -1)

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_tokens = count_tokens(sent)
        sent_emb = embeddings[i].reshape(1, -1)

        sim = cosine_similarity(current_emb, sent_emb)[0][0]
        would_exceed = (current_tokens + sent_tokens) > CHUNK_SIZE
        low_sim = sim < SEMANTIC_THRESHOLD

        if would_exceed or low_sim:
            chunk_text = " ".join(current_sents).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_sents = [sent]
            current_tokens = sent_tokens
            current_emb = sent_emb
        else:
            current_sents.append(sent)
            current_tokens += sent_tokens
            current_emb = np.mean(
                [current_emb[0], sent_emb[0]], axis=0
            ).reshape(1, -1)

    if current_sents:
        chunk_text = " ".join(current_sents).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def build_rows(doc_id: str, filename: str, chunks: list[str], strategy: str) -> list[dict]:
    rows = []
    for i, chunk_text in enumerate(chunks):
        rows.append({
            "chunk_id": f"{doc_id}_{strategy}_{i:04d}",
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "strategy": strategy,
            "chunk_size_tokens": count_tokens(chunk_text),
            "word_count": len(chunk_text.split()),
            "text": chunk_text,
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect markdown files
    md_files = sorted(INPUT_DIR.glob("*.md"))

    if not md_files:
        print(f"No .md files found in {INPUT_DIR}")
        return

    print(f"Found {len(md_files)} markdown files.")
    print(f"Loading embedding model '{EMBEDDING_MODEL}' for semantic chunking...")
    sem_model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")

    all_rows = []
    stats = {"fixed": 0, "recursive": 0, "semantic": 0}
    doc_stats = []

    for idx, md_path in enumerate(md_files, 1):
        doc_id = md_path.stem
        filename = md_path.name

        raw_text = md_path.read_text(encoding="utf-8").strip()
        text = strip_markdown(raw_text)

        if not text:
            print(f"[{idx:02d}] {md_path.name} — EMPTY, skipping")
            continue

        fixed_c = fixed_size_chunks(text)
        recur_c = recursive_chunks(text)
        seman_c = semantic_chunks(text, sem_model)

        all_rows += build_rows(doc_id, filename, fixed_c, "fixed")
        all_rows += build_rows(doc_id, filename, recur_c, "recursive")
        all_rows += build_rows(doc_id, filename, seman_c, "semantic")

        stats["fixed"] += len(fixed_c)
        stats["recursive"] += len(recur_c)
        stats["semantic"] += len(seman_c)

        doc_stats.append({
            "doc_id": doc_id,
            "fixed": len(fixed_c),
            "recursive": len(recur_c),
            "semantic": len(seman_c),
        })

        print(
            f"[{idx:02d}/{len(md_files)}] {md_path.name:30} "
            f"fixed={len(fixed_c):4}  recursive={len(recur_c):4}  semantic={len(seman_c):4}"
        )

    df = pd.DataFrame(all_rows, columns=[
        "chunk_id", "doc_id", "filename", "chunk_index",
        "strategy", "chunk_size_tokens", "word_count", "text"
    ])

    csv_map = {
        "fixed": OUTPUT_CSV_FIXED,
        "recursive": OUTPUT_CSV_RECURSIVE,
        "semantic": OUTPUT_CSV_SEMANTIC,
    }

    for strat, path in csv_map.items():
        subset = df[df["strategy"] == strat].reset_index(drop=True)
        subset.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {len(subset):,} rows → {path.relative_to(BASE_DIR)}")

    sep = "=" * 60
    lines = [
        sep,
        "CHUNKING REPORT — Markdown Corpus",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Chunk size : {CHUNK_SIZE} tokens  |  Overlap: {CHUNK_OVERLAP} tokens",
        f"Sem. model : {EMBEDDING_MODEL}",
        f"Sem. thresh: {SEMANTIC_THRESHOLD}",
        sep,
        "",
        "Total chunks per strategy:",
        f"  Fixed-size : {stats['fixed']:,}",
        f"  Recursive  : {stats['recursive']:,}",
        f"  Semantic   : {stats['semantic']:,}",
        "",
        "Average tokens per chunk:",
    ]

    for strat in ["fixed", "recursive", "semantic"]:
        sub = df[df["strategy"] == strat]["chunk_size_tokens"]
        if len(sub) > 0:
            lines.append(
                f"  {strat:<12}: avg={sub.mean():.0f}  min={sub.min()}  max={sub.max()}"
            )
        else:
            lines.append(f"  {strat:<12}: no chunks")

    lines += [
        "",
        sep,
        f"{'Doc':<24} {'Fixed':>7} {'Recursive':>10} {'Semantic':>9}",
        "-" * 55,
        *[
            f"{d['doc_id']:<24} {d['fixed']:>7} {d['recursive']:>10} {d['semantic']:>9}"
            for d in doc_stats
        ],
    ]

    report_text = "\n".join(lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nReport saved → {REPORT_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()