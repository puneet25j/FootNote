"""Build and query the Chroma index from data/processed JSON.

Usage (from the Footnote folder):
    python -m footnote.index build
    python -m footnote.index search "what is a peak?"
"""
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer

from footnote.chunking import load_all_chunks

INDEX_DIR = "data/index"
COLLECTION = "footnote"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def embedder() -> SentenceTransformer:
    """Load the embedding model once, on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    return embedder().encode(texts, normalize_embeddings=True).tolist()


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=INDEX_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass  # nothing to delete on the very first build
    # embedding_function=None: we always supply our own vectors
    return client.get_or_create_collection(COLLECTION, embedding_function=None)


def build() -> None:
    t0 = time.perf_counter()
    chunks = load_all_chunks()
    texts = [c["text"] for c in chunks]
    collection = get_collection(reset=True)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embed(texts),
    )
    print(f"indexed {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s -> {INDEX_DIR}")


def search(query: str, k: int = 5) -> list[dict]:
    res = get_collection().query(
        query_embeddings=embed([QUERY_PREFIX + query]), n_results=k
    )
    return [
        {"id": i, "text": t, "metadata": m, "score": 1 - d / 2}
        for i, t, m, d in zip(res["ids"][0], res["documents"][0],
                              res["metadatas"][0], res["distances"][0])
    ]


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def cite(meta: dict) -> str:
    """Build a human-readable citation from chunk metadata (never from the LLM)."""
    if meta["source_type"] == "video":
        return f"{meta['lecture']} video {fmt_ts(meta['start'])}-{fmt_ts(meta['end'])}"
    if meta["page_start"] == meta["page_end"]:
        return f"{meta['lecture']} notes p. {meta['page_start']}"
    return f"{meta['lecture']} notes pp. {meta['page_start']}-{meta['page_end']}"


def main() -> None:
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["search"] and len(sys.argv) > 2:
        for rank, hit in enumerate(search(" ".join(sys.argv[2:])), 1):
            print(f"{rank}. [{hit['score']:.3f}] {cite(hit['metadata'])}")
            print(f"   {hit['text'][:200]}...")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()