"""
Part 1: Document Ingestion System
==================================
Implements:
  1. 2-page Sliding Window chunking (character-based with overlap)
  2. Hierarchical Knowledge Pyramid (4 layers)
  3. Query retrieval via cosine similarity across pyramid levels

Author  : Candidate
Project : Vexoo Labs AI Engineer Assignment
"""

from __future__ import annotations

import re
import math
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import Counter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants  (tunable via environment / config)
# ---------------------------------------------------------------------------
PAGE_CHARS       = 2000   # ~1 page ≈ 2 000 chars (A4, 12pt)
WINDOW_PAGES     = 2      # sliding window = 2 pages
OVERLAP_CHARS    = 400    # ~20 % overlap keeps inter-chunk context
MAX_SUMMARY_SENT = 3      # sentences kept for chunk summary

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","on","at","by","for","with","about","from",
    "that","this","these","those","it","its","not","and","or","but","if",
}

CATEGORY_RULES: Dict[str, List[str]] = {
    "Machine Learning / AI" : [
        "neural","model","training","dataset","accuracy","loss","gradient",
        "deep","learning","transformer","embedding","inference","llm","gpt",
        "bert","fine-tune","lora","tokenize","epoch","batch","overfitting",
    ],
    "Mathematics" : [
        "equation","theorem","proof","integer","prime","matrix","vector",
        "calculus","derivative","integral","algebra","geometry","probability",
        "statistics","distribution","variance","correlation",
    ],
    "Natural Language Processing" : [
        "nlp","text","corpus","sentence","token","parse","syntax","semantic",
        "sentiment","classification","summarization","ner","pos","dependency",
    ],
    "Software Engineering" : [
        "api","function","class","module","library","framework","database",
        "schema","query","server","client","architecture","design","pattern",
    ],
    "General / Other" : [],     # fallback
}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RawChunk:
    """Layer 0 – raw text window."""
    chunk_id   : str
    text       : str
    start_char : int
    end_char   : int

@dataclass
class PyramidNode:
    """
    Hierarchical Knowledge Pyramid node.

    Layer 0 – raw_text     : exact window content
    Layer 1 – summary      : first N sentences (placeholder summarisation)
    Layer 2 – category     : rule-based theme label
    Layer 3 – distilled    : top-K keyword vector (mocked embeddings)
    """
    chunk_id  : str
    raw_text  : str          # Layer 0
    summary   : str          # Layer 1
    category  : str          # Layer 2
    keywords  : List[str]    # Layer 3 (distilled knowledge)
    tfidf_vec : Dict[str, float] = field(default_factory=dict)   # sparse vector

# ---------------------------------------------------------------------------
# Helper: simple TF-IDF sparse vector (no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase alphanum tokens, strip stopwords."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _tf(tokens: List[str]) -> Dict[str, float]:
    c = Counter(tokens)
    n = max(len(tokens), 1)
    return {w: cnt / n for w, cnt in c.items()}


def build_tfidf_corpus(docs: List[List[str]]) -> List[Dict[str, float]]:
    """
    Compute TF-IDF vectors for all docs.
    Each doc is a token list.  Returns list of sparse {term: score} dicts.
    """
    N = len(docs)
    df: Counter = Counter()
    for doc_tokens in docs:
        df.update(set(doc_tokens))

    idf: Dict[str, float] = {
        w: math.log((N + 1) / (cnt + 1)) + 1.0   # smoothed IDF
        for w, cnt in df.items()
    }

    vectors = []
    for doc_tokens in docs:
        tf_scores = _tf(doc_tokens)
        vec = {w: tf_scores[w] * idf[w] for w in tf_scores}
        vectors.append(vec)
    return vectors


def cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0
    dot = sum(v1[w] * v2[w] for w in shared)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))
    return dot / (mag1 * mag2 + 1e-9)


# ---------------------------------------------------------------------------
# Stage 1 – Sliding Window Chunker
# ---------------------------------------------------------------------------

class SlidingWindowChunker:
    """
    Splits a document into overlapping windows of WINDOW_PAGES pages.

    Parameters
    ----------
    page_chars   : characters representing one logical page
    window_pages : number of pages per window
    overlap_chars: character overlap between consecutive windows
    """

    def __init__(
        self,
        page_chars: int   = PAGE_CHARS,
        window_pages: int = WINDOW_PAGES,
        overlap_chars: int = OVERLAP_CHARS,
    ) -> None:
        self.window_size = page_chars * window_pages   # e.g. 4 000
        self.overlap     = overlap_chars
        self.stride      = self.window_size - overlap_chars

    def chunk(self, text: str) -> List[RawChunk]:
        """Return a list of RawChunk windows covering the full text."""
        chunks: List[RawChunk] = []
        start = 0
        doc_len = len(text)

        while start < doc_len:
            end = min(start + self.window_size, doc_len)
            chunk_text = text[start:end]

            # Snap end to word boundary (avoid mid-word cuts)
            if end < doc_len:
                last_space = chunk_text.rfind(" ")
                if last_space > 0:
                    end = start + last_space
                    chunk_text = text[start:end]

            cid = hashlib.md5(f"{start}-{end}".encode()).hexdigest()[:8]
            chunks.append(RawChunk(cid, chunk_text.strip(), start, end))
            log.debug("Chunk %s | chars %d-%d", cid, start, end)

            if end >= doc_len:
                break
            start = end - self.overlap   # shift by stride, keep overlap

        log.info("Sliding window produced %d chunks", len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Stage 2 – Knowledge Pyramid Builder
# ---------------------------------------------------------------------------

class KnowledgePyramidBuilder:
    """
    Transforms a RawChunk into a 4-layer PyramidNode.

    Layer 0  Raw Text      : kept verbatim
    Layer 1  Chunk Summary : first MAX_SUMMARY_SENT sentences
    Layer 2  Category      : rule-based keyword match → theme label
    Layer 3  Distilled     : top-15 TF-IDF keywords (mocked embedding)
    """

    # ------------------------------------------------------------------ #
    # Layer 1 – placeholder summarisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _summarise(text: str, n_sents: int = MAX_SUMMARY_SENT) -> str:
        """
        Extractive summary: return the first n_sents sentences.
        In production swap this for a real summariser (e.g. BART, T5).
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = [s.strip() for s in sentences if s.strip()][:n_sents]
        return " ".join(kept)

    # ------------------------------------------------------------------ #
    # Layer 2 – rule-based category / theme
    # ------------------------------------------------------------------ #
    @staticmethod
    def _categorise(text: str) -> str:
        """
        Assign a category label by counting keyword hits per category.
        Returns the category with the highest hit count.
        """
        lower = text.lower()
        scores: Dict[str, int] = {}
        for cat, keywords in CATEGORY_RULES.items():
            scores[cat] = sum(1 for kw in keywords if kw in lower)

        # Remove 'General / Other' from competition; use as fallback
        general = scores.pop("General / Other", 0)
        best_cat = max(scores, key=scores.get) if scores else "General / Other"
        if scores.get(best_cat, 0) == 0:
            return "General / Other"
        return best_cat

    # ------------------------------------------------------------------ #
    # Layer 3 – distilled knowledge (keyword list / mock embedding)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _distil(text: str, top_k: int = 15) -> List[str]:
        """
        Extract top-K keywords by TF score within the chunk.
        In production replace with a real embedding (e.g. sentence-transformers).
        """
        tokens = _tokenize(text)
        tf = _tf(tokens)
        ranked = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:top_k]]

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def build(self, chunk: RawChunk) -> PyramidNode:
        summary  = self._summarise(chunk.text)
        category = self._categorise(chunk.text)
        keywords = self._distil(chunk.text)

        node = PyramidNode(
            chunk_id = chunk.chunk_id,
            raw_text = chunk.text,
            summary  = summary,
            category = category,
            keywords = keywords,
        )
        log.debug("Pyramid node %s | cat=%s | kw=%s", node.chunk_id, category, keywords[:5])
        return node


# ---------------------------------------------------------------------------
# Stage 3 – Index + Retrieval Engine
# ---------------------------------------------------------------------------

class PyramidIndex:
    """
    Stores all PyramidNodes and supports multi-level semantic retrieval.

    Strategy
    --------
    For a query we compute TF-IDF similarity against THREE representations:
      (a) full raw text
      (b) summary
      (c) keyword string
    A weighted sum gives a final relevance score per node.

    The weights favour the level that best matches the query intent:
      raw_text  : 0.4
      summary   : 0.35
      keywords  : 0.25
    """

    LEVEL_WEIGHTS = {"raw": 0.40, "summary": 0.35, "keywords": 0.25}

    def __init__(self) -> None:
        self.nodes: List[PyramidNode] = []
        self._raw_vecs   : List[Dict[str, float]] = []
        self._sum_vecs   : List[Dict[str, float]] = []
        self._kw_vecs    : List[Dict[str, float]] = []

    def add_nodes(self, nodes: List[PyramidNode]) -> None:
        self.nodes = nodes
        raw_tokens = [_tokenize(n.raw_text)              for n in nodes]
        sum_tokens = [_tokenize(n.summary)               for n in nodes]
        kw_tokens  = [_tokenize(" ".join(n.keywords))    for n in nodes]

        self._raw_vecs = build_tfidf_corpus(raw_tokens)
        self._sum_vecs = build_tfidf_corpus(sum_tokens)
        self._kw_vecs  = build_tfidf_corpus(kw_tokens)

        # Store per-node for export
        for i, node in enumerate(nodes):
            node.tfidf_vec = self._raw_vecs[i]

        log.info("Index built with %d pyramid nodes", len(nodes))

    def query(
        self,
        query: str,
        top_k: int = 3,
        level: Optional[str] = None,
    ) -> List[Tuple[PyramidNode, float, str]]:
        """
        Retrieve top-k relevant nodes.

        Parameters
        ----------
        query : natural-language question or search string
        top_k : number of results to return
        level : if set to 'raw'|'summary'|'keywords', search only that level

        Returns
        -------
        List of (node, score, matched_level)
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Build single-doc IDF for query vector (just TF in practice)
        q_vec = _tf(q_tokens)

        results: List[Tuple[float, int, str]] = []

        for i, node in enumerate(self.nodes):
            if level is None or level == "raw":
                raw_score = cosine_sim(q_vec, self._raw_vecs[i])
            else:
                raw_score = 0.0

            if level is None or level == "summary":
                sum_score = cosine_sim(q_vec, self._sum_vecs[i])
            else:
                sum_score = 0.0

            if level is None or level == "keywords":
                kw_score  = cosine_sim(q_vec, self._kw_vecs[i])
            else:
                kw_score  = 0.0

            w = self.LEVEL_WEIGHTS
            composite = (
                w["raw"]     * raw_score +
                w["summary"] * sum_score +
                w["keywords"]* kw_score
            )

            # Determine which pyramid level was most informative
            scores = {"raw": raw_score, "summary": sum_score, "keywords": kw_score}
            best_level = max(scores, key=scores.get)

            results.append((composite, i, best_level))

        results.sort(reverse=True)
        output = []
        for score, idx, lv in results[:top_k]:
            output.append((self.nodes[idx], round(score, 4), lv))
        return output


# ---------------------------------------------------------------------------
# Top-level Pipeline
# ---------------------------------------------------------------------------

class DocumentIngestionPipeline:
    """
    Orchestrates: Load → Chunk → Build Pyramid → Index → Query.

    Usage
    -----
    >>> pipeline = DocumentIngestionPipeline()
    >>> pipeline.ingest_text("Your long document text here...")
    >>> results = pipeline.search("What is gradient descent?")
    >>> for node, score, level in results:
    ...     print(node.summary, score, level)
    """

    def __init__(
        self,
        page_chars: int    = PAGE_CHARS,
        window_pages: int  = WINDOW_PAGES,
        overlap_chars: int = OVERLAP_CHARS,
    ) -> None:
        self.chunker = SlidingWindowChunker(page_chars, window_pages, overlap_chars)
        self.builder = KnowledgePyramidBuilder()
        self.index   = PyramidIndex()
        self.nodes   : List[PyramidNode] = []

    # ------------------------------------------------------------------
    def ingest_text(self, text: str) -> None:
        """Full ingestion: chunk → pyramid → index."""
        log.info("Ingestion started | doc_len=%d chars", len(text))
        chunks = self.chunker.chunk(text)
        nodes  = [self.builder.build(c) for c in chunks]
        self.index.add_nodes(nodes)
        self.nodes = nodes
        log.info("Ingestion complete | %d pyramid nodes indexed", len(nodes))

    def ingest_file(self, filepath: str) -> None:
        """Read plain-text file and ingest."""
        with open(filepath, "r", encoding="utf-8") as fh:
            self.ingest_text(fh.read())

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 3,
        level: Optional[str] = None,
    ) -> List[Tuple[PyramidNode, float, str]]:
        """Query the pyramid index and return top_k results."""
        log.info("Query: '%s'", query)
        results = self.index.query(query, top_k=top_k, level=level)
        for node, score, lv in results:
            log.info("  → [%s] chunk=%s score=%.4f | %s", lv, node.chunk_id, score, node.summary[:80])
        return results

    # ------------------------------------------------------------------
    def export_index(self, path: str) -> None:
        """Dump all pyramid nodes to JSON for inspection."""
        data = []
        for n in self.nodes:
            d = asdict(n)
            d.pop("tfidf_vec", None)   # too verbose
            data.append(d)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log.info("Index exported to %s", path)

    # ------------------------------------------------------------------
    def print_pyramid(self, node: PyramidNode) -> None:
        """Pretty-print all 4 pyramid layers for a node."""
        sep = "─" * 70
        print(sep)
        print(f"🔷 Chunk ID  : {node.chunk_id}")
        print(sep)
        print("📄 [Layer 0] Raw Text (first 300 chars):")
        print(f"   {node.raw_text[:300]}...")
        print()
        print("📝 [Layer 1] Chunk Summary:")
        print(f"   {node.summary}")
        print()
        print("🏷  [Layer 2] Category / Theme:")
        print(f"   {node.category}")
        print()
        print("🔑 [Layer 3] Distilled Keywords:")
        print(f"   {', '.join(node.keywords)}")
        print(sep)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

IBM has a rich history with machine learning. One of its own, Arthur Samuel, is credited for coining the term, "machine learning" with his research around the game of checkers. Robert Nealey, the self-proclaimed checkers master, played the game on an IBM 7094 computer in 1962, and he lost to the computer. Compared to what can be done today, this feat seems trivial, but it's considered a major milestone in the field of artificial intelligence.

Over the last couple of decades, the technological advances in storage and processing power have enabled some innovative products based on machine learning, such as Netflix's recommendation engine and self-driving cars.

Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase. They will be required to help identify the most relevant business questions and the data to answer them.

Supervised learning, also known as supervised machine learning, is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process. Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets. These algorithms discover hidden patterns or data groupings without the need for human intervention. Its ability to discover similarities and differences in information make it the ideal solution for exploratory data analysis, cross-selling strategies, customer segmentation, and image recognition.

Semi-supervised learning offers a happy medium between supervised and unsupervised learning. During training, it uses a smaller labeled data set to guide classification and feature extraction from a larger, unlabeled data set. Semi-supervised learning can solve the problem of having not enough labeled data (or not being able to afford to label enough data) to train a supervised learning algorithm.

Reinforcement learning is a machine learning training method based on rewarding desired behaviors and/or punishing undesired ones. In general, a reinforcement learning agent is able to perceive and interpret its environment, take actions and learn through trial and error. Reinforcement learning is used in training robots to perform tasks, like walking around a room, and video game playing.

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

Natural language processing (NLP) is a type of machine learning that processes text. NLP applies machine learning algorithms to natural language processing tasks like sentiment analysis, text generation, and translation. BERT and GPT are popular NLP-specific pretrained models.

Transformers and the models trained on them have been called the most disruptive innovation in AI in the past 10 years. Because they can be used for a wide variety of NLP tasks and produce state of the art results, they have become extremely popular.

The training process for a transformer model involves feeding it large amounts of data. The model learns to predict the next word in a sequence by looking at all the words that came before it. This prediction task is known as language modeling. The data used in training can be anything from books to websites and requires a large amount of compute power.

Fine-tuning is the process of taking a pretrained model and training it on a specific dataset. This is useful for adapting a general model to a specific task or domain. LoRA (Low-Rank Adaptation) is a popular technique for efficient fine-tuning where the model's original weights are frozen and only small adapter matrices are trained.

Gradient descent is an optimization algorithm used to train machine learning models by minimizing the loss function. In each iteration, the model computes the gradient of the loss with respect to its parameters and updates them in the direction that reduces the loss. Variants include stochastic gradient descent (SGD), mini-batch gradient descent, Adam, and AdaGrad.
"""


def demo() -> None:
    print("\n" + "=" * 70)
    print("  VEXOO LABS – Part 1: Document Ingestion + Knowledge Pyramid Demo")
    print("=" * 70)

    pipeline = DocumentIngestionPipeline(page_chars=800, window_pages=2, overlap_chars=150)
    pipeline.ingest_text(SAMPLE_TEXT)

    # Show one pyramid node
    if pipeline.nodes:
        print("\n[Sample Pyramid Node]")
        pipeline.print_pyramid(pipeline.nodes[0])

    # Run queries
    queries = [
        "What is gradient descent and how does it work?",
        "Explain transformer models and fine-tuning",
        "What is reinforcement learning?",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = pipeline.search(q, top_k=2)
        for rank, (node, score, level) in enumerate(results, 1):
            print(f"  [{rank}] score={score:.4f}  level={level}  chunk={node.chunk_id}")
            print(f"      Category : {node.category}")
            print(f"      Summary  : {node.summary[:120]}...")

    # Export index
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _logs_dir = _os.path.join(_script_dir, "..", "logs")
    _os.makedirs(_logs_dir, exist_ok=True)
    pipeline.export_index(_os.path.join(_logs_dir, "pyramid_index.json"))
    print("\n✅ Demo complete. Index exported to logs/pyramid_index.json\n")


if __name__ == "__main__":
    demo()
