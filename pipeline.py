"""
pipeline.py
───────────
Part D: Full RAG Pipeline
User Query → Retrieval → Context Selection → Prompt → LLM → Response

- Logging at every stage
- Displays retrieved docs, similarity scores, final prompt
- Integrates feedback memory (Part G innovation)
"""

import os
import json
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # loads from .env for local dev

from data_loader    import prepare_all_chunks
from retriever      import build_retriever, load_retriever, VectorStore
from prompt_builder import build_prompt

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ]
)

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "data/Ghana_Election_Result.csv"
PDF_PATH      = "data/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
FEEDBACK_PATH = "logs/feedback.json"
LOG_PATH      = "logs/query_log.jsonl"

# ── LLM Config ──────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.1-8b-instant"   # llama3-8b-8192 was decommissioned May 2025
MAX_TOKENS   = 1024

# ══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    """
    Orchestrates the complete RAG flow with stage-level logging.

    Stages:
      INIT  → load data, chunk, embed, build FAISS index
      QUERY → embed query, retrieve, build prompt, call LLM, return result
    """

    def __init__(self):
        self.retriever  = None
        self.ready      = False
        self.query_log  = []
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # INIT
    # ══════════════════════════════════════════════════════════════════════════

    def initialize(self, force_rebuild: bool = False, on_progress=None):
        """
        Build or load the FAISS index.
        on_progress: optional callable(str) for live UI progress updates.
        On first run builds from scratch; subsequent runs load from disk.
        """
        def _p(msg):
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        from retriever import (INDEX_PATH, CHUNKS_PATH,
                                EmbeddingPipeline, VectorStore, HybridRetriever)

        index_exists = (
            os.path.exists(INDEX_PATH) and
            os.path.exists(CHUNKS_PATH) and
            not force_rebuild
        )

        if index_exists:
            _p("📂 Existing index found — loading from disk…")
            _p("  → Loading embedding model into memory…")
            embedder = EmbeddingPipeline()
            _p("  → Reading FAISS index and chunk cache…")
            store = VectorStore(dim=embedder.dim)
            store.load()
            self.retriever = HybridRetriever(store, embedder)
        else:
            _p("🔨 No index found — building from scratch…")
            _p("🧠 Step 1/3 — Loading embedding model…")
            _p("   (First run: may download ~80 MB from HuggingFace — please wait)")
            embedder = EmbeddingPipeline()
            _p(f"   ✓ Model loaded  ({embedder.dim}-dim embeddings)")

            _p("📄 Step 2/3 — Loading & chunking documents…")
            chunks = prepare_all_chunks(CSV_PATH, PDF_PATH, chunking_strategy="sentence")
            _p(f"   ✓ {len(chunks)} chunks created")

            _p(f"⚡ Step 3/3 — Embedding {len(chunks)} chunks & building FAISS index…")
            _p("   (This takes 1-3 min on first run — index is cached after this)")
            store = VectorStore(dim=embedder.dim)
            store.add_chunks(chunks, embedder)
            store.save()
            self.retriever = HybridRetriever(store, embedder)
            _p("   ✓ FAISS index built and saved to disk")

            _p("💾 Loading feedback memory…")
            self._load_feedback_memory()

        self.ready = True
        _p("✅ Pipeline ready!")

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY
    # ══════════════════════════════════════════════════════════════════════════

    def query(self, query: str, top_k: int = 5,
              prompt_style: str = "Hallucination-Controlled") -> dict:
        """
        Full pipeline: query → retrieve → prompt → LLM → response.

        Returns dict with all intermediate data for display.
        """
        if not self.ready:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        all_logs = []
        ts = datetime.now().isoformat()

        logger.info("─" * 60)
        logger.info(f"[QUERY] '{query}' | top_k={top_k} | template={prompt_style}")
        all_logs.append(f"[QUERY] Received: {query}")

        # ── Stage 1: Retrieval ──────────────────────────────────────────────
        logger.info("[STAGE 1] Hybrid retrieval")
        all_logs.append("[STAGE 1] Retrieving relevant chunks")

        retrieved = self.retriever.retrieve(query, top_k=top_k, use_expansion=True)

        for r in retrieved:
            all_logs.extend(r.get("logs", []))
            logger.info(
                f"  → {r['source'][:40]} | "
                f"vec={r['vector_score']:.3f} kw={r['keyword_score']:.3f} "
                f"final={r['final_score']:.3f}"
                f"{' ⚠️' if r['is_failure'] else ''}"
            )

        # ── Stage 2: Context window management ─────────────────────────────
        logger.info("[STAGE 2] Building prompt")
        all_logs.append(f"[STAGE 2] Building prompt with template: {prompt_style}")

        prompt, kept_chunks = build_prompt(query, retrieved, prompt_style)
        all_logs.append(f"[STAGE 2] Context: {len(kept_chunks)} chunks, {len(prompt)} chars")
        logger.info(f"[STAGE 2] Prompt: {len(kept_chunks)} chunks, {len(prompt)} chars")

        # ── Stage 3: LLM Generation ─────────────────────────────────────────
        logger.info("[STAGE 3] Calling LLM")
        all_logs.append("[STAGE 3] Sending prompt to Claude API")

        answer = self._call_llm(prompt)
        all_logs.append(f"[STAGE 3] Response received ({len(answer)} chars)")
        logger.info(f"[STAGE 3] Response: {len(answer)} chars")

        # ── Stage 4: Package result ─────────────────────────────────────────
        result = {
            "query":    query,
            "answer":   answer,
            "chunks":   [c["text"]         for c in kept_chunks],
            "scores":   [c["final_score"]  for c in kept_chunks],
            "sources":  [c["source"]       for c in kept_chunks],
            "prompt":   prompt,
            "logs":     all_logs,
            "retrieved_full": retrieved,   # full metadata for experiment log
            "timestamp": ts,
        }

        # ── Stage 5: Log to disk ────────────────────────────────────────────
        self._log_query(result)

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # LLM CALL
    # ══════════════════════════════════════════════════════════════════════════

    def _call_llm(self, prompt: str) -> str:
        """
        Direct HTTP call to Anthropic API.
        No SDK — raw requests only (satisfies the 'no pre-built pipeline' constraint).
        """
        # Read from Streamlit secrets (cloud) or .env (local)
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY", "")

        if not api_key:
            return (
                "⚠️ GROQ_API_KEY not set. "
                "Add it to .streamlit/secrets.toml locally, or via the Streamlit Cloud dashboard."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":      MODEL,
            "max_tokens": MAX_TOKENS,
            "messages":   [{"role": "user", "content": prompt}],
        }

        try:
            resp = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data   = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return answer

        except requests.exceptions.Timeout:
            return "⚠️ LLM API timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            return f"⚠️ LLM API error: {e.response.status_code} — {e.response.text[:200]}"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING & FEEDBACK MEMORY (Part G)
    # ══════════════════════════════════════════════════════════════════════════

    def _log_query(self, result: dict):
        """Append query + result metadata to JSONL log for experiment analysis."""
        entry = {
            "timestamp": result["timestamp"],
            "query":     result["query"],
            "answer_preview": result["answer"][:120],
            "chunks_used":    len(result["chunks"]),
            "top_score":      result["scores"][0] if result["scores"] else 0,
            "sources":        list(set(result["sources"])),
        }
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _load_feedback_memory(self):
        """
        Part G — Innovation: Feedback loop memory.
        Reads past 👎 feedback and logs warnings about known bad queries.
        In a production system, this would retrain or re-weight the retriever.
        """
        if not os.path.exists(FEEDBACK_PATH):
            return

        try:
            with open(FEEDBACK_PATH) as f:
                feedback = json.load(f)

            bad = [fb for fb in feedback if fb.get("rating") == "👎"]
            if bad:
                logger.info(
                    f"[MEMORY] Loaded {len(bad)} negative feedback entries. "
                    f"These queries had poor results previously."
                )
                for fb in bad[-5:]:   # log last 5
                    logger.info(f"  ⚠️  Bad query: {fb.get('answer_preview', '')[:60]}")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # ADVERSARIAL TEST HELPERS (Part E)
    # ══════════════════════════════════════════════════════════════════════════

    def run_adversarial_tests(self) -> list[dict]:
        """
        Part E: Runs the 2 required adversarial queries and logs results.
        Also runs the same queries against LLM-only (no retrieval) for comparison.
        """
        adversarial_queries = [
            {
                "query":       "Who won?",
                "type":        "Ambiguous",
                "description": "No context — who won what? What year? What region?"
            },
            {
                "query":       "What was Ghana's debt in 2019 according to the 2025 budget?",
                "type":        "Misleading",
                "description": "Asks for 2019 data from a 2025 document — tests hallucination."
            },
        ]

        results = []
        for test in adversarial_queries:
            logger.info(f"[ADVERSARIAL] Running: {test['query']}")

            # RAG response
            rag_result = self.query(test["query"], top_k=5)

            # LLM-only response (no retrieval context)
            llm_only_prompt = f"Question: {test['query']}\nAnswer:"
            llm_only_answer = self._call_llm(llm_only_prompt)

            results.append({
                **test,
                "rag_answer":      rag_result["answer"],
                "rag_top_score":   rag_result["scores"][0] if rag_result["scores"] else 0,
                "llm_only_answer": llm_only_answer,
                "rag_chunks_used": len(rag_result["chunks"]),
            })

            logger.info(f"  RAG:      {rag_result['answer'][:80]}…")
            logger.info(f"  LLM-only: {llm_only_answer[:80]}…")

        # Save to file for evidence-based comparison
        with open("logs/adversarial_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("[ADVERSARIAL] Results saved to logs/adversarial_results.json")
        return results


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.initialize()

    result = pipeline.query(
        "What is Ghana's inflation target for 2025?",
        top_k=5,
        prompt_style="Hallucination-Controlled"
    )
    print("\n" + "="*60)
    print("ANSWER:", result["answer"])
    print("\nTOP CHUNKS:")
    for i, (chunk, score) in enumerate(zip(result["chunks"], result["scores"])):
        print(f"  [{i+1}] Score={score:.3f} | {chunk[:80]}…")