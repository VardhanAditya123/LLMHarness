# Retrieval Mode Comparison

Compare four retrieval strategies on TokenSmith's official BuzzDB benchmark.

- **Base** - Raw LLM, no retrieval
- **RAG** - BM25 keyword retrieval  
- **LangchainEnsemble** - Vector embeddings + BM25 hybrid (Langchain-based)
- **TokenSmith** - Official hybrid (keywords + embeddings)

---

## Quick Start

```bash
# Activate conda environment
conda activate tokensmith

# Install FAISS for vector search (if not already installed)
pip install faiss-cpu

# Run comparison
python compare_on_buzzdb.py
```

Results saved to: `results/buzzdb_comparison.json`

---

## Setup

**Prerequisites:**
- Python 3.12 via conda environment `tokensmith`
- Ollama running locally with gemma3:1b and nomic-embed-text models
- TokenSmith cloned at `../TokenSmith`

**First time:**
```bash
# Install TokenSmith Python runtime (embeddings, dependencies)
cd ../TokenSmith
npm run setup:python-runtime

# Install this project's dependencies
cd ../tokensmith-benchmark
pip install -r requirements.txt
```

**Ensure Ollama has required models:**
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

---

## What Gets Tested

Evaluates all four modes on **34 database systems questions** from BuzzDB:

- "Why is page pinning needed in a buffer manager?"
- "Why is B+ tree preferred over binary search tree?"
- "Why does hash index struggle with range queries?"
- (31 more similar questions)

**Results:**
- Base: 0/34 (0.0%)
- RAG: 24/34 (70.6%)
- LangchainEnsemble: 27/34 (79.4%) ✓ vector + BM25 hybrid
- TokenSmith: 17/21 (81.0%)

See [comparison.md](comparison.md) for detailed results and insights.

---

## How It Works

### 1. Base Mode (Baseline)
- Queries raw model with no retrieval
- Demonstrates hallucination without grounding
- Always 0% (baseline by design)

### 2. RAG Mode (Keyword Retrieval)
- Builds BM25 index from 1,574 BuzzDB chunks
- Retrieves top-8 chunks by keyword match
- Passes context to model
- 70.6% accuracy with simple, fast retrieval

### 3. LangchainEnsemble Mode (Hybrid Retrieval)
- Langchain-based hybrid retriever combining:
  - **Vector search**: Nomic embeddings (137M parameters)
  - **Keyword search**: BM25 (TF-IDF based)
  - **Ensemble**: Combined results with weighted scoring
- Retrieves top-8 chunks from ensemble
- ~75-80% accuracy, balanced performance
- Requires embedding model (nomic-embed-text)
- More computationally intensive than RAG

### 4. TokenSmith Mode (Official)
- Runs TokenSmith's official test suite
- Hybrid retrieval: BM25 + semantic embeddings
- 81.0% accuracy, best-in-class
- Note: Tests 21 of 34 cases (by official design)

---

## Project Structure

```
.
├── compare_on_buzzdb.py          # Main comparison script
├── comparison.md                 # Results & insights
│
├── systems/
│   ├── base.py                   # Base classes & Timer
│   ├── raw_model.py              # Base mode (no retrieval)
│   ├── standard_rag.py           # RAG mode (BM25)
│   └── langchain_ensemble.py     # LangchainEnsemble (vector + BM25 hybrid)
│
├── data/
│   └── buzzdb-book.tokensmith.md # 1.5MB fixture (1,574 chunks)
├── configs/                       # Model config
└── results/                       # Saved benchmark results
```

---

## Modify & Extend

### Change RAG parameters:
Edit `systems/standard_rag.py`:
- `top_k` - chunks to retrieve (default: 8)
- Chunk size/overlap in tokensmith parsing
- BM25 parameters (k1, b)

### Add a new mode:
1. Create `systems/my_mode.py` (inherit from QASystem)
2. Implement `query(question: str, context: str) → QAAnswer`
3. Add to `main()` in `compare_on_buzzdb.py`

### Test on different data:
1. Replace `data/buzzdb-book.tokensmith.md`
2. Update test case JSON files in `benchmarks/`
3. Run `python compare_on_buzzdb.py`

---

## Dependencies

```
python=3.12
langchain
langchain-community
rank-bm25
ollama
pyyaml
tokensmith (from ../TokenSmith)
```

See `requirements.txt` for full list.
