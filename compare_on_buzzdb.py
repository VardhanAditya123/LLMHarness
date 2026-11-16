#!/usr/bin/env python3
"""
Compare retrieval modes on TokenSmith's BuzzDB benchmark.

Compares:
- Base: Raw model, no retrieval
- RAG: BM25-based retrieval on BuzzDB chunks
- TokenSmith: Official TokenSmith benchmark (npm runner with all suites)
"""

import json
import sys
import re
import subprocess
import tempfile
import os
from pathlib import Path

# Add TokenSmith to path
TOKENSMITH_ROOT = Path(__file__).parent.parent / "TokenSmith"
if str(TOKENSMITH_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENSMITH_ROOT))

# ignore thee 2 import warnings
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store

# Add systems to path
sys.path.insert(0, str(Path(__file__).parent))
from systems.raw_model import RawModelQA
from systems.standard_rag import StandardRAG


# TokenSmith fixture paths
FIXTURE_PATH = TOKENSMITH_ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
BENCHMARK_DIR = TOKENSMITH_ROOT / "tests" / "benchmarks"

# All available BuzzDB test suites
CASE_SUITES = [
    ("grounding", BENCHMARK_DIR / "buzzdb_grounding_cases.json"),
    ("multiturn", BENCHMARK_DIR / "buzzdb_multiturn_cases.json"),
    ("reasoning", BENCHMARK_DIR / "buzzdb_reasoning_cases.json"),
    ("reasoning_variants", BENCHMARK_DIR / "buzzdb_reasoning_variants.json"),
    ("regression", BENCHMARK_DIR / "buzzdb_regression_cases.json"),
]


def load_cases(max_cases=None):
    """Load all BuzzDB test suites (34 total cases)"""
    all_cases = []
    
    for suite_name, case_path in CASE_SUITES:
        if not case_path.exists():
            print(f"Warning: {case_path} not found")
            continue
        
        suite_cases = json.loads(case_path.read_text(encoding="utf-8"))
        
        # Tag each case with its suite for tracking
        for case in suite_cases:
            case["suite"] = suite_name
        
        all_cases.extend(suite_cases)
        print(f"  Loaded {len(suite_cases)} cases from {suite_name}")
    
    if max_cases:
        all_cases = all_cases[:max_cases]
    
    return all_cases


def evaluate_retrieval(chunk_by_id, chunk_id_by_rowid, case, retrieved_chunk_ids):
    """
    Evaluate whether retrieved chunks match expected chunks for a case.
    
    Returns: (passed: bool, matched_count: int, expected_count: int)
    """
    expected = case.get("expectedChunkIds", [])
    min_expected = int(case.get("minExpectedHits", len(expected)))
    
    matched = [chunk_id for chunk_id in expected if chunk_id in retrieved_chunk_ids]
    passed = len(matched) >= min_expected
    
    return passed, len(matched), min_expected


def get_query_text(case):
    """Extract query text from case (handles different formats)"""
    # Try different field names in priority order
    return (
        case.get("retrievalQuery") or 
        case.get("prompt") or 
        case.get("priorQuestion") or
        case.get("question") or 
        f"Case {case.get('id', 'unknown')}"
    )


def get_expected_chunks(case):
    """Extract expected chunk IDs from case (handles different formats)"""
    return case.get("expectedChunkIds", [])


def run_base_mode(cases, chunk_by_id=None):
    """Compare Base mode (no retrieval) - counts expected chunks as 0"""
    print("\n" + "="*80)
    print("BASE MODE: Raw model (no retrieval)")
    print("="*80 + "\n")
    
    qa = RawModelQA()
    results = {
        "mode": "base",
        "passed": 0,
        "total": len(cases),
        "cases": []
    }
    
    for case in cases:
        question = get_query_text(case)
        expected_ids = get_expected_chunks(case)
        expected_count = len(expected_ids)
        
        # Base mode never retrieves anything, so it always fails
        expected_content = []
        if chunk_by_id:
            for eid in expected_ids:
                if eid in chunk_by_id:
                    expected_content.append({
                        "chunk_id": eid,
                        "text": chunk_by_id[eid].get("text", "")[:200]
                    })
        
        results["cases"].append({
            "case_id": case["id"],
            "suite": case.get("suite", "unknown"),
            "question": question,
            "passed": False,
            "matched": 0,
            "expected": expected_count,
            "expected_ids": expected_ids,
            "expected_content": expected_content,
            "reason": "No retrieval in base mode"
        })
    
    results["pass_rate"] = f"{100 * results['passed'] / len(cases):.1f}%"
    
    print(f"Passed: {results['passed']}/{results['total']} ({results['pass_rate']})")
    print("Base mode retrieves 0 chunks by design.\n")
    
    return results


def load_buzzdb_chunks():
    """Load and parse BuzzDB fixture chunks"""
    print("Loading BuzzDB fixture...")
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    chunks = engine.chunk_tokensmith_markdown(fixture_text)
    
    if not chunks:
        print("Error: Failed to parse BuzzDB chunks")
        return None, None, None
    
    print(f"Loaded {len(chunks)} chunks from BuzzDB")
    
    # Build chunk mapping
    chunk_by_id = {}
    chunk_texts = []
    chunk_ids = []
    
    for chunk in chunks:
        chunk_id = chunk.get("tokensmithChunkId")
        if chunk_id:
            chunk_by_id[chunk_id] = chunk
            chunk_texts.append(chunk.get("text", ""))
            chunk_ids.append(chunk_id)
    
    print(f"Built index for {len(chunk_by_id)} chunks\n")
    
    return chunk_by_id, chunk_texts, chunk_ids


def run_rag_mode(cases, chunk_by_id, chunk_texts, chunk_ids):
    """Compare RAG mode (BM25 retrieval with grounding on BuzzDB chunks)"""
    print("\n" + "="*80)
    print("RAG MODE: BM25 retrieval on BuzzDB chunks")
    print("="*80 + "\n")
    
    print("Building BM25 index...")
    
    # Build BM25 index
    from rank_bm25 import BM25Okapi
    
    # Tokenize chunks into words
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    
    print("BM25 index ready\n")
    
    results = {
        "mode": "rag",
        "passed": 0,
        "total": len(cases),
        "cases": []
    }
    
    # Retrieve chunks for each case
    for i, case in enumerate(cases):
        question = get_query_text(case)
        expected_ids = get_expected_chunks(case)
        min_expected = int(case.get("minExpectedHits", len(expected_ids)))
        
        try:
            # Query BM25
            query_tokens = question.lower().split()
            scores = bm25.get_scores(query_tokens)
            
            # Get top-8 chunks
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
            retrieved_ids = [chunk_ids[idx] for idx in top_indices]
            
            # Evaluate
            matched = [cid for cid in expected_ids if cid in retrieved_ids]
            passed = len(matched) >= min_expected
            
            if passed:
                results["passed"] += 1
            
            # Build expected chunk content
            expected_content = []
            for eid in expected_ids:
                if eid in chunk_by_id:
                    expected_content.append({
                        "chunk_id": eid,
                        "text": chunk_by_id[eid].get("text", "")[:200]  # First 200 chars
                    })
            
            results["cases"].append({
                "case_id": case["id"],
                "question": question,
                "passed": passed,
                "matched": len(matched),
                "expected": min_expected,
                "expected_ids": expected_ids,
                "expected_content": expected_content,
                "retrieved_ids": retrieved_ids
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(cases)} cases")
        
        except Exception as e:
            results["cases"].append({
                "case_id": case["id"],
                "question": question,
                "passed": False,
                "matched": 0,
                "expected": min_expected,
                "error": str(e)
            })
    
    results["pass_rate"] = f"{100 * results['passed'] / len(cases):.1f}%"
    
    print(f"\nPassed: {results['passed']}/{results['total']} ({results['pass_rate']})\n")
    
    return results


def run_tokensmith_mode():
    """Run TokenSmith's actual BuzzDB test suite"""
    print("\n" + "="*80)
    print("TOKENSMITH MODE: BuzzDB retrieval test suite")
    print("="*80 + "\n")
    
    print("Running: python -m tests.benchmarks.test_buzzdb_grounding --json\n")
    
    result = subprocess.run(
        ["python", "-m", "tests.benchmarks.test_buzzdb_grounding", "--json"],
        cwd=str(TOKENSMITH_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(TOKENSMITH_ROOT)}
    )
    
    results = {
        "mode": "tokensmith",
        "passed": 0,
        "total": 0,
        "pass_rate": "0%",
        "cases": [],
    }
    
    # Parse JSON output (can be in stdout or stderr)
    output = result.stdout.strip() or result.stderr.strip()
    
    if not output:
        results["error"] = f"No output from test (exit code {result.returncode})"
        return results
    
    try:
        # Get last JSON object (in case of multiple lines)
        output_lines = [l for l in output.split('\n') if l.strip()]
        json_line = output_lines[-1]
        test_results = json.loads(json_line)
        
        results["passed"] = test_results.get("passed", 0)
        results["total"] = test_results.get("total", 0)
        
        if results["total"] > 0:
            results["pass_rate"] = f"{100 * results['passed'] / results['total']:.1f}%"
        
        # Store case details
        for case in test_results.get("cases", []):
            results["cases"].append({
                "case_id": case.get("id"),
                "passed": case.get("passed", False),
                "failures": case.get("failures", []),
                "hits": case.get("hits", [])
            })
        
    except json.JSONDecodeError as e:
        print(f"Error parsing test output: {e}")
        results["error"] = str(e)
        results["raw_output"] = output[:500]
    
    print(f"Test Results: {results['passed']}/{results['total']} ({results['pass_rate']})\n")
    
    return results


def main():
    """Compare modes on BuzzDB benchmark"""
    print("\n" + "="*80)
    print("RETRIEVAL MODE COMPARISON: BuzzDB Benchmark (All Suites)")
    print("="*80)
    
    # Verify benchmark directory exists
    if not BENCHMARK_DIR.exists():
        print(f"\nError: TokenSmith benchmarks not found at {BENCHMARK_DIR}")
        print("Expected: TokenSmith cloned at ../TokenSmith relative to this repo")
        sys.exit(1)
    
    # Load test cases from all suites
    print("\nLoading test cases from all BuzzDB suites...")
    cases = load_cases()
    print(f"\n✓ Loaded {len(cases)} total test cases")
    
    # Show suite breakdown
    suite_counts = {}
    for case in cases:
        suite = case.get("suite", "unknown")
        suite_counts[suite] = suite_counts.get(suite, 0) + 1
    print("  Suite breakdown:")
    for suite, count in sorted(suite_counts.items()):
        print(f"    - {suite}: {count} cases")
    
    # Load chunks once for all modes
    print()
    chunk_by_id, chunk_texts, chunk_ids = load_buzzdb_chunks()
    if chunk_by_id is None:
        print("Failed to load chunks")
        sys.exit(1)
    
    # Run comparisons
    base_results = run_base_mode(cases, chunk_by_id)
    rag_results = run_rag_mode(cases, chunk_by_id, chunk_texts, chunk_ids)
    tokensmith_results = run_tokensmith_mode()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: 3-MODE COMPARISON (34 Cases Across All Suites)")
    print("="*80 + "\n")
    
    print("| Mode | Passed | Total | Pass Rate |")
    print("|------|--------|-------|-----------|")
    print(f"| Base | {base_results['passed']} | {base_results['total']} | {base_results['pass_rate']} |")
    print(f"| RAG  | {rag_results['passed']} | {rag_results['total']} | {rag_results['pass_rate']} |")
    print(f"| TokenSmith | {tokensmith_results['passed']} | {tokensmith_results['total']} | {tokensmith_results['pass_rate']} |")
    print("")
    
    # Save results
    results_file = Path(__file__).parent / "results" / "buzzdb_comparison.json"
    results_file.parent.mkdir(exist_ok=True)
    results_file.write_text(json.dumps({
        "base": base_results,
        "rag": rag_results,
        "tokensmith": tokensmith_results
    }, indent=2))
    print(f"✓ Results saved to {results_file}\n")


if __name__ == "__main__":
    main()
