#!/usr/bin/env python3
"""
Compare retrieval modes on TokenSmith's BuzzDB benchmark.

Compares:
- Base: Raw model, no retrieval
- RAG: BM25-based retrieval on BuzzDB chunks
- LangchainEnsemble: Vector embeddings + BM25 hybrid (Langchain)
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
from systems.langchain_ensemble import LangchainEnsemble


# TokenSmith fixture paths
FIXTURE_PATH = TOKENSMITH_ROOT / "tests" / "fixtures" / "buzzdb" / "buzzdb-book.tokensmith.md"
BENCHMARK_DIR = TOKENSMITH_ROOT / "tests" / "benchmarks"

# All available BuzzDB test suites (matching TokenSmith's test_buzzdb_embeddings.py)
CASE_SUITES = [
    ("grounding", BENCHMARK_DIR / "buzzdb_grounding_cases.json"),
    ("reasoning", BENCHMARK_DIR / "buzzdb_reasoning_cases.json"),
    ("regression", BENCHMARK_DIR / "buzzdb_regression_cases.json"),
    ("cross_domain", BENCHMARK_DIR / "cross_domain_cases.json"),
]


def load_cases(max_cases=None):
    """Load BuzzDB test suites matching TokenSmith's test (27 total cases)"""
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
        return None, None, None, None
    
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
    
    return chunk_by_id, chunk_texts, chunk_ids, chunks


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


def run_langchain_ensemble_mode(cases, chunks_data, chunk_texts, chunk_ids):
    """Compare Langchain Ensemble mode (Vector embeddings + BM25 hybrid)"""
    print("\n" + "="*80)
    print("LANGCHAIN ENSEMBLE MODE: Vector embeddings + BM25 hybrid")
    print("="*80 + "\n")
    
    try:
        # Initialize ensemble retriever
        ensemble = LangchainEnsemble(
            chunks_data=chunks_data,
            chunk_texts=chunk_texts,
            chunk_ids=chunk_ids,
            embedding_model="nomic-embed-text:latest",
            top_k=8,
            bm25_weight=0.5,
            vector_weight=0.5
        )
    except Exception as e:
        print(f"⚠ Failed to initialize Langchain Ensemble: {e}")
        return {
            "mode": "langchain_ensemble",
            "passed": 0,
            "total": len(cases),
            "pass_rate": "0.0%",
            "cases": [],
            "error": str(e)
        }
    
    results = {
        "mode": "langchain_ensemble",
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
            # Get answer from ensemble (which returns retrieved chunks)
            qa_answer = ensemble.answer(question)
            
            # Extract retrieved chunk IDs from metadata
            retrieved_ids = qa_answer.metadata.get("retrieved_chunk_ids", [])
            
            # Evaluate
            matched = [cid for cid in expected_ids if cid in retrieved_ids]
            passed = len(matched) >= min_expected
            
            if passed:
                results["passed"] += 1
            
            results["cases"].append({
                "case_id": case["id"],
                "question": question,
                "passed": passed,
                "matched": len(matched),
                "expected": min_expected,
                "expected_ids": expected_ids,
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


def normalize_text(value):
    """Normalize text for comparison (from TokenSmith test)"""
    text = str(value).casefold().replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def run_tokensmith_mode():
    """Run TokenSmith's official vector-based hybrid retrieval test via unittest"""
    print("\n" + "="*80)
    print("TOKENSMITH MODE: Vector-based hybrid retrieval (FAISS 1.15.0 + keyword fallback)")
    print("="*80 + "\n")
    
    print("Running: TOKENSMITH_RUN_EMBEDDING_BENCHMARK=1 python tests/benchmarks/test_buzzdb_embeddings.py --json\n")
    
    # Build env with TOKENSMITH_RUN_EMBEDDING_BENCHMARK=1
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TOKENSMITH_ROOT)
    env["TOKENSMITH_RUN_EMBEDDING_BENCHMARK"] = "1"
    
    result = subprocess.run(
        "python tests/benchmarks/test_buzzdb_embeddings.py --json",
        cwd=str(TOKENSMITH_ROOT),
        capture_output=True,
        text=True,
        env=env,
        shell=True,
        timeout=300  # 5 min timeout for embedding generation
    )
    
    # Default result structure
    results = {
        "mode": "tokensmith",
        "passed": 0,
        "total": 0,
        "pass_rate": "0%",
        "cases": []
    }
    
    # Combine stdout and stderr
    output = result.stdout.strip()
    if result.stderr:
        output += "\n" + result.stderr.strip()
    
    if not output:
        results["error"] = f"Test failed with exit code {result.returncode}. No output."
        print(f"✗ Error: {results['error']}")
        return results
    
    try:
        lines = output.split('\n')
        
        # Look for JSON output (--json flag outputs JSON at the end)
        json_line = None
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{') and '"passed"' in line:
                json_line = line
                break
        
        if json_line:
            # Parse the JSON report from TokenSmith test
            report = json.loads(json_line)
            results["passed"] = report.get("passed", 0)
            results["total"] = report.get("total", 0)
            
            if results["total"] > 0:
                results["pass_rate"] = f"{100 * results['passed'] / results['total']:.1f}%"
            
            # Load grounding cases to extract questions
            grounding_cases_path = BENCHMARK_DIR / "buzzdb_grounding_cases.json"
            cross_domain_cases_path = BENCHMARK_DIR / "buzzdb_reasoning_variants.json"
            
            case_by_id = {}
            if grounding_cases_path.exists():
                for case in json.loads(grounding_cases_path.read_text(encoding="utf-8")):
                    case_by_id[case["id"]] = case
            if cross_domain_cases_path.exists():
                for case in json.loads(cross_domain_cases_path.read_text(encoding="utf-8")):
                    case_by_id[case["id"]] = case
            
            # Extract case details with questions and evidence
            for case in report.get("cases", []):
                case_id = case.get("id")
                lookup = case_by_id.get(case_id, {})
                
                # Extract evidence (chunks retrieved)
                evidence = case.get("evidence", [])
                chunk_summary = []
                for ev in evidence:
                    chunk_ids = ev.get("chunkIds", [])
                    section = ev.get("section", "")
                    if chunk_ids:
                        chunk_summary.append({
                            "chunk_ids": chunk_ids,
                            "section": section[:100] if section else "(no section)"
                        })
                
                results["cases"].append({
                    "case_id": case_id,
                    "question": lookup.get("question", lookup.get("prompt", case_id)),
                    "passed": case.get("passed", False),
                    "failures": case.get("failures", []),
                    "retrieved_chunks": chunk_summary
                })
            
            status = "✓" if results["passed"] == results["total"] else "⚠"
            print(f"{status} Ollama Nomic hybrid benchmark: {results['passed']}/{results['total']} questions")
            print(f"Passed: {results['passed']}/{results['total']} ({results['pass_rate']})\n")
            
            # Display sample questions and retrieved chunks (first 3 cases)
            print("Sample cases (questions & retrieved chunks):\n")
            for i, case in enumerate(results["cases"][:3]):
                print(f"  [{i+1}] {case['case_id']}")
                print(f"      Q: {case['question'][:80]}{'...' if len(case['question']) > 80 else ''}")
                if case['retrieved_chunks']:
                    for chunk_info in case['retrieved_chunks'][:2]:  # Show first 2 chunk retrieval groups
                        chunk_ids_str = ", ".join(chunk_info['chunk_ids'][:3])  # First 3 IDs
                        if len(chunk_info['chunk_ids']) > 3:
                            chunk_ids_str += f", +{len(chunk_info['chunk_ids'])-3} more"
                        print(f"      Chunks: {chunk_ids_str}")
                        if chunk_info['section']:
                            print(f"      Section: {chunk_info['section']}")
                else:
                    print(f"      Chunks: (keyword fallback, no vectors)")
                print()
            
            if len(results["cases"]) > 3:
                print(f"  ... and {len(results['cases']) - 3} more cases\n")
            
            # Show first 3 failures if any
            if results["passed"] < results["total"]:
                print("Failed cases:")
                fail_count = 0
                for case in results["cases"]:
                    if not case.get("passed") and fail_count < 3:
                        failures_str = "; ".join(case.get("failures", ["unknown"])[:1])
                        print(f"  ✗ {case.get('case_id', 'unknown'):40} {failures_str}")
                        fail_count += 1
                if len([c for c in results["cases"] if not c.get("passed")]) > 3:
                    print(f"  ... and {len([c for c in results['cases'] if not c.get('passed')]) - 3} more")
                print()
            
            return results
        
        # Fallback: look for TokenSmith's benchmark line
        benchmark_line = None
        for line in lines:
            if "Ollama Nomic hybrid benchmark:" in line:
                benchmark_line = line
                break
        
        if benchmark_line:
            import re
            match = re.search(r'(\d+)/(\d+)\s+questions', benchmark_line)
            if match:
                results["passed"] = int(match.group(1))
                results["total"] = int(match.group(2))
                results["pass_rate"] = f"{100 * results['passed'] / results['total']:.1f}%" if results["total"] > 0 else "0%"
                
                status = "✓" if results["passed"] == results["total"] else "⚠"
                print(f"{status} Ollama Nomic hybrid benchmark: {results['passed']}/{results['total']} questions")
                print(f"\nPassed: {results['passed']}/{results['total']} ({results['pass_rate']})\n")
                return results
        
        # No JSON and no benchmark line found
        results["error"] = f"Could not parse test output. Exit code: {result.returncode}"
        print(f"✗ Error: {results['error']}")
        print("Output:")
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
        
    except json.JSONDecodeError as e:
        results["error"] = f"JSON parse error: {e}"
        print(f"✗ Error: {results['error']}")
    except Exception as e:
        results["error"] = f"Unexpected error: {str(e)}"
        print(f"✗ Error: {results['error']}")
        import traceback
        traceback.print_exc()
    
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
    chunk_by_id, chunk_texts, chunk_ids, chunks_data = load_buzzdb_chunks()
    if chunk_by_id is None:
        print("Failed to load chunks")
        sys.exit(1)
    
    # Run comparisons
    base_results = run_base_mode(cases, chunk_by_id)
    rag_results = run_rag_mode(cases, chunk_by_id, chunk_texts, chunk_ids)
    langchain_results = run_langchain_ensemble_mode(cases, chunks_data, chunk_texts, chunk_ids)
    tokensmith_results = run_tokensmith_mode()
    
    # Summary with clean formatting and analysis
    print("\n" + "="*80)
    print("SUMMARY: 4-MODE COMPARISON (27 Cases - Matching TokenSmith's test_buzzdb_embeddings.py)")
    print("="*80 + "\n")
    
    # Build summary table with proper alignment
    modes = [
        ("BASE", base_results),
        ("RAG", rag_results),
        ("LANGCHAIN_ENSEMBLE", langchain_results),
        ("TOKENSMITH", tokensmith_results)
    ]
    
    # Print header
    print(f"  {'Mode':<20} | {'Passed':<8} | {'Pass Rate':<10} | {'Cases':<8}")
    print("  " + "-"*58)
    
    # Print rows
    for mode_name, mode_data in modes:
        passed = mode_data['passed']
        total = mode_data['total']
        rate = mode_data['pass_rate']
        case_label = "27 (official)" if mode_name == "TOKENSMITH" else f"{total} (full suite)"
        print(f"  {mode_name:<20} | {passed:>2}/{total:<2} | {rate:>8} | {case_label:<8}")
    
    print("\n" + "="*80)
    
    # Rudimentary analysis
    print("\n📊 ANALYSIS:\n")
    
    le_passed = langchain_results['passed']
    rag_passed = rag_results['passed']
    base_passed = base_results['passed']
    ts_passed = tokensmith_results['passed']
    
    # Improvement over base
    print(f"  • LangchainEnsemble vs Base:")
    print(f"    └─ +{le_passed - base_passed} cases ({le_passed/base_results['total']*100:.1f}% pass rate)")
    
    # Improvement over RAG
    if le_passed > rag_passed:
        improvement_pct = (le_passed - rag_passed) / rag_passed * 100
        print(f"\n  • LangchainEnsemble vs RAG:")
        print(f"    └─ +{le_passed - rag_passed} cases (+{improvement_pct:.1f}% improvement)")
    
    # Gap to TokenSmith
    if ts_passed > 0:
        gap_pct = (ts_passed - le_passed) / ts_passed * 100
        print(f"\n  • LangchainEnsemble vs TokenSmith:")
        print(f"    └─ {le_passed}/27 vs {ts_passed}/27 (same test suite: grounding+reasoning+regression+cross_domain)")
        if gap_pct > 0:
            print(f"    └─ Gap: {gap_pct:.1f}% below TokenSmith official")
        else:
            print(f"    └─ Matches TokenSmith performance!")
    
    print(f"\n  💡 Key Insight: Hybrid (vector+BM25) beats keyword-only by {le_passed-rag_passed} cases")
    print("="*80 + "\n")
    
    # Save results
    results_file = Path(__file__).parent / "results" / "buzzdb_comparison.json"
    results_file.parent.mkdir(exist_ok=True)
    results_file.write_text(json.dumps({
        "base": base_results,
        "rag": rag_results,
        "langchain_ensemble": langchain_results,
        "tokensmith": tokensmith_results
    }, indent=2))
    print(f"✓ Results saved to {results_file}\n")


if __name__ == "__main__":
    main()
