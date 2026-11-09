#!/usr/bin/env python3
"""
Compare RawModelQA vs StandardRAG on textbook questions about ACID, transactions, isolation.
Questions extracted from textbook.pdf
"""

import json
import sys
from pathlib import Path
from systems.raw_model import RawModelQA
from systems.standard_rag import StandardRAG

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Test questions about database transactions and ACID
QUESTIONS = [
    "Describe how isolation is enforced when two transactions interleave operations",
    "Describe how transactions that failed in the middle are being handled.",
    "Describe how interleave operations work with an example?",
    "Describe how interleave transactions work with an example?",
    "How does a database enforce ACID properties?",
    "Difference between ACID and BASE properties."
]

def run_comparison():
    """Run both systems on all questions and compare."""
    
    print("=" * 80)
    print("TEXTBOOK QA COMPARISON: RawModelQA vs StandardRAG")
    print("=" * 80)
    print(f"\nDataset: textbook.pdf (Database Transactions & ACID)")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"\nInitializing systems...\n")
    
    # Initialize systems
    try:
        raw_model = RawModelQA()
        print("✓ RawModelQA initialized")
    except Exception as e:
        print(f"✗ RawModelQA failed: {e}")
        return
    
    try:
        rag = StandardRAG()
        print("✓ StandardRAG initialized (loaded textbook.pdf)")
    except Exception as e:
        print(f"✗ StandardRAG failed: {e}")
        return
    
    # Store results
    results = []
    
    # Run each question
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}/{len(QUESTIONS)}")
        print(f"{'=' * 80}")
        print(f"Q: {question}\n")
        
        # RawModelQA (no retrieval)
        print("RawModelQA (No Retrieval):")
        print("-" * 40)
        try:
            raw_answer = raw_model.answer(question)
            print(f"Answer: {raw_answer.answer[:200]}...")
            print(f"Latency: {raw_answer.latency_seconds:.2f}s")
            print(f"Evidence: None (no retrieval)")
        except Exception as e:
            print(f"Error: {e}")
            raw_answer = None
        
        # StandardRAG (with retrieval)
        print("\nStandardRAG (BM25 Retrieval from textbook.pdf):")
        print("-" * 40)
        try:
            rag_answer = rag.answer(question)
            print(f"Answer: {rag_answer.answer[:200]}...")
            print(f"Latency: {rag_answer.latency_seconds:.2f}s")
            print(f"Evidence chunks retrieved: {len(rag_answer.evidence)}")
            if rag_answer.evidence:
                print(f"First chunk: {rag_answer.evidence[0][:100]}...")
        except Exception as e:
            print(f"Error: {e}")
            rag_answer = None
        
        # Compare
        if raw_answer and rag_answer:
            print("\nComparison:")
            print("-" * 40)
            latency_diff = rag_answer.latency_seconds - raw_answer.latency_seconds
            print(f"Latency difference: {latency_diff:+.2f}s (RAG took {latency_diff:.2f}s {'more' if latency_diff > 0 else 'less'})")
            print(f"Raw model retrieved evidence: {len(raw_answer.evidence) if raw_answer.evidence else 0}")
            print(f"RAG model retrieved evidence: {len(rag_answer.evidence) if rag_answer.evidence else 0}")
            
            # Store for output
            results.append({
                "question_id": i,
                "question": question,
                "raw_model": {
                    "answer": raw_answer.answer,
                    "latency_seconds": raw_answer.latency_seconds,
                    "evidence_count": len(raw_answer.evidence) if raw_answer.evidence else 0,
                    "metadata": raw_answer.metadata
                },
                "standard_rag": {
                    "answer": rag_answer.answer,
                    "latency_seconds": rag_answer.latency_seconds,
                    "evidence_count": len(rag_answer.evidence) if rag_answer.evidence else 0,
                    "evidence_samples": [e[:200] for e in rag_answer.evidence[:2]] if rag_answer.evidence else [],
                    "metadata": rag_answer.metadata
                }
            })
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}\n")
    
    if results:
        raw_latencies = [r["raw_model"]["latency_seconds"] for r in results]
        rag_latencies = [r["standard_rag"]["latency_seconds"] for r in results]
        
        print(f"RawModelQA:")
        print(f"  - Avg Latency: {sum(raw_latencies) / len(raw_latencies):.2f}s")
        print(f"  - Min: {min(raw_latencies):.2f}s, Max: {max(raw_latencies):.2f}s")
        
        print(f"\nStandardRAG:")
        print(f"  - Avg Latency: {sum(rag_latencies) / len(rag_latencies):.2f}s")
        print(f"  - Min: {min(rag_latencies):.2f}s, Max: {max(rag_latencies):.2f}s")
        print(f"  - Avg retrieval overhead: {(sum(rag_latencies) - sum(raw_latencies)) / len(rag_latencies):.2f}s")
        
        # Save results
        output_file = Path(__file__).parent / "results" / "textbook_comparison.jsonl"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    run_comparison()
