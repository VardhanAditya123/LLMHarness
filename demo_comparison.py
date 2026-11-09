"""
Comparison Demo: Raw Model vs Standard RAG
Tests both systems on the same questions
"""

from systems.raw_model import RawModelQA
from systems.standard_rag import StandardRAG
import json
from pathlib import Path


def main():
    print("\n" + "="*70)
    print("PHASE 3: RAG BASELINE COMPARISON TEST")
    print("="*70)
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "Explain the concept of neural networks.",
        "How do database indices improve performance?"
    ]
    
    # Initialize systems
    print("\n[1/4] Initializing systems...")
    try:
        raw_model = RawModelQA()
        print("✓ Raw Model initialized")
    except ConnectionError as e:
        print(f"✗ Raw Model failed: {e}")
        return
    
    try:
        rag = StandardRAG()
        print("✓ Standard RAG initialized")
    except ConnectionError as e:
        print(f"✗ Standard RAG failed: {e}")
        return
    
    # Test questions
    print("\n[2/4] System Configurations:")
    print(f"  Raw Model: {raw_model.get_config()['model_name']}")
    print(f"  Standard RAG: {rag.get_config()['model_name']} + BM25 retrieval")
    
    print("\n[3/4] Testing on sample questions...\n")
    
    results = {
        "raw_model": [],
        "standard_rag": []
    }
    
    # Run questions through both systems
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}/3: {question}")
        print("-" * 70)
        
        # Raw Model
        print("Raw Model:")
        try:
            raw_result = raw_model.answer(question)
            print(f"  Answer: {raw_result.answer[:100]}...")
            print(f"  Latency: {raw_result.latency_seconds:.2f}s")
            results["raw_model"].append(raw_result)
        except Exception as e:
            print(f"  Error: {e}")
        
        # Standard RAG
        print("\nStandard RAG:")
        try:
            rag_result = rag.answer(question)
            print(f"  Answer: {rag_result.answer[:100]}...")
            print(f"  Evidence: {len(rag_result.evidence)} passages")
            print(f"  Latency: {rag_result.latency_seconds:.2f}s")
            results["standard_rag"].append(rag_result)
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    # Summary
    print("\n[4/4] Summary Statistics:")
    print("-" * 70)
    
    if results["raw_model"]:
        raw_latencies = [r.latency_seconds for r in results["raw_model"]]
        print(f"\nRaw Model:")
        print(f"  Total latency: {sum(raw_latencies):.2f}s")
        print(f"  Average latency: {sum(raw_latencies)/len(raw_latencies):.2f}s")
        print(f"  Successful responses: {len(results['raw_model'])}/{len(test_questions)}")
    
    if results["standard_rag"]:
        rag_latencies = [r.latency_seconds for r in results["standard_rag"]]
        print(f"\nStandard RAG:")
        print(f"  Total latency: {sum(rag_latencies):.2f}s")
        print(f"  Average latency: {sum(rag_latencies)/len(rag_latencies):.2f}s")
        print(f"  Successful responses: {len(results['standard_rag'])}/{len(test_questions)}")
        print(f"  Evidence retrieval: BM25 (top-3)")
    
    # Save results
    results_path = Path("results/comparison_demo.jsonl")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, "w") as f:
        for system, sys_results in results.items():
            for result in sys_results:
                f.write(json.dumps({
                    "system": system,
                    "question": result.question,
                    "answer": result.answer[:200],  # Truncate for readability
                    "latency_seconds": result.latency_seconds,
                    "num_evidence": len(result.evidence),
                }) + "\n")
    
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*70)
    print("✓ PHASE 3 COMPARISON TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
