"""Demo script to test the Phase 2 harness with Ollama."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from systems.raw_model import RawModelQA


def main():
    """Run demo with test questions using Ollama."""
    
    print("\n" + "="*70)
    print("PHASE 2 HARNESS: RAW MODEL BASELINE TEST (Ollama)")
    print("="*70 + "\n")
    
    # Initialize model
    print("[1/4] Connecting to Ollama...")
    try:
        qa = RawModelQA(config_path="configs/model.yaml")
        print("✓ Ollama connection successful\n")
    except ConnectionError as e:
        print(f"✗ Failed to connect: {e}\n")
        print("SETUP REQUIRED:")
        print("  1. Install Ollama: https://ollama.ai/download")
        print("  2. Start Ollama server in a new terminal:")
        print("     ollama serve")
        print("  3. In another terminal, pull Gemma model:")
        print("     ollama pull gemma")
        print("  4. Then run this script again:")
        print("     python demo.py")
        print("\nFor detailed setup instructions, see OLLAMA_SETUP.md")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Print configuration
    print("[2/4] System Configuration:")
    config = qa.get_config()
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "Explain the concept of neural networks.",
        "What is a database index?"
    ]
    
    print(f"[3/4] Testing with {len(test_questions)} questions...\n")
    
    results = []
    total_latency = 0.0
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}/{len(test_questions)}: {question}")
        
        try:
            result = qa.answer(question)
            results.append(result)
            total_latency += result.latency_seconds
            
            # Print result
            print(f"Answer: {result.answer[:200]}...")
            print(f"Latency: {result.latency_seconds:.2f}s")
            print(f"Metadata: {json.dumps(result.metadata, indent=2)}")
            print("-" * 70)
            
        except Exception as e:
            print(f"✗ Error answering question: {e}")
            return 1
    
    # Summary statistics
    print(f"\n[4/4] Summary Statistics:")
    print(f"  - Total questions: {len(test_questions)}")
    print(f"  - Total latency: {total_latency:.2f}s")
    print(f"  - Average latency: {total_latency / len(test_questions):.2f}s")
    print(f"  - Successful responses: {len(results)}")
    
    print("\n" + "="*70)
    print("✓ PHASE 2 HARNESS VALIDATION COMPLETE")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
