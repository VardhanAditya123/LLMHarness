"""Extract questions from textbook.pdf and compare RawModel vs StandardRAG"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from systems.raw_model import RawModelQA
from systems.standard_rag import StandardRAG

def extract_text_from_pdf(pdf_path, num_pages=5):
    """Extract text from first N pages of PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(min(num_pages, len(doc))):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def find_questions_in_text(text, num_questions=5):
    """Find sentences that look like questions"""
    lines = text.split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        # Look for lines ending with ?
        if line.endswith('?') and len(line) > 10:
            questions.append(line)
        
        if len(questions) >= num_questions:
            break
    
    return questions[:num_questions]

def run_comparison(questions):
    """Run RawModelQA and StandardRAG on the same questions"""
    print("\n" + "="*80)
    print("FINE-TUNING BASELINE: RAW MODEL vs STANDARD RAG")
    print("="*80)
    
    # Initialize systems
    print("\n[1/3] Initializing systems...")
    raw_qa = RawModelQA()
    print("  ✓ RawModelQA initialized")
    
    standard_rag = StandardRAG()
    print("  ✓ StandardRAG initialized")
    
    results = {
        "questions": questions,
        "raw_model": [],
        "standard_rag": [],
        "comparison": []
    }
    
    print(f"\n[2/3] Running {len(questions)} questions through both systems...\n")
    
    for idx, question in enumerate(questions, 1):
        print(f"Question {idx}/{len(questions)}: {question[:60]}...")
        
        # Get Raw Model answer
        raw_result = raw_qa.answer(question)
        results["raw_model"].append(raw_result.to_dict())
        
        # Get StandardRAG answer
        rag_result = standard_rag.answer(question)
        results["standard_rag"].append(rag_result.to_dict())
        
        # Store comparison
        comparison = {
            "question": question,
            "raw_model": {
                "answer": raw_result.answer[:100] + "..." if len(raw_result.answer) > 100 else raw_result.answer,
                "latency": raw_result.latency_seconds,
                "has_evidence": len(raw_result.evidence) > 0
            },
            "standard_rag": {
                "answer": rag_result.answer[:100] + "..." if len(rag_result.answer) > 100 else rag_result.answer,
                "latency": rag_result.latency_seconds,
                "num_evidence": len(rag_result.evidence),
                "evidence_sample": rag_result.evidence[0][:80] + "..." if rag_result.evidence else "N/A"
            }
        }
        results["comparison"].append(comparison)
        
        print(f"  Raw Model:   {raw_result.latency_seconds:.2f}s | Evidence: No")
        print(f"  StandardRAG: {rag_result.latency_seconds:.2f}s | Evidence: {len(rag_result.evidence)} passages")
        print()
    
    print("[3/3] Generating comparison table...\n")
    
    # Generate comparison table
    print("="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Q#':<3} | {'Question':<30} | {'Raw (s)':<8} | {'RAG (s)':<8} | {'Speedup':<8}")
    print("-" * 80)
    
    for idx, comp in enumerate(results["comparison"], 1):
        q_text = comp["question"][:27] + "..." if len(comp["question"]) > 30 else comp["question"]
        raw_lat = comp["raw_model"]["latency"]
        rag_lat = comp["standard_rag"]["latency"]
        speedup = "N/A" if raw_lat == 0 else f"{raw_lat/rag_lat:.2f}x"
        
        print(f"{idx:<3} | {q_text:<30} | {raw_lat:<8.2f} | {rag_lat:<8.2f} | {speedup:<8}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_raw = sum(c["raw_model"]["latency"] for c in results["comparison"]) / len(results["comparison"])
    avg_rag = sum(c["standard_rag"]["latency"] for c in results["comparison"]) / len(results["comparison"])
    
    print(f"Average latency (Raw Model):  {avg_raw:.2f}s")
    print(f"Average latency (StandardRAG): {avg_rag:.2f}s")
    print(f"Slowdown ratio: {avg_rag/avg_raw:.2f}x (retrieval cost)")
    
    # Save results
    results_file = Path("results/raw_vs_rag_comparison.jsonl")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        for comp in results["comparison"]:
            f.write(json.dumps(comp) + '\n')
    
    print(f"\n✓ Results saved to {results_file}")
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    pdf_path = Path("data/textbook.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please ensure textbook.pdf is in data/")
        exit(1)
    
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path, num_pages=5)
    
    print("Finding questions in PDF...")
    questions = find_questions_in_text(text, num_questions=5)
    
    if not questions:
        print("❌ Could not find questions in PDF")
        print("\nUsing sample questions instead...\n")
        questions = [
            "What is the definition of machine learning?",
            "How do neural networks work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain the concept of backpropagation",
            "What are the main challenges in deep learning?"
        ]
    else:
        print(f"✓ Found {len(questions)} questions in PDF:\n")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}\n")
    
    # Run comparison
    results = run_comparison(questions)
