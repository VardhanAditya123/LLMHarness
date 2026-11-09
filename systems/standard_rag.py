"""StandardRAG: Load PDFs from data/, retrieve with BM25, generate with Ollama."""

import requests
import yaml
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi

from systems.base import QASystem, QAAnswer, Timer


class StandardRAG(QASystem):
    """RAG baseline: BM25 retrieval from PDFs + Ollama generation."""
    
    def __init__(self, config_path: str = "configs/model.yaml", data_dir: str = "data"):
        """Initialize StandardRAG.
        
        Args:
            config_path: Path to model configuration (YAML)
            data_dir: Directory containing PDF files
        """
        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.model_config = config.get("ollama_model")
        self.model_name = self.model_config["model_name"]
        self.ollama_host = self.model_config.get("ollama_host", "http://localhost:11434")
        self.temperature = self.model_config.get("temperature", 0.7)
        self.top_p = self.model_config.get("top_p", 0.9)
        
        # Verify Ollama connection
        print(f"Connecting to Ollama at {self.ollama_host}")
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            tags = response.json().get("models", [])
            available_models = [m.get("name", "").split(":")[0] for m in tags]
            
            if self.model_name not in available_models:
                raise ConnectionError(f"Model '{self.model_name}' not found")
            
            # Get full model name
            self.actual_model_name = None
            for model in tags:
                if model.get("name", "").split(":")[0] == self.model_name:
                    self.actual_model_name = model.get("name")
                    break
            
            if not self.actual_model_name:
                self.actual_model_name = f"{self.model_name}:latest"
            
            print(f"✓ Ollama connected. Model '{self.actual_model_name}' available")
        except Exception as e:
            raise ConnectionError(f"Ollama connection failed: {e}") from e
        
        # Load and chunk PDFs
        print(f"Loading PDFs from {data_dir}/...")
        self.chunks = self._load_pdfs_and_chunk(data_dir)
        print(f"✓ Loaded {len(self.chunks)} text chunks")
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✓ BM25 ready\n")
    
    def _load_pdfs_and_chunk(self, data_dir: str) -> List[str]:
        """Load all PDFs and chunk them into passages."""
        try:
            import fitz  # pymupdf
        except ImportError:
            print("⚠ pymupdf not found. Install with: pip install pymupdf")
            return self._placeholder_chunks()
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠ {data_dir} not found")
            return self._placeholder_chunks()
        
        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠ No PDFs in {data_dir}")
            return self._placeholder_chunks()
        
        print(f"Found {len(pdf_files)} PDF(s)")
        
        all_text = []
        for pdf_file in pdf_files:
            try:
                print(f"  Reading {pdf_file.name}...")
                doc = fitz.open(str(pdf_file))
                total_pages = len(doc)
                
                for page_num, page in enumerate(doc):
                    if (page_num + 1) % 10 == 0:
                        print(f"    Page {page_num + 1}/{total_pages}")
                    
                    text = page.get_text()
                    if text and len(text) > 100:
                        # Clean text
                        text = text.replace("\x00", "").replace("\n", " ")
                        text = " ".join(text.split())  # Normalize whitespace
                        all_text.append(text)
                
                doc.close()
                print(f"  ✓ Loaded {total_pages} pages")
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
        
        # Chunk into passages (~1000 chars)
        chunks = []
        chunk_size = 1000
        
        for text in all_text:
            # Simple chunking: split by sentences
            sentences = text.replace(". ", ".\n").split("\n")
            current = ""
            
            for sent in sentences:
                if len(current) + len(sent) < chunk_size:
                    current += " " + sent if current else sent
                else:
                    if len(current) > 100:
                        chunks.append(current)
                    current = sent
            
            if len(current) > 100:
                chunks.append(current)
        
        return chunks if chunks else self._placeholder_chunks()
    
    def _placeholder_chunks(self) -> List[str]:
        """Return placeholder chunks for testing."""
        return [
            "Machine learning is a subset of artificial intelligence that learns from data.",
            "Neural networks are systems inspired by biological brains with layers of neurons.",
            "Database indices improve the speed of data retrieval operations.",
            "Supervised learning uses labeled data to train predictive models.",
            "Unsupervised learning finds patterns in unlabeled data.",
        ]
    
    def answer(self, question: str) -> QAAnswer:
        """Answer a question using BM25 retrieval + Ollama generation.
        
        Args:
            question: The question to answer
            
        Returns:
            QAAnswer with evidence and generated response
        """
        with Timer() as timer:
            try:
                # Retrieve top 3 chunks
                query_tokens = question.lower().split()
                scores = self.bm25.get_scores(query_tokens)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
                evidence = [self.chunks[i] for i in top_idx]
                
                # Build prompt
                context = "\n\n".join(evidence)
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                
                # Generate with Ollama
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.actual_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    answer_text = response.json().get("response", "").strip()
                    if prompt in answer_text:
                        answer_text = answer_text.replace(prompt, "").strip()
                else:
                    answer_text = f"[Error {response.status_code}]"
            
            except Exception as e:
                answer_text = f"[Error: {str(e)}]"
                evidence = []
        
        return QAAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            abstained=False,
            latency_seconds=timer.elapsed,
            metadata={
                "model": self.actual_model_name,
                "retrieval": "bm25",
                "num_sources": len(evidence),
            }
        )
    
    def get_config(self) -> dict:
        """Return configuration."""
        return {
            "system": "standard_rag",
            "model": self.actual_model_name,
            "retrieval": "bm25",
            "num_chunks": len(self.chunks),
        }
