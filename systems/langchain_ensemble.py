"""LangchainEnsemble: Vector embeddings + BM25 hybrid retrieval with Langchain."""

import requests
import yaml
from pathlib import Path
from typing import List, Any, Dict
import numpy as np

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from systems.base import QASystem, QAAnswer, Timer


class LangchainEnsemble(QASystem):
    """Hybrid retrieval: Langchain ensemble (vector + BM25) + Ollama generation."""
    
    def __init__(
        self,
        chunks_data: List[dict],
        chunk_texts: List[str],
        chunk_ids: List[str],
        config_path: str = "configs/model.yaml",
        embedding_model: str = "nomic-embed-text:latest",
        top_k: int = 8,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ):
        """Initialize LangchainEnsemble with pre-loaded chunks.
        
        Args:
            chunks_data: List of chunk metadata dicts
            chunk_texts: List of chunk text strings
            chunk_ids: List of chunk IDs
            config_path: Path to model configuration
            embedding_model: Ollama embedding model to use
            top_k: Number of top results to retrieve
            bm25_weight: Weight for BM25 retriever in ensemble (0-1)
            vector_weight: Weight for vector retriever in ensemble (0-1)
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
        
        # Store chunk data
        self.chunk_texts = chunk_texts
        self.chunk_ids = chunk_ids
        self.chunks_data = chunks_data
        
        print(f"\nSetting up Langchain Ensemble Retriever...")
        print(f"  - Chunks: {len(chunk_texts)}")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - Weights: BM25={bm25_weight}, Vector={vector_weight}")
        
        # Create Langchain documents from chunks
        self.documents = [
            Document(
                page_content=text,
                metadata={"chunk_id": chunk_ids[i], "index": i}
            )
            for i, text in enumerate(chunk_texts)
        ]
        
        # Initialize BM25 retriever
        print("\n  Setting up BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=top_k
        )
        print("  ✓ BM25 ready")
        
        # Initialize embedding model and vector retriever
        print("  Setting up vector embeddings retriever...")
        try:
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=self.ollama_host
            )
            
            # Build FAISS index
            print("    Computing embeddings (this may take a moment)...")
            self.vector_store = FAISS.from_documents(
                self.documents,
                self.embeddings
            )
            
            # Create vector retriever from FAISS
            self.vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            print("  ✓ Vector embeddings ready")
        except Exception as e:
            print(f"  ⚠ Vector embedding setup failed: {e}")
            print("    Falling back to BM25-only retrieval")
            self.vector_retriever = None
        
        # Create ensemble retriever
        if self.vector_retriever:
            print("  Setting up ensemble retriever...")
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[bm25_weight, vector_weight]
            )
            print("  ✓ Ensemble retriever ready")
        else:
            self.ensemble_retriever = self.bm25_retriever
            print("  ✓ Using BM25-only retrieval")
        
        self.top_k = top_k
        print("✓ LangchainEnsemble initialization complete\n")
    
    def answer(self, question: str) -> QAAnswer:
        """Answer a question using ensemble retrieval + Ollama."""
        with Timer() as timer:
            # Retrieve using ensemble
            results = self.ensemble_retriever.invoke(question)
            
            # Extract evidence and chunk IDs
            evidence = []
            retrieved_chunk_ids = []
            
            for doc in results:
                evidence.append(doc.page_content)
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id:
                    retrieved_chunk_ids.append(chunk_id)
            
            # Prepare context
            context = "\n\n".join(evidence) if evidence else ""
            
            # Generate answer with Ollama
            prompt = f"""Answer the following question based on the provided context.

Question: {question}

Context:
{context}

Answer:"""
            
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.actual_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "num_predict": 256
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    answer_text = response.json().get("response", "").strip()
                else:
                    answer_text = f"Error: {response.status_code}"
            except Exception as e:
                answer_text = f"Error generating answer: {e}"
        
        return QAAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            latency_seconds=timer.elapsed,
            metadata={
                "mode": "langchain_ensemble",
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "num_results": len(results),
                "retriever_type": "ensemble" if self.vector_retriever else "bm25"
            }
        )
    
    def get_config(self) -> dict:
        """Return system configuration for reproducibility."""
        return {
            "mode": "langchain_ensemble",
            "model_name": self.model_name,
            "ollama_host": self.ollama_host,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_chunks": len(self.chunk_texts),
            "retriever_type": "ensemble" if self.vector_retriever else "bm25",
        }
