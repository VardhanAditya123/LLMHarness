"""Raw Model QA system: Gemma 1B without retrieval via Ollama."""

import requests
import yaml
import json
from pathlib import Path

from systems.base import QASystem, QAAnswer, Timer


class RawModelQA(QASystem):
    """Baseline: Gemma 1B generation without retrieval via Ollama."""
    
    def __init__(self, config_path: str = "configs/model.yaml"):
        """Initialize raw model with Ollama configuration.
        
        Args:
            config_path: Path to model configuration YAML file
            
        Raises:
            FileNotFoundError: If config file not found
            ConnectionError: If Ollama server not accessible
        """
        # Load configuration
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.model_config = config.get("ollama_model") or config.get("raw_model")
        self.model_name = self.model_config["model_name"]
        self.ollama_host = self.model_config.get("ollama_host", "http://localhost:11434")
        self.max_tokens = self.model_config["max_tokens"]
        self.temperature = self.model_config["temperature"]
        self.top_p = self.model_config["top_p"]
        
        # Verify Ollama connection and get actual model name
        print(f"Connecting to Ollama at {self.ollama_host}")
        self.actual_model_name = None  # Will store full model name with tag
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server returned status {response.status_code}")
            
            # Check if model is available and get full name with tag
            tags = response.json().get("models", [])
            available_models = [m.get("name", "").split(":")[0] for m in tags]
            
            if self.model_name not in available_models:
                print(f"⚠ Model '{self.model_name}' not found in Ollama")
                print(f"Available models: {', '.join(set(available_models))}")
                print(f"Pull the model with: ollama pull {self.model_name}")
                raise ConnectionError(f"Model '{self.model_name}' not available in Ollama")
            
            # Get the actual full model name (with tag) from tags response
            for model in tags:
                full_name = model.get("name", "")
                base_name = full_name.split(":")[0]
                if base_name == self.model_name:
                    self.actual_model_name = full_name
                    break
            
            if not self.actual_model_name:
                self.actual_model_name = f"{self.model_name}:latest"
            
            print(f"✓ Ollama connected. Model '{self.actual_model_name}' available")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_host}. "
                f"Make sure Ollama is running: ollama serve"
            ) from e
    
    def answer(self, question: str) -> QAAnswer:
        """Generate an answer to the question via Ollama.
        
        Args:
            question: The question to answer
            
        Returns:
            QAAnswer with generated response from Ollama
        """
        with Timer() as timer:
            # Create prompt
            prompt = f"Question: {question}\n\nAnswer:"
            
            # Call Ollama API with actual model name
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.actual_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    },
                    timeout=300  # 5 minute timeout for generation
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")
                
                result = response.json()
                answer_text = result.get("response", "").replace(prompt, "").strip()
                
            except requests.exceptions.Timeout:
                answer_text = "[Timeout: Model took too long to respond]"
            except requests.exceptions.ConnectionError:
                answer_text = "[Error: Could not connect to Ollama server]"
            except json.JSONDecodeError:
                answer_text = "[Error: Invalid response from Ollama]"
        
        return QAAnswer(
            question=question,
            answer=answer_text,
            evidence=[],  # No retrieval for raw model
            abstained=False,
            latency_seconds=timer.elapsed,
            metadata={
                "model": self.actual_model_name,
                "source": "ollama",
                "ollama_host": self.ollama_host,
                "temperature": self.temperature,
            }
        )
    
    def get_config(self) -> dict:
        """Return system configuration."""
        return {
            "system": "raw_model_ollama",
            "model_name": self.model_name,
            "source": "ollama",
            "ollama_host": self.ollama_host,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


if __name__ == "__main__":
    # Quick test with Ollama
    try:
        qa = RawModelQA()
        
        test_questions = [
            "What is machine learning?",
            "Explain the concept of neural networks.",
            "What is a database index?"
        ]
        
        print("\n" + "="*60)
        print("RAW MODEL BASELINE TEST (via Ollama)")
        print("="*60)
        
        for q in test_questions:
            print(f"\nQ: {q}")
            result = qa.answer(q)
            print(f"A: {result.answer}")
            print(f"Latency: {result.latency_seconds:.2f}s")
            print(f"Model: {result.metadata['model']}")
            print("-" * 60)
    
    except ConnectionError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use Ollama:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. In another terminal, pull the model:")
        print("   ollama pull gemma")
        print("4. Then run this script again")

