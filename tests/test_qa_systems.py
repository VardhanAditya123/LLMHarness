"""Unit tests for QA systems."""

import pytest
from systems.base import QAAnswer, QASystem, Timer
from systems.raw_model import RawModelQA


class TestQAAnswer:
    """Tests for QAAnswer dataclass."""
    
    def test_qa_answer_structure(self):
        """Test QAAnswer dataclass structure."""
        answer = QAAnswer(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            evidence=["AI is a field of computer science."],
            abstained=False,
            latency_seconds=1.5,
            metadata={"model": "test"}
        )
        
        assert answer.question == "What is AI?"
        assert answer.answer == "AI is artificial intelligence."
        assert isinstance(answer.evidence, list)
        assert len(answer.evidence) == 1
        assert isinstance(answer.abstained, bool)
        assert answer.latency_seconds == 1.5
        assert answer.metadata["model"] == "test"
    
    def test_qa_answer_defaults(self):
        """Test QAAnswer with default values."""
        answer = QAAnswer(
            question="Test?",
            answer="Test answer."
        )
        
        assert answer.evidence == []
        assert answer.abstained is False
        assert answer.latency_seconds == 0.0
        assert answer.metadata == {}
    
    def test_qa_answer_to_dict(self):
        """Test conversion to dictionary."""
        answer = QAAnswer(
            question="What is ML?",
            answer="Machine Learning is...",
            evidence=["ML definition"],
            abstained=False,
            latency_seconds=2.0,
            metadata={"model": "test"}
        )
        
        d = answer.to_dict()
        assert isinstance(d, dict)
        assert d["question"] == "What is ML?"
        assert d["answer"] == "Machine Learning is..."
        assert d["latency_seconds"] == 2.0


class TestTimer:
    """Tests for Timer context manager."""
    
    def test_timer_elapsed(self):
        """Test timer measures elapsed time correctly."""
        import time
        
        with Timer() as timer:
            time.sleep(0.1)
        
        assert timer.elapsed > 0.09
        assert timer.elapsed < 0.2
    
    def test_timer_properties(self):
        """Test timer properties."""
        timer = Timer()
        assert timer.start_time is None
        assert timer.end_time is None
        assert timer.elapsed == 0.0


class TestRawModelQA:
    """Tests for RawModelQA system."""
    
    @pytest.fixture
    def raw_model(self):
        """Initialize raw model (use pytest mark to skip if model unavailable)."""
        try:
            return RawModelQA(config_path="configs/model.yaml")
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")
    
    def test_raw_model_config(self, raw_model):
        """Test raw model configuration."""
        config = raw_model.get_config()
        
        assert config["system"] == "raw_model"
        assert "model_name" in config
        assert "device" in config
        assert "max_tokens" in config
        assert "temperature" in config
    
    def test_raw_model_answer_structure(self, raw_model):
        """Test raw model answer structure."""
        result = raw_model.answer("What is AI?")
        
        assert isinstance(result, QAAnswer)
        assert result.question == "What is AI?"
        assert isinstance(result.answer, str)
        assert result.answer != ""
        assert result.evidence == []  # Raw model has no retrieval
        assert result.abstained is False
        assert result.latency_seconds > 0
    
    def test_raw_model_multiple_questions(self, raw_model):
        """Test raw model with multiple questions."""
        questions = [
            "What is machine learning?",
            "Explain neural networks."
        ]
        
        results = []
        for q in questions:
            result = raw_model.answer(q)
            results.append(result)
            assert result.question == q
            assert len(result.answer) > 0
        
        assert len(results) == 2


class TestQASystemInterface:
    """Tests for QASystem interface compliance."""
    
    def test_raw_model_implements_interface(self):
        """Test that RawModelQA implements QASystem interface."""
        assert issubclass(RawModelQA, QASystem)
        
        # Check required methods exist
        assert hasattr(RawModelQA, 'answer')
        assert hasattr(RawModelQA, 'get_config')
        assert callable(getattr(RawModelQA, 'answer'))
        assert callable(getattr(RawModelQA, 'get_config'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
