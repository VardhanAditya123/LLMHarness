"""Base classes and interfaces for all QA systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
import time


@dataclass
class QAAnswer:
    """Unified output format for all QA systems."""
    
    question: str
    answer: str
    evidence: List[str] = field(default_factory=list)      # Retrieved passages (empty for raw model)
    abstained: bool = False                                 # Did model decline to answer?
    latency_seconds: float = 0.0                            # Total execution time
    metadata: Dict[str, Any] = field(default_factory=dict)  # System-specific data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence,
            "abstained": self.abstained,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata,
        }


class QASystem(ABC):
    """Abstract base class for all QA system implementations."""
    
    @abstractmethod
    def answer(self, question: str) -> QAAnswer:
        """
        Answer a question.
        
        Args:
            question: The question to answer
            
        Returns:
            QAAnswer with structured output
            
        Raises:
            Exception: If answering fails
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return system configuration for logging and reproducibility."""
        pass


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
