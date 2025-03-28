"""Knowledge base module for RAG-style fallback implementation."""

from typing import Dict, Any, List, Optional
import random

class KnowledgeBase:
    """Simple knowledge base for RAG-style fallback implementation.
    
    This class provides a simple knowledge retrieval system that can be used
    by LLM providers as a fallback mechanism when their primary models
    are unavailable or unsuitable for a query.
    """
    
    def __init__(self):
        """Initialize the knowledge base with sample data."""
        # In a real implementation, this would connect to a vector database
        # or other knowledge storage system. For this mock implementation,
        # we'll use a simple dictionary of pre-defined responses.
        self.knowledge = {
            "general_knowledge": [
                "Paris is the capital of France.",
                "The Earth orbits around the Sun.",
                "Water boils at 100 degrees Celsius at sea level.",
                "There are 7 continents on Earth.",
                "The Great Wall of China is visible from space."
            ],
            "coding": [
                "Python is a high-level programming language known for its readability.",
                "JavaScript is primarily used for web development.",
                "A function is a reusable block of code that performs a specific task.",
                "Object-oriented programming is a programming paradigm based on objects.",
                "Git is a distributed version control system."
            ],
            "math": [
                "Pi (π) is approximately equal to 3.14159.",
                "The Pythagorean theorem states that a² + b² = c² in a right triangle.",
                "Calculus is the mathematical study of continuous change.",
                "A prime number is a natural number greater than 1 that is not divisible by any other number except 1 and itself.",
                "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a."
            ],
            "science": [
                "The theory of relativity was developed by Albert Einstein.",
                "DNA stands for deoxyribonucleic acid.",
                "The periodic table organizes chemical elements by their properties.",
                "Newton's third law states that for every action, there is an equal and opposite reaction.",
                "Photosynthesis is the process by which plants convert light energy into chemical energy."
            ]
        }
    
    def retrieve(self, query: str, domain: str = "general_knowledge", num_results: int = 1) -> List[str]:
        """Retrieve relevant information from the knowledge base.
        
        Args:
            query: The input query string
            domain: The domain to search in (defaults to general_knowledge)
            num_results: Number of results to return
            
        Returns:
            List of relevant information snippets
        """
        # In a real implementation, this would perform semantic search
        # or other retrieval methods. For this mock implementation,
        # we'll just return random entries from the specified domain.
        domain_data = self.knowledge.get(domain, self.knowledge["general_knowledge"])
        
        # Simulate retrieval by returning random entries
        # In a real implementation, this would return the most relevant entries
        if len(domain_data) <= num_results:
            return domain_data
        else:
            return random.sample(domain_data, num_results)
    
    def get_available_domains(self) -> List[str]:
        """Get the list of available knowledge domains.
        
        Returns:
            List of domain names
        """
        return list(self.knowledge.keys())