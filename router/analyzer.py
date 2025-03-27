"""Query analyzer component for the Dynamic LLM Router."""

from typing import Dict, Any, List, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ROUTING_CONFIG


class QueryAnalyzer:
    """Analyzes queries to extract key characteristics for routing."""
    
    def __init__(self):
        """Initialize the query analyzer."""
        self.domains = ROUTING_CONFIG["domains"]
        self.complexity_thresholds = ROUTING_CONFIG["complexity"]
        self.domain_keywords = self._build_domain_keywords()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """Build a dictionary of keywords for each domain.
        
        Returns:
            Dictionary mapping domains to lists of relevant keywords
        """
        # These are simplified keyword lists for each domain
        # In a production system, these would be much more comprehensive
        return {
            "coding": ["code", "function", "programming", "algorithm", "bug", "debug", 
                      "python", "javascript", "java", "c++", "ruby", "golang", "rust",
                      "api", "library", "framework", "class", "object", "method", 
                      "variable", "loop", "recursion", "data structure"],
            
            "creative_writing": ["story", "poem", "novel", "character", "plot", "setting", 
                               "narrative", "dialogue", "scene", "description", "creative", 
                               "write", "author", "book", "fiction", "nonfiction", "essay"],
            
            "math": ["equation", "calculation", "algebra", "calculus", "geometry", 
                    "statistics", "probability", "theorem", "proof", "number", 
                    "formula", "mathematical", "solve", "compute", "arithmetic"],
            
            "science": ["physics", "chemistry", "biology", "astronomy", "geology", 
                       "experiment", "theory", "hypothesis", "research", "scientific", 
                       "molecule", "atom", "cell", "organism", "reaction", "force", 
                       "energy", "mass", "velocity", "acceleration"],
            
            "general_knowledge": ["what", "who", "where", "when", "why", "how", 
                                "explain", "describe", "tell", "information", "fact", 
                                "history", "geography", "culture", "society", "world"],
            
            "reasoning": ["logic", "argument", "fallacy", "premise", "conclusion", 
                         "inference", "deduction", "induction", "analyze", "evaluate", 
                         "critique", "assess", "reason", "think", "consider", "judgment"],
            
            "safety": ["ethical", "moral", "safety", "security", "privacy", "bias", 
                      "fairness", "harm", "risk", "danger", "protect", "prevent", 
                      "guideline", "policy", "regulation", "compliance"],
            
            "multimodal": ["image", "picture", "photo", "video", "audio", "sound", 
                          "visual", "see", "look", "watch", "listen", "hear", "multimedia", 
                          "graphic", "diagram", "chart", "plot", "visualization"],
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze a query to extract key characteristics.
        
        Args:
            query: The input query string
            
        Returns:
            Dictionary of query characteristics
        """
        # Calculate complexity score
        complexity_score = self._calculate_complexity(query)
        
        # Determine complexity category
        if complexity_score < self.complexity_thresholds["low"]:
            complexity_category = "low"
        elif complexity_score > self.complexity_thresholds["high"]:
            complexity_category = "high"
        else:
            complexity_category = "medium"
        
        # Identify relevant domains
        domain_scores = self._identify_domains(query)
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general_knowledge"
        
        # Check for code snippets
        has_code = self._contains_code(query)
        
        # Estimate token count (very rough approximation)
        estimated_tokens = len(query.split()) * 1.3  # Rough approximation
        
        return {
            "complexity_score": complexity_score,
            "complexity_category": complexity_category,
            "domain_scores": domain_scores,
            "primary_domain": primary_domain,
            "has_code": has_code,
            "estimated_tokens": estimated_tokens,
            "query_length": len(query),
        }
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate a complexity score for the query.
        
        Args:
            query: The input query string
            
        Returns:
            Complexity score between 0 and 1
        """
        # This is a simplified complexity calculation
        # In a production system, this would use more sophisticated NLP techniques
        
        # Factors that contribute to complexity:
        # 1. Query length
        length_score = min(len(query) / 500, 1.0)  # Normalize to 0-1
        
        # 2. Sentence count and average length
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        if sentence_count > 0:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / sentence_count
            sentence_complexity = min(avg_sentence_length / 20, 1.0)  # Normalize to 0-1
        else:
            sentence_complexity = 0.0
        
        # 3. Question complexity (multiple questions, nested questions)
        question_count = query.count('?')
        question_complexity = min(question_count / 5, 1.0)  # Normalize to 0-1
        
        # 4. Vocabulary complexity
        if len(query.split()) > 3:  # Only if we have enough words
            unique_words = set(word.lower() for word in re.findall(r'\b\w+\b', query))
            vocab_complexity = min(len(unique_words) / 100, 1.0)  # Normalize to 0-1
        else:
            vocab_complexity = 0.0
        
        # 5. Presence of specialized terminology
        specialized_terms = sum(1 for domain_keywords in self.domain_keywords.values() 
                              for keyword in domain_keywords 
                              if re.search(r'\b' + re.escape(keyword) + r'\b', query.lower()))
        specialized_complexity = min(specialized_terms / 10, 1.0)  # Normalize to 0-1
        
        # Combine factors with weights
        weights = [0.15, 0.25, 0.2, 0.2, 0.2]  # Adjust weights as needed
        factors = [length_score, sentence_complexity, question_complexity, 
                  vocab_complexity, specialized_complexity]
        
        return sum(w * f for w, f in zip(weights, factors))
    
    def _identify_domains(self, query: str) -> Dict[str, float]:
        """Identify relevant domains for the query.
        
        Args:
            query: The input query string
            
        Returns:
            Dictionary mapping domains to relevance scores
        """
        query_lower = query.lower()
        domain_scores = {}
        
        # Check for keyword matches in each domain
        for domain, keywords in self.domain_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords 
                         if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower))
            
            # Calculate score based on matches and domain size
            if matches > 0:
                domain_scores[domain] = matches / len(keywords)
        
        # Normalize scores
        total_score = sum(domain_scores.values()) if domain_scores else 1.0
        normalized_scores = {domain: score / total_score 
                            for domain, score in domain_scores.items()}
        
        return normalized_scores
    
    def _contains_code(self, query: str) -> bool:
        """Check if the query contains code snippets.
        
        Args:
            query: The input query string
            
        Returns:
            True if code is detected, False otherwise
        """
        # Look for common code indicators
        code_indicators = [
            # Code blocks
            r'```\w*\n[\s\S]+?```',
            # Function definitions
            r'\bdef\s+\w+\s*\(',
            r'\bfunction\s+\w+\s*\(',
            # Variable assignments
            r'\b\w+\s*=\s*[^=]',
            # Common programming keywords
            r'\b(if|else|for|while|return|import|class|try|catch|except)\b',
            # Brackets and parentheses patterns
            r'\{[^\{\}]*\}',
            r'\([^\(\)]*\)',
            r'\[[^\[\]]*\]',
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, query):
                return True
        
        return False