"""Query analyzer component for the Dynamic LLM Router."""

from typing import Dict, Any, List, Tuple
import re
import numpy as np
import langid
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import ROUTING_CONFIG, ANALYZER_CONFIG


class QueryAnalyzer:
    """Analyzes queries to extract key characteristics for routing."""
    
    def __init__(self):
        """Initialize the query analyzer."""
        self.domains = ROUTING_CONFIG["domains"]
        self.complexity_thresholds = ROUTING_CONFIG["complexity"]
        self.domain_keywords = self._build_domain_keywords()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Initialize topic modeling components
        self.n_topics = 10  # Number of topics to extract
        self.topic_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        self.topic_names = [
            "general", "technology", "science", "business", "arts",
            "politics", "health", "education", "sports", "entertainment"
        ]
        
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
        # Calculate complexity score using enhanced method
        complexity_score, complexity_details = self._calculate_complexity(query)
        
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
        
        # Perform sentiment analysis
        sentiment_data = self._analyze_sentiment(query)
        
        # Detect language
        language_data = self._detect_language(query)
        
        # Perform topic classification
        topic_data = self._classify_topics(query)
        
        data = {
            "complexity_score": complexity_score,
            "complexity_category": complexity_category,
            "complexity_details": complexity_details,
            "domain_scores": domain_scores,
            "primary_domain": primary_domain,
            "has_code": has_code,
            "sentiment": sentiment_data,
            "language": language_data,
            "topics": topic_data,
            "estimated_tokens": estimated_tokens,
            "query_length": len(query),
        }
        
        print(data)
        
        return {
            "complexity_score": complexity_score,
            "complexity_category": complexity_category,
            "complexity_details": complexity_details,
            "domain_scores": domain_scores,
            "primary_domain": primary_domain,
            "has_code": has_code,
            "sentiment": sentiment_data,
            "language": language_data,
            "topics": topic_data,
            "estimated_tokens": estimated_tokens,
            "query_length": len(query),
        }
    
    def _calculate_complexity(self, query: str) -> Tuple[float, Dict[str, float]]:
        """Calculate a complexity score for the query.
        
        Args:
            query: The input query string
            
        Returns:
            Tuple of (complexity score between 0 and 1, details dictionary)
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
        
        complexity_score = sum(w * f for w, f in zip(weights, factors))
        
        # Return both the score and the detailed breakdown
        details = {
            "length_score": length_score,
            "sentence_complexity": sentence_complexity,
            "question_complexity": question_complexity,
            "vocab_complexity": vocab_complexity,
            "specialized_complexity": specialized_complexity
        }
        
        return complexity_score, details
    
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
            query: The query string
            
        Returns:
            True if the query likely contains code, False otherwise
        """
        # Look for common code indicators
        code_indicators = [
            # Function definitions
            r"def\s+\w+\s*\(",
            r"function\s+\w+\s*\(",
            # Variable assignments with specific types
            r"\b(?:var|let|const|int|float|double|string|bool|char)\s+\w+\s*=",
            # Control structures
            r"\b(?:if|for|while|switch|case)\s*\(",
            # Class definitions
            r"class\s+\w+\s*(?:\(|\{|:)",
            # Import statements
            r"\b(?:import|from|require|include)\s+[\w\.*]+",
            # Code blocks
            r"```[\w]*\n[\s\S]*?```",
            # Inline code
            r"`[^`]+`",
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, query):
                return True
                
        return False
        
    def _analyze_sentiment(self, query: str) -> Dict[str, Any]:
        """Analyze the sentiment and emotional tone of the query.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Use TextBlob for sentiment analysis
        blob = TextBlob(query)
        polarity = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)
        
        # Determine sentiment category
        if polarity > 0.3:
            sentiment = "positive"
        elif polarity < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Check for urgency based on keywords
        urgency_keywords = ANALYZER_CONFIG["sentiment_analysis"]["urgency_keywords"]
        urgency_score = 0.0
        
        for keyword in urgency_keywords:
            if keyword.lower() in query.lower():
                urgency_score += 0.2  # Increase urgency score for each keyword found
        
        urgency_score = min(1.0, urgency_score)  # Cap at 1.0
        
        # Determine urgency category
        if urgency_score > 0.6:
            urgency = "high"
        elif urgency_score > 0.2:
            urgency = "medium"
        else:
            urgency = "low"
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment,
            "urgency_score": urgency_score,
            "urgency": urgency
        }
    
    def _detect_language(self, query: str) -> Dict[str, Any]:
        """Detect the language of the query.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with language detection results
        """
        # Use langid for language detection
        lang, confidence = langid.classify(query)
        
        # Get confidence threshold from config
        threshold = ANALYZER_CONFIG["language_detection"]["confidence_threshold"]
        
        # If confidence is below threshold, default to English
        if confidence < threshold:
            lang = "en"
            is_reliable = False
        else:
            is_reliable = True
        
        # Map language codes to names (simplified)
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
        }
        
        language_name = language_names.get(lang, f"Unknown ({lang})")
        
        return {
            "code": lang,
            "name": language_name,
            "confidence": confidence,
            "is_reliable": is_reliable
        }
    
    def _classify_topics(self, query: str) -> Dict[str, Any]:
        """Classify the query into topics.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with topic classification results
        """
        # Simple keyword-based classification for common topics
        topics = {}
        
        # Check if we have enough text for meaningful topic modeling
        if len(query.split()) < 5:
            return {
                "primary_topic": "unknown",
                "topic_scores": {},
                "method": "insufficient_text"
            }
        
        try:
            # Use LDA for topic modeling
            # Convert query to document-term matrix
            dtm = self.topic_vectorizer.fit_transform([query])
            
            # If vocabulary is too small, fall back to keyword matching
            if dtm.shape[1] < 5:
                return self._classify_topics_by_keywords(query)
            
            # Transform to topic space
            topic_distribution = self.lda.fit_transform(dtm)[0]
            
            # Get topic scores
            topic_scores = {}
            for i, score in enumerate(topic_distribution):
                if i < len(self.topic_names):
                    topic_scores[self.topic_names[i]] = float(score)
            
            # Get primary topic
            primary_topic = self.topic_names[topic_distribution.argmax()] if topic_distribution.max() > 0.2 else "general"
            
            return {
                "primary_topic": primary_topic,
                "topic_scores": topic_scores,
                "method": "lda"
            }
            
        except Exception as e:
            # Fall back to keyword matching if LDA fails
            return self._classify_topics_by_keywords(query)
    
    def _classify_topics_by_keywords(self, query: str) -> Dict[str, Any]:
        """Classify topics using keyword matching as a fallback method.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with topic classification results
        """
        # Define keywords for each topic
        topic_keywords = {
            "technology": ["computer", "software", "hardware", "app", "technology", "tech", "digital"],
            "science": ["science", "scientific", "experiment", "research", "theory", "hypothesis"],
            "business": ["business", "company", "market", "finance", "economy", "investment", "stock"],
            "arts": ["art", "music", "film", "movie", "book", "literature", "painting", "creative"],
            "politics": ["politics", "government", "election", "policy", "law", "president", "congress"],
            "health": ["health", "medical", "doctor", "disease", "treatment", "medicine", "symptom"],
            "education": ["education", "school", "university", "college", "student", "teacher", "learn"],
            "sports": ["sport", "game", "team", "player", "coach", "tournament", "championship"],
            "entertainment": ["entertainment", "celebrity", "tv", "show", "actor", "actress", "movie"],
        }
        
        # Count keyword matches for each topic
        topic_scores = {topic: 0.0 for topic in topic_keywords}
        query_lower = query.lower()
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    topic_scores[topic] += 0.2  # Increase score for each keyword match
        
        # Normalize scores
        max_score = max(topic_scores.values()) if topic_scores.values() else 0
        if max_score > 0:
            topic_scores = {t: min(1.0, s / max_score) for t, s in topic_scores.items()}
        
        # Determine primary topic
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0] if max_score > 0.2 else "general"
        
        return {
            "primary_topic": primary_topic,
            "topic_scores": topic_scores,
            "method": "keyword_matching"
        }