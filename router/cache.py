"""Caching module for storing frequent queries and their responses."""

import time
from typing import Dict, Any, Optional, List, Tuple, Callable
import hashlib
import json
from collections import OrderedDict
from threading import Lock


class CacheEntry:
    """Represents a single cached response with metadata."""
    
    def __init__(self, query: str, response: Dict[str, Any], provider: str, ttl: int = 3600):
        """Initialize a cache entry.
        
        Args:
            query: The original query string
            response: The response data
            provider: The provider that generated the response
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.query = query
        self.response = response
        self.provider = provider
        self.ttl = ttl
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        return time.time() > (self.created_at + self.ttl)
    
    def access(self) -> None:
        """Record an access to this cache entry."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the cache entry to a dictionary.
        
        Returns:
            Dictionary representation of the cache entry
        """
        return {
            "query": self.query,
            "response": self.response,
            "provider": self.provider,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "expires_at": self.created_at + self.ttl,
            "age": time.time() - self.created_at,
            "is_expired": self.is_expired()
        }


class QueryCache:
    """Cache for storing query responses with configurable invalidation strategies."""
    
    # Invalidation strategy functions
    INVALIDATION_STRATEGIES = {
        "ttl": lambda entry: entry.is_expired(),
        "lru": lambda entry: False,  # LRU is handled by OrderedDict
        "lfu": lambda entry: False,  # LFU requires separate implementation
    }
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600, 
                 strategy: str = "ttl"):
        """Initialize the query cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds
            strategy: Cache invalidation strategy ('ttl', 'lru', or 'lfu')
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        # Validate strategy
        if strategy not in self.INVALIDATION_STRATEGIES:
            raise ValueError(f"Invalid cache invalidation strategy: {strategy}")
        
        # Use OrderedDict for LRU functionality
        self.cache = OrderedDict()
        
        # For LFU, we'll track access counts separately
        self.access_counts = {}
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def _generate_key(self, query: str) -> str:
        """Generate a cache key for a query.
        
        Args:
            query: The query string
            
        Returns:
            Cache key string
        """
        # Normalize the query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        
        # Generate a hash of the normalized query
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get a cached response for a query.
        
        Args:
            query: The query string
            
        Returns:
            Cached response with metadata or None if not found
        """
        key = self._generate_key(query)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    # Remove expired entry
                    del self.cache[key]
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    return None
                
                # Update access information
                entry.access()
                
                # For LRU strategy, move to end of OrderedDict
                if self.strategy == "lru":
                    self.cache.move_to_end(key)
                
                # For LFU strategy, update access count
                if self.strategy == "lfu":
                    self.access_counts[key] = entry.access_count
                
                self.stats["hits"] += 1
                
                # Return a copy of the response with cache metadata
                response_with_metadata = entry.response.copy()
                if "metadata" not in response_with_metadata:
                    response_with_metadata["metadata"] = {}
                
                response_with_metadata["metadata"]["cached"] = True
                response_with_metadata["metadata"]["cache_age"] = time.time() - entry.created_at
                response_with_metadata["metadata"]["original_provider"] = entry.provider
                
                return response_with_metadata
            else:
                self.stats["misses"] += 1
                return None
    
    def set(self, query: str, response: Dict[str, Any], provider: str, 
            ttl: Optional[int] = None) -> None:
        """Store a response in the cache.
        
        Args:
            query: The query string
            response: The response to cache
            provider: The provider that generated the response
            ttl: Optional custom TTL in seconds
        """
        key = self._generate_key(query)
        ttl = ttl if ttl is not None else self.default_ttl
        
        with self.lock:
            # Create new cache entry
            entry = CacheEntry(query, response, provider, ttl)
            
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_entries()
            
            # Add to cache
            self.cache[key] = entry
            
            # For LRU strategy, ensure it's at the end (most recently used)
            if self.strategy == "lru":
                self.cache.move_to_end(key)
            
            # For LFU strategy, initialize access count
            if self.strategy == "lfu":
                self.access_counts[key] = 0
    
    def _evict_entries(self, count: int = 1) -> None:
        """Evict entries based on the current invalidation strategy.
        
        Args:
            count: Number of entries to evict
        """
        if not self.cache:
            return
        
        # First, try to remove expired entries
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired_keys:
            del self.cache[key]
            if self.strategy == "lfu" and key in self.access_counts:
                del self.access_counts[key]
            self.stats["expirations"] += 1
        
        # If we removed enough or the cache is now empty, we're done
        if len(expired_keys) >= count or not self.cache:
            return
        
        # Otherwise, evict based on strategy
        remaining = count - len(expired_keys)
        
        if self.strategy == "lru":
            # LRU: remove from the beginning of the OrderedDict (least recently used)
            for _ in range(min(remaining, len(self.cache))):
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
        
        elif self.strategy == "lfu":
            # LFU: remove entries with lowest access counts
            if self.access_counts:
                # Sort by access count
                sorted_keys = sorted(self.access_counts.items(), key=lambda x: x[1])
                
                # Remove the least frequently used entries
                for key, _ in sorted_keys[:remaining]:
                    if key in self.cache:
                        del self.cache[key]
                        del self.access_counts[key]
                        self.stats["evictions"] += 1
    
    def invalidate(self, query: Optional[str] = None) -> None:
        """Invalidate cache entries.
        
        Args:
            query: Optional specific query to invalidate. If None, invalidates all expired entries.
        """
        with self.lock:
            if query:
                # Invalidate specific query
                key = self._generate_key(query)
                if key in self.cache:
                    del self.cache[key]
                    if self.strategy == "lfu" and key in self.access_counts:
                        del self.access_counts[key]
            else:
                # Invalidate all expired entries
                expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
                for key in expired_keys:
                    del self.cache[key]
                    if self.strategy == "lfu" and key in self.access_counts:
                        del self.access_counts[key]
                    self.stats["expirations"] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "strategy": self.strategy
            }
    
    def get_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get cache entries for inspection.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of cache entries as dictionaries
        """
        with self.lock:
            entries = []
            for key, entry in list(self.cache.items())[:limit]:
                entries.append({
                    "key": key,
                    **entry.to_dict()
                })
            return entries