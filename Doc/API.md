**Overview**

This **Flask-based API** routes user queries to dynamically selected LLM providers like OpenAI, Anthropic, etc., based on query analysis, caching, and routing strategies. It also supports experimentation, caching statistics, and user feedback collection.

---

**1. Authentication**

This API requires authentication using an API key (if implemented). Currently, no strict header-based auth is enforced, but can be added for production.

---

**2. Query LLM Provider**

**Endpoint:**

POST /query

**Description:**

Analyzes the input query and dynamically routes it to the best-suited LLM provider. Returns the generated response with metadata like model used, provider, token usage, latency, etc.

**Request Body:**

```python
{
  "query": "What is the capital of France?",
  "context": "geography"  // optional
}
```

**Response Example:**

```python
{
  "content": "The capital of France is Paris.",
  "cost": 0.00036,
  "latency": 0.0234,
  "metadata": {
    "api_latency": 1.923,
    "provider": "openai",
    "query_analysis": {
      "complexity_category": "low",
      "complexity_score": 0.0131,
      "language": {
        "code": "en",
        "name": "English",
        "confidence": 9.06,
        "is_reliable": true
      },
      "sentiment": {
        "polarity": 0,
        "sentiment": "neutral",
        "urgency": "low",
        "urgency_score": 0
      },
      "primary_topic": "geography"
    }
  },
  "model": "Llama3-8b-8192",
  "tokens": {
    "prompt": 11,
    "completion": 25,
    "total": 36
  }
}
```

---

**3. List Available Providers**

**Endpoint:**

GET /providers

**Description:**

Returns the list of all registered LLM providers along with their metadata like cost, response time, and strengths.

**Response Example:**

```python
{
  "providers": {
    "openai": {
      "default_model": "gpt-4",
      "fallback_model": "gpt-3.5-turbo",
      "strengths": ["reasoning", "code"],
      "response_time": 0.9,
      "cost_per_1k_tokens": 0.002
    },
    "anthropic": {
      "default_model": "claude-3",
      "fallback_model": "claude-instant",
      "strengths": ["conversation", "ethics"],
      "response_time": 1.2,
      "cost_per_1k_tokens": 0.0015
    }
  }
}
```

---

**4. Submit Feedback for a Query**

**Endpoint:**

POST /feedback

**Description:**

Submits a feedback rating and optional comment for a previously handled query.

**Request Body:**

```json
{
  "query_id": "abc123",
  "rating": 4,
  "comments": "Accurate answer, slight delay."
}
```

**Response Example:**

```jsx
{
  "message": "Feedback recorded successfully"
}
```

---

**5. Get Cache Statistics**

**Endpoint:**

GET /cache/stats

**Description:**

Returns internal cache usage stats such as hits, misses, eviction count, and strategy being used.

**Response Example:**

```python
{
  "enabled": true,
  "stats": {
    "evictions": 0,
    "expirations": 0,
    "hit_rate": 0,
    "hits": 0,
    "max_size": 1000,
    "misses": 1,
    "size": 1,
    "strategy": "ttl"
  }
}
```

---

**6. Clear Cache**

**Endpoint:**

POST /cache/clear

**Description:**

Clears the internal cache, removing all stored query results.

**Response Example:**

```python
{
  "message": "Cache cleared successfully"
}
```

---

**7. Experimentation Status**

**Endpoint:**

GET /experiments

**Description:**

Returns the current experimentation settings such as active strategies being tested.

**Response Example:**

```python
{
  "enabled": true,
  "active_experiment": "adaptive-routing-v2",
  "strategies": ["round_robin", "load_balanced", "adaptive"]
}
```

---

**8. Welcome Message (Root)**

**Endpoint:**

GET /

**Description:**

A simple root message to confirm the server is live.

**Response Example:**

```python
{
  "message": "Welcome to the Dynamic LLM Router API"
}
```

---

**9. Health Check**

**Endpoint:**

GET /health

**Description:**

Verifies the system is up and healthy.

**Response Example:**

```python
{
  "status": "healthy"
}
```

---

**Summary**

| **Endpoint** | **Method** | **Description** | **Authentication** |
| --- | --- | --- | --- |
| /query | POST | Routes query to best-fit LLM and returns a response | ❌ Optional |
| /providers | GET | Lists available LLM providers and metadata | ❌ Optional |
| /feedback | POST | Submit user feedback for a query | ❌ Optional |
| /cache/stats | GET | Returns cache performance statistics | ❌ Optional |
| /cache/clear | POST | Clears all cached data | ❌ Optional |
| /experiments | GET | Shows experimentation strategy and status | ❌ Optional |
| / | GET | Welcome root endpoint | ❌ Optional |
| /health | GET | Health check | ❌ Optional |

---