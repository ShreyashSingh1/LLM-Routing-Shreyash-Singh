from fastapi import FastAPI, HTTPException
from router.cache import CacheManager

app = FastAPI(title="Cache Service")

# Initialize Cache Manager
cache_manager = CacheManager()

@app.get("/stats")
def get_cache_stats():
    try:
        return cache_manager.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
def clear_cache():
    try:
        cache_manager.clear()
        return {"message": "Cache cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
