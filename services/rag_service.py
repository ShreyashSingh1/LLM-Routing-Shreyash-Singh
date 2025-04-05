from fastapi import FastAPI, HTTPException
from models.mistral import MistralProvider
from config import MODEL_CONFIGS

app = FastAPI(title="RAG Service")

# Initialize Mistral provider
mistral_config = MODEL_CONFIGS["mistral"]
mistral = MistralProvider(mistral_config)

@app.post("/generate")
def generate_rag_response(query: str):
    try:
        response = mistral.generate(query, use_fallback=True)
        return {
            "content": response["content"],
            "rag_info": response.get("rag_info", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
