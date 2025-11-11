"""FastAPI app."""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional

from src.models import CorrelationRequest, CorrelationResponse
from src.ml.inference import CorrelationPredictor

# Global predictor instance
predictor: Optional[CorrelationPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    global predictor

    print("Initializing model...")
    try:
        # Initialize predictor with optional fine-tuned model path
        predictor = CorrelationPredictor(
            model_path="./checkpoints/final_model",  # Will use base model if not found
            index_path="./faiss_index"
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize model: {e}")
        print("API will start but predictions may fail")

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Contract Correlation API",
    description="Predict correlations between prediction market contracts using Llama 3.1 8B + RAG",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Contract Correlation API",
        "model_loaded": predictor is not None
    }


@app.get("/health")
def health():
    """Health check with model status."""
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "rag_available": predictor.rag is not None if predictor else False
    }


@app.post("/predict-correlation", response_model=CorrelationResponse)
def predict_correlation(request: CorrelationRequest):
    """
    Predict correlation between two prediction market contracts.

    Args:
        request: Correlation request with contract descriptions

    Returns:
        Prediction with correlation score, type, confidence, and reasoning
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Make prediction
        result = predictor.predict(
            contract_a=request.contract_a,
            contract_b=request.contract_b,
            use_rag=request.use_rag
        )

        return CorrelationResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
