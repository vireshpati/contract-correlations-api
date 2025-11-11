"""FastAPI application for contract correlation prediction."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .models import (
    PredictCorrelationRequest,
    PredictCorrelationResponse,
    HealthResponse
)
from .ml.inference import get_predictor
from .ml.rag import retrieve_context
from .database import get_db_session
from .db.queries import get_correlation_stats
from .config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    print("Starting up...")
    predictor = get_predictor()
    predictor.load_model()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Contract Correlation API",
    description="Predict correlations between prediction market contracts using Llama 3.1 8B + RAG",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Contract Correlation API",
        "endpoints": {
            "POST /predict-correlation": "Predict correlation between two contracts",
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    predictor = get_predictor()

    # Check database connection
    db_connected = False
    total_correlations = None
    try:
        with get_db_session() as session:
            stats = get_correlation_stats(session)
            total_correlations = stats["total_correlations"]
            db_connected = True
    except Exception as e:
        print(f"Database health check failed: {e}")

    return HealthResponse(
        status="healthy" if (predictor.is_loaded and db_connected) else "degraded",
        model_loaded=predictor.is_loaded,
        database_connected=db_connected,
        total_correlations=total_correlations
    )


@app.post("/predict-correlation", response_model=PredictCorrelationResponse)
async def predict_correlation(request: PredictCorrelationRequest):
    """
    Predict correlation between two prediction market contracts.

    Uses Llama 3.1 8B with optional RAG context from historical correlations.
    """
    try:
        # Get predictor
        predictor = get_predictor()

        # Retrieve RAG context if requested
        rag_context = None
        similar_examples_count = 0

        if request.use_rag:
            try:
                with get_db_session() as session:
                    settings = get_settings()
                    rag_context, similar_correlations = retrieve_context(
                        session=session,
                        contract_a=request.contract_a,
                        contract_b=request.contract_b,
                        top_k=settings.rag_top_k
                    )
                    similar_examples_count = len(similar_correlations)
            except Exception as e:
                print(f"RAG retrieval failed: {e}")
                # Continue without RAG context

        # Run prediction
        prediction = predictor.predict(
            contract_a=request.contract_a,
            contract_b=request.contract_b,
            rag_context=rag_context
        )

        # Build response
        return PredictCorrelationResponse(
            underlying_correlation=prediction["underlying_correlation"],
            correlation_type=prediction["correlation_type"],
            confidence=prediction["confidence"],
            reasoning=prediction["reasoning"],
            rag_context_used=rag_context is not None,
            similar_examples_count=similar_examples_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
