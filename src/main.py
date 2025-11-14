"""FastAPI app."""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
import numpy as np
from sqlalchemy import text

from src.models import CorrelationRequest, CorrelationResponse
from src.ml.inference import CorrelationPredictor
from src.ml.rag import SemanticRAG, embed_contracts
from src.db.database import get_engine
from src.config import get_settings

# Global predictor instance
predictor: Optional[CorrelationPredictor] = None


def build_faiss_index_from_db(index_path: str = "./faiss_index"):
    """
    Build FAISS index from database on startup if it doesn't exist.

    Args:
        index_path: Path to save the index
    """
    index_dir = Path(index_path)

    # Check if index already exists
    if (index_dir / "index.faiss").exists() and (index_dir / "metadata.pkl").exists():
        print(f"FAISS index already exists at {index_path}")
        return

    print("=" * 80)
    print("Building FAISS index from database (first-time setup)...")
    print("=" * 80)

    settings = get_settings()
    engine = get_engine()

    # Load contract pairs from database
    query = """
    SELECT
        contract_a_title,
        contract_b_title,
        underlying_event_correlation
    FROM contract_correlations
    WHERE is_active = true
    ORDER BY created_at
    """

    with engine.connect() as conn:
        import pandas as pd
        df = pd.read_sql(text(query), conn)

    if len(df) == 0:
        print("Warning: No data found in contract_correlations table")
        print("RAG will not be available")
        return

    print(f"Found {len(df)} contract pairs in database")

    # Load model for embeddings (lightweight, just for embedding)
    print("Loading model for embeddings...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name,
        token=settings.hf_token,
        cache_dir=settings.model_cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=settings.hf_token,
        torch_dtype=torch.float16,
        cache_dir=settings.model_cache_dir
    )
    model.eval()

    # Compute embeddings
    print("Computing embeddings...")
    embeddings = []
    metadata = []

    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(df)}")

        embedding = embed_contracts(
            model,
            tokenizer,
            row['contract_a_title'],
            row['contract_b_title'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        embeddings.append(embedding)
        metadata.append({
            "index": idx,
            "contract_a": row['contract_a_title'],
            "contract_b": row['contract_b_title'],
            "correlation": float(row['underlying_event_correlation'])
        })

    embeddings = np.array(embeddings)
    print(f"Computed {len(embeddings)} embeddings of shape {embeddings.shape}")

    # Build and save index
    print("Building FAISS index...")
    rag = SemanticRAG(index_path=index_path)
    rag.build_index(embeddings, metadata, index_type="L2")
    rag.save()

    print(f"FAISS index saved to {index_path}")
    print("=" * 80)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - build index and load model on startup."""
    global predictor

    print("\n" + "=" * 80)
    print("Starting Contract Correlation API")
    print("=" * 80)

    # Build FAISS index if it doesn't exist
    try:
        build_faiss_index_from_db(index_path="./faiss_index")
    except Exception as e:
        print(f"Warning: Failed to build FAISS index: {e}")
        print("API will start but RAG may not be available")

    # Load model
    print("\nInitializing model...")
    try:
        predictor = CorrelationPredictor(
            model_path="./checkpoints/final_model",  # Will use base model if not found
            index_path="./faiss_index"
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize model: {e}")
        print("API will start but predictions may fail")

    print("=" * 80)
    print("API ready to serve requests")
    print("=" * 80 + "\n")

    yield

    # Cleanup
    print("\nShutting down...")


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
