# Contract Correlation API - Simple Application

A FastAPI application that predicts correlation between prediction market contracts using Llama 3.1 8B with QLoRA fine-tuning and RAG.

## Architecture

### Core Components

1. **FastAPI API** (`src/main.py`)
   - `/predict-correlation` - Main endpoint for predictions
   - `/health` - Health check with model status
   - Loads model on startup via lifespan manager

2. **Models** (`src/models.py`)
   - `CorrelationRequest` - Input validation
   - `CorrelationResponse` - Output format with correlation, type, confidence, reasoning

3. **Inference** (`src/ml/inference.py`)
   - `CorrelationPredictor` - Handles model loading and prediction
   - Supports QLoRA fine-tuned models
   - Integrates RAG for similar examples

4. **RAG System** (`src/ml/rag.py`)
   - `SemanticRAG` - FAISS-based vector search
   - `embed_contracts()` - Generate embeddings from Llama
   - Retrieves top-k similar contract pairs

5. **Training** (`src/ml/training.py`)
   - **Hybrid Loss**: Cross-entropy (generation) + MSE (correlation value)
   - **RAG Integration**: Always uses RAG during training
   - **Proper Splits**: 70% train / 15% val / 15% test
   - `HybridLossTrainer` - Custom trainer with dual objectives

6. **Database** (`src/db/`)
   - Schema for contract correlations
   - Connection management with SQLAlchemy

## Key Features

### 1. Hybrid Loss Training
- **Language Modeling Loss**: Standard cross-entropy for text generation
- **Correlation Loss**: MSE on predicted vs. true correlation values
- Configurable weight (default: 30% correlation, 70% LM)

### 2. RAG During Training & Inference
- Dataset automatically adds RAG context to each example
- Retrieves 3 similar contract pairs
- Helps model learn from similar historical correlations

### 3. Proper Train/Val/Test Splits
- 70% training data
- 15% validation data (for checkpoint selection)
- 15% test data (held out for final evaluation)

## Expected Output Format

```json
{
    "underlying_correlation": 0.5,
    "correlation_type": "positive",
    "confidence": 0.95,
    "reasoning": "Both contracts are related to the same underlying event..."
}
```

## Usage

### Running the API

```bash
# On H200 GPU machine
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**On first startup, the API will automatically:**
1. Check if FAISS index exists
2. If not, build it from the database (one-time setup)
3. Load the model and RAG system
4. Start serving requests

No manual index building required!

### Training

```bash
# Train with hybrid loss + RAG
python examples/train_model.py
```

### Testing

```bash
pytest tests/
```

### Making Predictions

```python
import requests

response = requests.post("http://localhost:8000/predict-correlation", json={
    "contract_a": "Will Bitcoin reach $100,000 by 2025?",
    "contract_b": "Will Ethereum reach $10,000 by 2025?",
    "use_rag": True
})

print(response.json())
```

## File Structure

```
src/
├── main.py              # FastAPI app
├── models.py            # Pydantic models
├── config.py            # Settings
├── db/
│   ├── database.py      # DB connection
│   └── schema.py        # SQLAlchemy schema
└── ml/
    ├── inference.py     # Prediction logic
    ├── rag.py           # RAG system
    └── training.py      # Training with hybrid loss

tests/
├── test_api.py          # API tests
└── test_models.py       # Model validation tests

examples/
├── predict_correlation.py    # Example usage
├── train_model.py           # Training script
└── build_faiss_index.py     # Build RAG index
```

## Environment Variables

```bash
# Database
PGHOST=your_host
PGPORT=5432
PGDATABASE=postgres
PGUSER=your_user
PGPASSWORD=your_password

# HuggingFace Token
HF_TOKEN=your_token
```

## Design Principles

- **Functional patterns** - Pure functions, minimal side effects
- **Concise & clean** - No unnecessary complexity
- **Modular** - Each component has single responsibility
- **Type hints** - Full type annotations for clarity
