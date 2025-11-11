# Contract Correlation API

FastAPI application that predicts how prediction market contracts correlate with each other using Llama 3.1 8B + RAG.

## Features

- **Llama 3.1 8B**: Fine-tunable with QLoRA for domain-specific predictions
- **RAG System**: Retrieves similar historical correlations from database
- **Functional Design**: Pure functions, immutable data structures, composition
- **4-bit Quantization**: Runs efficiently on M2 Mac (16GB+ RAM recommended)
- **FastAPI**: Modern async API with automatic validation

## Architecture

```
src/
├── main.py              # FastAPI application
├── models.py            # Pydantic request/response models
├── config.py            # Settings management
├── database.py          # Database connection
├── db/
│   ├── schema.py        # SQLAlchemy models
│   └── queries.py       # Functional query builders
└── ml/
    ├── inference.py     # Llama 3.1 8B inference
    ├── training.py      # QLoRA fine-tuning (GPU)
    ├── rag.py           # RAG retrieval
    └── prompt_builder.py # Pure prompt functions
```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment** (`.env` already exists):
```bash
# Database credentials are already configured
# HF_TOKEN is already set
```

3. **Test database connection**:
```bash
python examples/test_database.py
```

## Usage

### Start API Server

```bash
# Development
uvicorn src.main:app --reload

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

**POST /predict-correlation**
```json
{
  "contract_a": "Will Bitcoin reach $100,000 by end of 2025?",
  "contract_b": "Will Ethereum reach $10,000 by end of 2025?",
  "use_rag": true
}
```

Response:
```json
{
  "underlying_correlation": 0.75,
  "correlation_type": "positive",
  "confidence": 0.85,
  "reasoning": "Both contracts relate to major cryptocurrency price movements...",
  "rag_context_used": true,
  "similar_examples_count": 5
}
```

**GET /health**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "total_correlations": 750
}
```

### Examples

```bash
# Predict correlation via API
python examples/predict_correlation.py

# Test database
python examples/test_database.py
```

## Training (GPU Machine)

Train the model with QLoRA on your GPU machine:

```bash
# Full training on all data
python examples/train_model.py

# Test with limited data
python examples/train_model.py --limit 100
```

Training config:
- **LoRA rank**: 16
- **4-bit quantization**: NF4
- **Batch size**: 4 (effective 16 with gradient accumulation)
- **Learning rate**: 2e-4
- **Epochs**: 3

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Data Schema

The `contract_correlations` table contains 750 examples with:
- Contract A/B titles, venues, and metadata
- Correlation values and types
- Reasoning and common factors
- Confidence scores

## Development Notes

### M2 Mac Optimization
- Uses 4-bit quantization (bitsandbytes)
- MPS backend when available
- CPU fallback support
- Estimated 8-12GB RAM usage

### Functional Patterns
- Pure functions for prompt building
- Immutable data structures
- Function composition for queries
- No side effects in core logic

### RAG Strategy
- Keyword-based similarity search
- Top-k retrieval (default: 5)
- Context formatting for LLM
- Graceful fallback without context

## Project Structure

```
.
├── src/              # Source code
├── tests/            # Unit tests
├── examples/         # Example scripts
├── noteboooks/       # Data exploration
├── resources/        # Documentation
├── requirements.txt  # Dependencies
└── .env             # Configuration
```

## License

MIT