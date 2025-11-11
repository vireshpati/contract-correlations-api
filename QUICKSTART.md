# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Note for M2 Mac**: Installing `bitsandbytes` and `torch` may take some time. The model will use ~8-12GB RAM when loaded.

## Verify Installation

```bash
# Verify code structure
python verify_code.py

# Run unit tests
pytest tests/ -v

# Test database connection
python examples/test_database.py
```

## Start API Server

```bash
# Development mode with auto-reload
uvicorn src.main:app --reload

# The server will start on http://localhost:8000
```

**Important**: First startup will download Llama 3.1 8B model (~4-5GB with 4-bit quantization). This takes 5-10 minutes depending on your connection.

## Test the API

### Option 1: Use the example script

```bash
# In a new terminal (while API server is running)
python examples/predict_correlation.py
```

### Option 2: Use curl

```bash
curl -X POST http://localhost:8000/predict-correlation \
  -H "Content-Type: application/json" \
  -d '{
    "contract_a": "Will Bitcoin reach $100,000 by end of 2025?",
    "contract_b": "Will Ethereum reach $10,000 by end of 2025?",
    "use_rag": true
  }'
```

### Option 3: Use the interactive docs

Open http://localhost:8000/docs in your browser for Swagger UI.

## Expected Response

```json
{
  "underlying_correlation": 0.75,
  "correlation_type": "positive",
  "confidence": 0.85,
  "reasoning": "Both contracts relate to cryptocurrency price movements...",
  "rag_context_used": true,
  "similar_examples_count": 5
}
```

## Training on GPU Machine

Copy the entire project to your GPU machine and run:

```bash
# Full training on 750 examples
python examples/train_model.py

# Test with limited data
python examples/train_model.py --limit 100
```

Model will be saved to `./models/llama-3.1-8b-correlation-qlora/`

## Project Structure

```
contract-correlations-api/
├── src/
│   ├── main.py              # FastAPI app
│   ├── models.py            # Request/response models
│   ├── config.py            # Settings
│   ├── database.py          # DB connection
│   ├── db/
│   │   ├── schema.py        # Table definitions
│   │   └── queries.py       # Query functions
│   └── ml/
│       ├── inference.py     # Llama 3.1 inference
│       ├── training.py      # QLoRA training
│       ├── rag.py           # RAG retrieval
│       └── prompt_builder.py # Prompt templates
├── tests/                   # Unit tests
├── examples/                # Example scripts
└── requirements.txt         # Dependencies
```

## Common Issues

### bitsandbytes installation fails on Mac
```bash
# Try installing from source
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

### Model loading is slow
- First load downloads ~5GB model
- Subsequent loads use cache (faster)
- Consider increasing swap if RAM < 16GB

### Database connection fails
- Check `.env` file has correct credentials
- Verify network connectivity to Azure PostgreSQL
- Test with: `python examples/test_database.py`

## Development Tips

1. **Testing without model loading**: Set environment variable:
   ```bash
   SKIP_MODEL_LOAD=1 pytest tests/
   ```

2. **Using a fine-tuned model**: After training, update `.env`:
   ```
   MODEL_NAME=./models/llama-3.1-8b-correlation-qlora
   ```

3. **Adjusting RAG retrieval**: Edit `config.py`:
   ```python
   rag_top_k: int = 10  # Retrieve more examples
   ```

## Next Steps

- Explore the API at http://localhost:8000/docs
- Try different contract pairs
- Train the model on GPU for better predictions
- Adjust RAG settings for your use case
