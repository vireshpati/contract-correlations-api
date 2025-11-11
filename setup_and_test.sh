#!/bin/bash
# Setup and test script for Contract Correlation API

echo "========================================"
echo "Contract Correlation API - Setup & Test"
echo "========================================"

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run unit tests
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --tb=short

# Test database connection
echo ""
echo "Testing database connection..."
python examples/test_database.py

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To start the API server:"
echo "  uvicorn src.main:app --reload"
echo ""
echo "To run example predictions:"
echo "  python examples/predict_correlation.py"
echo ""
