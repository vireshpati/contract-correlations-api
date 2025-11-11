"""Build FAISS index from contract correlations for semantic RAG."""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.training import load_training_data
from src.ml.semantic_rag import SemanticRAG, embed_contracts_for_rag
from src.config import get_settings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def build_index():
    """Build FAISS index from all contract correlations."""
    print("=" * 80)
    print("Building FAISS Index for Semantic RAG")
    print("=" * 80)

    settings = get_settings()

    # Load all training data
    print("\n1. Loading training data...")
    all_data = load_training_data()
    print(f"   Loaded {len(all_data)} contract pairs")

    # Load Llama for embeddings
    print("\n2. Loading Llama 3.1 8B for embedding extraction...")
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
        cache_dir=settings.model_cache_dir,
        torch_dtype=torch.float16,
    )
    model.eval()

    print("   Model loaded successfully")

    # Compute embeddings for all pairs
    print("\n3. Computing embeddings for all contract pairs...")
    embeddings = []
    metadata = []

    for i, example in enumerate(all_data):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(all_data)}")

        messages = example["messages"]
        user_msg = messages[1]["content"]

        # Extract contract texts
        lines = user_msg.split('\n')
        contract_a = lines[1].replace("Contract A: ", "")
        contract_b = lines[2].replace("Contract B: ", "")

        # Get ground truth correlation
        assistant_msg = messages[2]["content"]
        try:
            import json
            corr_data = eval(assistant_msg)
            true_correlation = corr_data["underlying_correlation"]
        except:
            true_correlation = 0.0

        # Compute embedding
        embedding = embed_contracts_for_rag(
            model, tokenizer, contract_a, contract_b, device="cuda"
        )

        embeddings.append(embedding)
        metadata.append({
            "index": i,
            "contract_a": contract_a,
            "contract_b": contract_b,
            "correlation": true_correlation,
            "full_example": example
        })

    embeddings = np.array(embeddings)
    print(f"\n   Computed {len(embeddings)} embeddings of shape {embeddings.shape}")

    # Build FAISS index
    print("\n4. Building FAISS index...")
    rag = SemanticRAG(index_path="./faiss_index")
    rag.build_index(embeddings, metadata, index_type="L2")

    # Save index
    print("\n5. Saving index to disk...")
    rag.save()

    print("\n" + "=" * 80)
    print("FAISS index built successfully!")
    print(f"   Index location: ./faiss_index/")
    print(f"   Total vectors: {len(embeddings)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print("=" * 80)


if __name__ == "__main__":
    build_index()
