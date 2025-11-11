"""Semantic RAG system for contract correlation prediction."""
import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle


def embed_contracts(model, tokenizer, contract_a: str, contract_b: str, device: str = "cuda") -> np.ndarray:
    """
    Compute embedding for a contract pair using Llama model.

    Args:
        model: Llama model
        tokenizer: Model tokenizer
        contract_a: First contract description
        contract_b: Second contract description
        device: Device to run on

    Returns:
        Embedding vector as numpy array
    """
    prompt = f"Contract A: {contract_a}\nContract B: {contract_b}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use mean of last hidden state as embedding
        embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

    return embedding


class SemanticRAG:
    """Semantic RAG using FAISS for contract correlation prediction."""

    def __init__(self, index_path: str = "./faiss_index"):
        """Initialize RAG system."""
        self.index_path = Path(index_path)
        self.index = None
        self.metadata = None

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], index_type: str = "L2"):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            metadata: List of metadata dicts for each embedding
            index_type: "L2" or "IP" (inner product)
        """
        dimension = embeddings.shape[1]

        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Normalize for IP similarity
        if index_type == "IP":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar contract pairs.

        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to return

        Returns:
            List of similar examples with metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # Valid index
                result = self.metadata[idx].copy()
                result["distance"] = float(dist)
                results.append(result)

        return results

    def save(self):
        """Save index and metadata to disk."""
        self.index_path.mkdir(exist_ok=True)

        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

        with open(self.index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """Load index and metadata from disk."""
        if not self.index_path.exists():
            raise ValueError(f"Index path does not exist: {self.index_path}")

        self.index = faiss.read_index(str(self.index_path / "index.faiss"))

        with open(self.index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def format_rag_context(self, similar_examples: List[Dict[str, Any]], max_examples: int = 3) -> str:
        """
        Format similar examples as RAG context.

        Args:
            similar_examples: List of similar examples from search
            max_examples: Maximum number of examples to include

        Returns:
            Formatted context string
        """
        context_parts = ["Here are similar contract pairs and their correlations:"]

        for i, example in enumerate(similar_examples[:max_examples], 1):
            context_parts.append(f"\nExample {i}:")
            context_parts.append(f"  Contract A: {example['contract_a']}")
            context_parts.append(f"  Contract B: {example['contract_b']}")
            context_parts.append(f"  Correlation: {example['correlation']:.2f}")

        return "\n".join(context_parts)
