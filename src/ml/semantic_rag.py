"""Semantic RAG using FAISS for vector similarity search."""
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss


class SemanticRAG:
    """FAISS-based semantic retrieval for contract correlations."""

    def __init__(self, index_path: str = "./faiss_index"):
        """
        Initialize semantic RAG.

        Args:
            index_path: Directory containing FAISS index and metadata
        """
        self.index_path = Path(index_path)
        self.index = None
        self.metadata = None
        self.dimension = None

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        index_type: str = "L2"
    ):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Array of embeddings [num_examples, embedding_dim]
            metadata: List of metadata dicts (one per embedding)
            index_type: "L2" for L2 distance or "IP" for inner product (cosine)
        """
        self.dimension = embeddings.shape[1]
        num_vectors = embeddings.shape[0]

        print(f"Building FAISS index with {num_vectors} vectors of dim {self.dimension}")

        # Create index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            # For cosine similarity, normalize embeddings first
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add vectors
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

        print(f"Index built successfully with {self.index.ntotal} vectors")

    def save(self):
        """Save index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save metadata
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)

        print(f"Index saved to {self.index_path}")

    def load(self):
        """Load index and metadata from disk."""
        # Load FAISS index
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.dimension = data['dimension']

        print(f"Index loaded from {self.index_path} with {self.index.ntotal} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_indices: Optional[List[int]] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector [embedding_dim]
            k: Number of neighbors to retrieve
            exclude_indices: Indices to exclude from results (e.g., avoid self-retrieval)

        Returns:
            Tuple of (metadata_list, distances)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() first.")

        # Reshape query
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Search for more results if we need to exclude some
        k_search = k + len(exclude_indices) if exclude_indices else k
        distances, indices = self.index.search(query, k_search)

        # Filter excluded indices
        results = []
        result_distances = []
        exclude_set = set(exclude_indices) if exclude_indices else set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx not in exclude_set and len(results) < k:
                results.append(self.metadata[idx])
                result_distances.append(dist)

        return results, np.array(result_distances)

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> Tuple[List[List[Dict[str, Any]]], np.ndarray]:
        """
        Search for k nearest neighbors for multiple queries.

        Args:
            query_embeddings: Query vectors [batch_size, embedding_dim]
            k: Number of neighbors per query

        Returns:
            Tuple of (list of metadata lists, distances array)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() first.")

        # Search
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)

        # Get metadata for each query
        results = []
        for batch_indices in indices:
            batch_results = [self.metadata[idx] for idx in batch_indices]
            results.append(batch_results)

        return results, distances


def embed_contracts_for_rag(
    model,
    tokenizer,
    contract_a_text: str,
    contract_b_text: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    Get embeddings for a contract pair using Llama.

    Args:
        model: Llama model
        tokenizer: Tokenizer
        contract_a_text: First contract text
        contract_b_text: Second contract text
        device: Device to run on

    Returns:
        Concatenated embedding [embedding_dim * 2]
    """
    def get_embedding(text: str) -> np.ndarray:
        """Get mean-pooled embedding for text."""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get last hidden state and mean pool
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            embedding = hidden_states.mean(dim=1)  # [1, hidden_dim]

        return embedding.cpu().numpy()[0]

    # Get embeddings for both contracts
    emb_a = get_embedding(contract_a_text)
    emb_b = get_embedding(contract_b_text)

    # Concatenate
    return np.concatenate([emb_a, emb_b])
