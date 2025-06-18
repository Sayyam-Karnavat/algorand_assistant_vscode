import numpy as np
from typing import List, Dict, Tuple
from preprocess import preprocess_text
from embeddings import get_embeddings
import json

def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix of document vectors.
    Args:
        query_vec: Query vector of shape (D,)
        doc_matrix: Matrix of document vectors of shape (N, D)
    Returns:
        Array of cosine similarity scores of shape (N,)
    """
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    dot_products = np.dot(doc_matrix, query_vec)
    return dot_products / (doc_norms * query_norm + 1e-10)  # Add epsilon to avoid division by zero

def find_best_match(query: str, qa_pairs: List[Dict]) -> Tuple[str, str, float]:
    """
    Find the best matching QA pair for a user query.
    Args:
        query: User input query
        qa_pairs: List of QA pairs with embeddings
    Returns:
        Tuple of (best question, best answer, similarity score)
    """
    processed_query = preprocess_text(query)
    query_embedding = get_embeddings([processed_query])[0]

    # Extract question embeddings into a matrix
    embeddings_matrix = np.array([pair["question_embedding"] for pair in qa_pairs])
    similarity_scores = cosine_similarity_matrix(query_embedding, embeddings_matrix)
    
    best_idx = int(np.argmax(similarity_scores))
    best_pair = qa_pairs[best_idx]
    best_score = float(similarity_scores[best_idx])
    
    return best_pair["original_question"], best_pair["original_answer"], best_score

# Example usage
if __name__ == "__main__":
    from preprocess import preprocess_qa_pairs
    from embeddings import embed_qa_pairs

    with open("qa_pairs.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Preprocess and embed dataset if embeddings are missing
    if "question_embedding" not in qa_pairs[0]:
        preprocessed_data = preprocess_qa_pairs(qa_pairs)
        qa_pairs = embed_qa_pairs(preprocessed_data)

    # Test query
    query = "What does ARC-0 mean in Algorand?"
    question, answer, score = find_best_match(query, qa_pairs)

    print(f"\nQuery: {query}")
    print(f"Best Match Question: {question}")
    print(f"Answer: {answer}")
    print(f"Similarity Score: {score:.4f}")
