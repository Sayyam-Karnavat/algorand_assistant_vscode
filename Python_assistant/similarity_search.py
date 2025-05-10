import numpy as np
from typing import List, Dict, Tuple
from preprocess import preprocess_text
from embeddings import get_embeddings
import json


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Args:
        a, b: Input vectors
    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(query: str, qa_pairs: List[Dict]) -> Tuple[str, str, float]:
    """
    Find the best matching QA pair for a user query.
    Args:
        query: User input query
        qa_pairs: List of QA pairs with embeddings
    Returns:
        Tuple of (best question, best answer, similarity score)
    """
    # Preprocess and embed query
    processed_query = preprocess_text(query)
    query_embedding = get_embeddings([processed_query])[0]
    
    best_score = -1
    best_pair = None
    
    # Compute similarity with each question
    for pair in qa_pairs:
        score = cosine_similarity(query_embedding, pair["question_embedding"])
        if score > best_score:
            best_score = score
            best_pair = pair
    
    return best_pair["original_question"], best_pair["original_answer"], best_score

# Example usage
if __name__ == "__main__":
    from preprocess import preprocess_qa_pairs
    from embeddings import embed_qa_pairs

    with open("qa_pairs.json" , "r") as f:
        qa_pairs = json.load(f)
    
    # Preprocess and embed dataset
    preprocessed_data = preprocess_qa_pairs(qa_pairs)
    embedded_data = embed_qa_pairs(preprocessed_data)
    
    # Test query
    query = "What does ARC-0 for in Algorand?"
    question, answer, score = find_best_match(query, embedded_data)
    
    print(f"Query: {query}")
    print(f"Best Match Question: {question}")
    print(f"Answer: {answer}")
    print(f"Similarity Score: {score:.4f}")