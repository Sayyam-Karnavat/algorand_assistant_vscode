import json
import pickle
from typing import List, Dict
from preprocess import preprocess_qa_pairs
from embeddings import embed_qa_pairs
from similarity_search import find_best_match

def load_qa_pairs(file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_embeddings(qa_pairs: List[Dict], file_path: str):
    """Save embedded QA pairs to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(qa_pairs, f)

def load_embeddings(file_path: str) -> List[Dict]:
    """Load embedded QA pairs from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    # Load QA pairs
    qa_file = "qa_pairs.json"
    qa_pairs = load_qa_pairs(qa_file)
    
    # Check if embeddings exist, else compute
    embeddings_file = "qa_embeddings.pkl"
    try:
        embedded_data = load_embeddings(embeddings_file)
    except FileNotFoundError:
        preprocessed_data = preprocess_qa_pairs(qa_pairs)
        embedded_data = embed_qa_pairs(preprocessed_data)
        save_embeddings(embedded_data, embeddings_file)
    
    # Interactive loop
    print("Enter your query (or 'quit' to exit):")
    while True:
        query = input("> ")
        if query.lower() == 'quit':
            break
        
        question, answer, score = find_best_match(query, embedded_data)

        if score > 0.70:

            print(f"\nBest Match Question: {question}")
            print(f"Answer: {answer}")
            print(f"Similarity Score: {score:.4f}\n")
        else:
            print("No relevant answer found for your query.")

if __name__ == "__main__":
    main()