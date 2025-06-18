from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import os
import sys

VECTOR_DIR = "extension/data"
TEXT_FILE = "arc_standards.txt"
EMBEDDING_FILE = os.path.join(VECTOR_DIR, "embeddings.npy")
DOCUMENT_FILE = os.path.join(VECTOR_DIR, "documents.json")

def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def text_to_vector(text_file=TEXT_FILE, model=None):
    os.makedirs(VECTOR_DIR, exist_ok=True)

    with open(text_file, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    np.save(EMBEDDING_FILE, embeddings)
    with open(DOCUMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)

def load_knowledge_vector():
    if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(DOCUMENT_FILE):
        raise FileNotFoundError("Embedding or document file not found. Run `text_to_vector()` first.")
    
    embeddings = np.load(EMBEDDING_FILE)
    with open(DOCUMENT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    return embeddings, documents

def query(text, model, embeddings, documents, top_k=1):
    query_embedding = model.encode(text, convert_to_numpy=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_k_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = [(documents[idx], float(scores[idx])) for idx in top_k_indices]
    return results

def main():
    model = get_model()

    if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(DOCUMENT_FILE):
        print("Generating embeddings...")
        text_to_vector(TEXT_FILE, model)

    embeddings, documents = load_knowledge_vector()

    print("RAG System Ready. Type 'quit' to exit.")
    while True:
        user_query = input("\nQuery: ").strip()
        if user_query.lower() in ("quit", "exit"):
            print("Exiting...")
            break

        results = query(user_query, model, embeddings, documents)
        best_result, score = results[0]
        print(f"\nTop Match (Score: {score:.4f}):\n{best_result}")

if __name__ == "__main__":
    main()
