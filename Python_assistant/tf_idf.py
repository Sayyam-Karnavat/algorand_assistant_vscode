import json
import re
import spacy
import unicodedata
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text


def load_qa_pairs(file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file with explicit UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_tfidf_search(qa_pairs: List[Dict]):
    """Set up TF-IDF vectorizer and matrix for questions."""
    questions = []
    for pair in qa_pairs:
        raw_question = pair["question"]
        preprocessed_question = preprocess_text(raw_question)
        questions.append(preprocessed_question)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix, questions, qa_pairs

def tfidf_search(query: str, vectorizer, tfidf_matrix, questions, qa_pairs):
    """Perform TF-IDF search with cosine similarity."""
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_idx = similarities.argmax()
    return qa_pairs[top_idx]["question"], qa_pairs[top_idx]["answer"], similarities[top_idx]

def main():
    qa_file = "qa_pairs.json"
    qa_pairs = load_qa_pairs(qa_file)
    
    # Setup TF-IDF search
    vectorizer, tfidf_matrix, questions, qa_pairs = setup_tfidf_search(qa_pairs)
    
    # Print sample QA pair
    print("Sample QA Pair:", qa_pairs[0])
    
    # Interactive loop
    print("Enter your query (or 'quit' to exit):")
    while True:
        query = input("> ")
        if query.lower() == 'quit':
            break
        
        question, answer, score = tfidf_search(query, vectorizer, tfidf_matrix, questions, qa_pairs)
        print(f"\nMatched Question: {question}")
        print(f"Answer: {answer}")
        print(f"Score: {score:.4f}\n")

if __name__ == "__main__":
    main()