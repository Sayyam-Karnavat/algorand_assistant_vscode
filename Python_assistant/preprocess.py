import spacy
import re
from typing import List, Dict
import json
import unicodedata

# Load spacy model (English, medium-sized for balance of speed and accuracy)
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  # Disable unused components for speed

def clean_utf8_text(text: str) -> str:
    """
    Remove invalid UTF-8 characters and normalize Unicode characters.
    Args:
        text: Input text that may contain invalid UTF-8 or special characters
    Returns:
        Cleaned text with normalized characters
    """
    # Replace curly apostrophes and quotes with straight ones
    text = text.replace("’", "'").replace("'", "'").replace("'", "'")
    # Normalize Unicode to ASCII
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    # Remove control characters
    text = "".join(c for c in text if unicodedata.category(c) != "Cc")
    return text

def normalize_number(text: str) -> str:
    """
    Normalize numeric tokens by stripping leading zeros.
    Args:
        text: Input text to be processed
    Returns:
        Processed text with normalized numeric tokens
    """
    return re.sub(r"(\d+)", lambda m: m.group(1).zfill(len(m.group())), text)

def preprocess_text(text: str) -> str:
    """
    Preprocess text data for NLP tasks.
    Args:
        text: Input text to be processed
    Returns:
        Processed text with normalized characters, lemmatized words, and removed stop words
    """
    # Normalize Unicode characters
    text = clean_utf8_text(text)
    # Tokenize text
    tokens = nlp.tokenizer.tokens_from_list(text.split())
    # Lemmatize words
    lemmatized_tokens = [token.lemma_ for token in tokens]
    # Remove stop words
    stop_words = set(nlp.Defaults.stop_words)
    filtered_lemmas = [lemma for lemma in lemmatized_tokens if lemma not in stop_words]
    return " ".join(filtered_lemmas)

def preprocess_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Preprocess all questions and answers in the dataset.
    Args:
        qa_pairs: List of dicts with 'question' and 'answer' keys
    Returns:
        List of dicts with preprocessed 'question' and 'answer'
    """
    preprocessed_pairs = []
    for pair in qa_pairs:
        preprocessed_pairs.append({
            "question": preprocess_text(pair["question"]),
            "answer": preprocess_text(pair["answer"]),
            "original_question": pair["question"],  # Keep original for display
            "original_answer": pair["answer"]
        })
    return preprocessed_pairs

# Example usage
if __name__ == "__main__":

    with open("qa_pairs.json" , "r") as f:
        qa_pairs = json.load(f)

    preprocessed_data = preprocess_qa_pairs(qa_pairs)
    for pair in preprocessed_data:
        print(f"Processed Question: {pair['question']}")
        print(f"Processed Answer: {pair['answer']}\n")
        break

    with open("qa_pairs.json" , "w") as f:
        json.dump(preprocessed_data , f)