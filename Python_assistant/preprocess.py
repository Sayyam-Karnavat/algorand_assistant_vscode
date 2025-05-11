import spacy
import re
from typing import List, Dict
import json

# Load spacy model (English, medium-sized for balance of speed and accuracy)
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  # Disable unused components for speed

def clean_utf8_text(text: str) -> str:
    """
    Remove characters not recognized by UTF-8 encoding from the input text.
    Args:
        text: Input text that may contain invalid UTF-8 characters
    Returns:
        Cleaned text with only valid UTF-8 characters
    """
    cleaned_text = text.encode('utf-8', errors='ignore').decode('utf-8')
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)
    return cleaned_text

def normalize_number(token: str) -> str:
    """
    Normalize numeric tokens by stripping leading zeros.
    Args:
        token: Input token (e.g., '0000', '0069')
    Returns:
        Normalized number (e.g., '0', '69')
    """
    if token.isdigit():
        stripped = token.lstrip('0')
        return stripped if stripped else '0'  # Return '0' if all zeros, else stripped number
    return token

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text: split identifiers (e.g., arc-0000), normalize numbers, remove stop words, and lemmatize.
    Args:
        text: Input text (question or answer)
    Returns:
        Cleaned, lemmatized text with numbers preserved and normalized
    """
    # Clean UTF-8 invalid characters
    text = clean_utf8_text(text)
    
    # Remove special characters except letters, numbers, and spaces; replace hyphens with spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().replace('-', ' '))
    
    # Process with spacy
    doc = nlp(text)
    
    # Keep words, lemmatized, including tokens with numbers
    tokens = []
    for token in doc:
        # Keep tokens that are alphabetic, numeric, or alphanumeric
        if token.is_alpha or token.is_digit or token.text.isalnum():
            # Normalize numbers (e.g., '0000' -> '0', '0069' -> '69')
            normalized_token = normalize_number(token.text)
            # Lemmatize non-numeric tokens, keep normalized numbers as-is
            tokens.append(token.lemma_ if token.is_alpha or token.text.isalnum() and not token.is_digit else normalized_token)
    
    return ' '.join(tokens)

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