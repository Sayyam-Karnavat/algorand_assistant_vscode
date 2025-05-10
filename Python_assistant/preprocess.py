import spacy
import re
from typing import List, Dict
import json


# Load spacy model (English, medium-sized for balance of speed and accuracy)
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

def clean_utf8_text(text: str) -> str:
    """
    Remove characters not recognized by UTF-8 encoding from the input text.
    Args:
        text: Input text that may contain invalid UTF-8 characters
    Returns:
        Cleaned text with only valid UTF-8 characters
    """
    # Step 1: Encode to UTF-8 and decode with 'ignore' to remove invalid characters
    cleaned_text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Step 2: Remove non-printable characters (e.g., control characters)
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)
    
    return cleaned_text


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text: remove special characters, stop words, and lemmatize.
    Args:
        text: Input text (question or answer)
    Returns:
        Cleaned, lemmatized text
    """
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Process with spacy
    doc = nlp(text)
    
    # Keep only non-stop words, lemmatized
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
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
        '''
        Show the sample cleaned data 
        '''
        print(f"Processed Question: {pair['question']}")
        print(f"Processed Answer: {pair['answer']}\n")
        break

    # Save the cleaned data back to json file 
    with open("qa_pairs.json" , "w") as f:
        json.dump(preprocessed_data , f)
        