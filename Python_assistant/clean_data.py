import spacy
import re
from typing import List, Dict

# Load spacy model (English, medium-sized for balance of speed and accuracy)
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  # Disable unused components for speed

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
    qa_pairs = [
        {
            "question": "What is the purpose of ARC-0000 in the Algorand ecosystem?",
            "answer": "ARC-0000 defines the Algorand Request for Comments (ARC) process, outlining how standards are proposed, reviewed, and adopted to ensure interoperability and consistency in Algorand’s blockchain development."
        },
        {
            "question": "What does ARC-0003 specify for Algorand Standard Assets (ASAs)?",
            "answer": "ARC-0003 defines the standard for fungible tokens on Algorand, specifying fields like total supply, decimals, and metadata for asset creation and management."
        }
    ]
    
    preprocessed_data = preprocess_qa_pairs(qa_pairs)
    for pair in preprocessed_data:
        print(f"Processed Question: {pair['question']}")
        print(f"Processed Answer: {pair['answer']}\n")