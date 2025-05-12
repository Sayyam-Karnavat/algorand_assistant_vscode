```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List, Dict
import json

# Load pre-trained Sentence-BERT model from TensorFlow Hub
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(model_url)

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Convert texts to dense vectors using the Sentence-BERT model.
    Args:
        texts: List of preprocessed texts
    Returns:
        Numpy array of embeddings
    """
    return embed_model(texts).numpy()

def preprocess_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for questions and answers.
    Args:
        qa_pairs: List of preprocessed QA pairs
    Returns:
        List of dicts with embeddings added
    """
    questions = [pair["question"] for pair in qa_pairs]
    answers = [pair["answer"] for pair in qa_pairs]
    
    # Generate embeddings
    question_embeddings = generate_embeddings(questions)
    answer_embeddings = generate_embeddings(answers)
    
    # Add embeddings to QA pairs
    for i, pair in enumerate(qa_pairs):
        pair["question_embedding"] = question_embeddings[i]
        pair["answer_embedding"] = answer_embeddings[i]
    
    return qa_pairs

# Example usage
if __name__ == "__main__":

    from preprocess import preprocess_qa_pairs

    try:
        with open("qa_pairs.json" , "r") as f:
            qa_pairs = json.load(f)
    
    except FileNotFoundError:
        print("File not found.")
        exit()
    
    preproccesed_data = preprocess_qa_pairs(qa_pairs)
    
    embedded_data = generate_embeddings(preproccesed_data)
    
    for pair in embedded_data:
        print(f"Question: {pair['original_question']}")
        print(f"Question Embedding Shape: {pair['question_embedding'].shape}\n")
        break
```