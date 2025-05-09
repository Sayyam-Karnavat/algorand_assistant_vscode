from sentence_transformers import SentenceTransformer , util
import numpy as np
import json
import os
import sys


VECTOR_DIR = "extention/data"

def text_to_vector(text_file="arc_standards.txt" ):

    

    if not os.path.exists(VECTOR_DIR):
        os.makedirs(VECTOR_DIR)

    with open( text_file, "r" , encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    embeddings = model.encode(documents)


    np.save(os.path.join(VECTOR_DIR,"embeddings.npy") , embeddings)
    with open(os.path.join(VECTOR_DIR,"documents.json") , "w") as f:
        json.dump(documents , f)


def load_knowledge_vector():
    with open(os.path.join(VECTOR_DIR,'documents.json'), 'r') as f:
        documents = json.load(f)
    embeddings = np.load(os.path.join(VECTOR_DIR,'embeddings.npy'))


    return embeddings , documents

def query(text):
    query_embedding = model.encode([text])[0]
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = scores.argmax()
    return documents[best_idx]


if __name__ == "__main__":
    
    model = SentenceTransformer("all-MiniLM-L6-v2")

    text_to_vector()
    embeddings , documents = load_knowledge_vector()

    while True:

        user_query = input("Query :- ")
        response = query(text=user_query)

        if user_query == "quit":
            sys.exit(0)

        print(response)