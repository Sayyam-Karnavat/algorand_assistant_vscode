import json
from typing import List, Dict
from elasticsearch import Elasticsearch
from preprocess import preprocess_text


def load_qa_pairs(file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file with explicit UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_elasticsearch_index(qa_pairs: List[Dict], index_name: str = "qa_pairs"):
    """Index QA pairs in Elasticsearch."""
    es = Elasticsearch(["http://localhost:9200"])
    # Delete existing index if it exists
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    
    # Create index with mapping for preprocessed question
    mapping = {
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "original_question": {"type": "keyword"},
                "answer": {"type": "text"}
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)
    
    # Index QA pairs
    for i, pair in enumerate(qa_pairs):
        raw_question = pair["question"]
        preprocessed_question = preprocess_text(raw_question)
        doc = {
            "question": preprocessed_question,
            "original_question": raw_question,
            "answer": pair["answer"]
        }
        es.index(index=index_name, id=i, body=doc)
        print(f"Indexed Raw Question: {raw_question}")
        print(f"Indexed Preprocessed Question: {preprocessed_question}")
    
    return es, index_name

def elasticsearch_search(query: str, es, index_name: str):
    """Perform Elasticsearch query to search for answers."""
    # Tokenize words in the query
    tokens = [token.text for token in preprocess_text(query)]
    
    # Use Elasticsearch's built-in query functions to retrieve relevant documents
    q = {
        "multi_match": {
            "query": tokens,
            "type": "most_fields"
        }
    }
    res = es.search(index=index_name, body={"query": q})
    
    # Return the top-ranked answer
    return res["hits"]["hits"][0]["answer"], res["hits"]["hits"][0]["score"]

if __name__ == "__main__":
    main()