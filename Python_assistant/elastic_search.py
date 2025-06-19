```
import json
from typing import List, Dict
from elasticsearch import Elasticsearch
from preprocess import preprocess_text
import nltk
nltk.download('punkt')

def load_qa_pairs(file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file with explicit UTF-8 encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def setup_elasticsearch_index(qa_pairs: List[Dict], index_name: str = "qa_pairs"):
    """Index QA pairs in Elasticsearch."""
    es = Elasticsearch(["http://localhost:9200"])
    if es.indices.exists(index=index_name):
        try:
            es.indices.delete(index=index_name)
        except Exception as e:
            print(f"Error deleting existing index: {e}")
    
    mapping = {
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "original_question": {"type": "keyword"},
                "answer": {"type": "text"}
            }
        }
    }
    try:
        es.indices.create(index=index_name, body=mapping)
    except Exception as e:
        print(f"Error creating index: {e}")
    
    for i, pair in enumerate(qa_pairs):
        raw_question = pair["question"]
        preprocessed_question = preprocess_text(raw_question)
        doc = {
            "question": preprocessed_question,
            "original_question": raw_question,
            "answer": pair["answer"]
        }
        try:
            es.index(index=index_name, id=i, body=doc)
            print(f"Indexed Raw Question: {raw_question}")
            print(f"Indexed Preprocessed Question: {preprocessed_question}")
        except Exception as e:
            print(f"Error indexing data: {e}")
    
    return es, index_name

def elasticsearch_search(query: str, es, index_name: str):
    tokens = nltk.word_tokenize(query)
    
    q = {
        "multi_match": {
            "query": {"match": {"question": tokens}},
            "type": "most_common"
        }
    }
    res = es.search(index=index_name, body={"query": q})
    
    if not res["hits"]["hits"]:
        return None
    else:
        return res["hits"]["hits"][0]["_source"]["answer"], res["hits"]["hits"][0]["_score"]

if __name__ == "__main__":
    try:
        es, index_name = setup_elasticsearch_index(load_qa_pairs("qa_pairs.json"))
        print(elasticsearch_search("What is Algorand?", es, index_name))
    except Exception as e:
        print(f"Error: {e}")
```