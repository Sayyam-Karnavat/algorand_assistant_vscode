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
    """Perform Elasticsearch search with custom scoring."""
    preprocessed_query = preprocess_text(query)
    query_body = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"question": {"query": preprocessed_query, "boost": 1}}},
                    {"match_phrase": {"question": {"query": preprocessed_query, "boost": 2}}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    res = es.search(index=index_name, body=query_body)
    if res["hits"]["hits"]:
        top_hit = res["hits"]["hits"][0]["_source"]
        score = res["hits"]["hits"][0]["_score"]
        print(f"Search Query Tokens: {preprocessed_query.split()}")
        print(f"Matched Preprocessed Question: {top_hit['question']}")
        return top_hit["original_question"], top_hit["answer"], score
    return None, "No match found", 0.0

def main():
    qa_file = "qa_pairs.json"
    qa_pairs = load_qa_pairs(qa_file)
    
    # Setup Elasticsearch index
    es, index_name = setup_elasticsearch_index(qa_pairs)
    
    # Print sample QA pair
    print("Sample QA Pair:", qa_pairs[0])
    
    # Interactive loop
    print("Enter your query (or 'quit' to exit):")
    while True:
        query = input("> ")
        if query.lower() == 'quit':
            break
        
        question, answer, score = elasticsearch_search(query, es, index_name)
        print(f"\nMatched Question: {question}")
        print(f"Answer: {answer}")
        print(f"Score: {score:.4f}\n")

if __name__ == "__main__":
    main()