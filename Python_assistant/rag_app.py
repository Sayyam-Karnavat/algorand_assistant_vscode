import json
import re
from typing import List, Dict
from langchain_ollama import OllamaEmbeddings ,OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

def clean_utf8_text(text: str) -> str:
    """
    Remove invalid UTF-8 characters and non-essential special characters, preserve identifiers.
    Args:
        text: Input text
    Returns:
        Cleaned text
    """
    # Remove invalid UTF-8 characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # Remove non-printable characters, keep letters, numbers, spaces, and hyphens
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]|[^a-zA-Z0-9\s-]', '', text.lower())
    return text.strip()

def load_qa_pairs(file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_documents(qa_pairs: List[Dict]) -> List[Document]:
    """
    Convert QA pairs to LangChain Documents with cleaned questions and metadata.
    Args:
        qa_pairs: List of QA pairs
    Returns:
        List of Document objects
    """
    documents = []
    for pair in qa_pairs:
        cleaned_question = clean_utf8_text(pair["question"])
        # Store original question and answer in metadata
        doc = Document(
            page_content=cleaned_question,
            metadata={
                "original_question": pair["question"],
                "answer": pair["answer"]
            }
        )
        documents.append(doc)
    return documents

def create_vector_store(documents: List[Document]) -> FAISS:
    """
    Create FAISS vector store with Ollama embeddings.
    Args:
        documents: List of Document objects
    Returns:
        FAISS vector store
    """
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def setup_rag_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Set up RAG chain with LLaMA3 LLM and custom prompt.
    Args:
        vector_store: FAISS vector store
    Returns:
        RetrievalQA chain
    """
    llm = OllamaLLM(model="llama3")
    
    # Custom prompt to ensure precise answer generation
    prompt_template = """You are an expert on the Algorand blockchain. Given the following question and its answer from the dataset, provide a concise and accurate response based on the provided answer. Do not add external information.

    Question: {question}
    Answer from dataset: {answer}

    Response: """
    
    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=prompt_template
    )
    
    # Set up RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),  # Retrieve top 1 match
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    # Load QA pairs
    qa_file = "qa_pairs.json"
    qa_pairs = load_qa_pairs(qa_file)
    
    # Prepare documents
    documents = prepare_documents(qa_pairs)
    
    # Create vector store
    vector_store = create_vector_store(documents)
    
    # Save vector store for reuse
    vector_store.save_local("faiss_index")
    
    # Set up RAG chain
    qa_chain = setup_rag_chain(vector_store)
    
    # Interactive query loop
    print("Enter your query (or 'quit' to exit):")
    while True:
        query = input("> ")
        if query.lower() == 'quit':
            break
        
        # Clean query
        cleaned_query = clean_utf8_text(query)
        
        # Run RAG chain
        result = qa_chain.invoke({"query": cleaned_query})
        
        # Extract answer and source
        answer = result["result"]
        source = result["source_documents"][0]
        
        print(f"\nMatched Question: {source['original_question']}")
        print(f"Answer: {answer}")
        print(f"Dataset Answer: {source['answer']}\n")

if __name__ == "__main__":
    main()