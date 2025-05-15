from langchain_ollama import OllamaLLM, OllamaEmbeddings
import chromadb
import os
from typing import Tuple, List, Dict
from config import (
    OLLAMA_BASE_URL, 
    EMBEDDING_MODEL, 
    GENERATION_MODEL, 
    EMBEDDING_PROMPT, 
    GENERATION_PROMPT,
    CHROMADB_PATH
)

# Model configurations
embedding_model = EMBEDDING_MODEL
generation_model = GENERATION_MODEL
ollama_url = OLLAMA_BASE_URL

# Custom embedding function for ChromaDB
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

def query_chromadb(collection, query_text: str, n_results: int = 10) -> Tuple[List[str], List[Dict]]:
    """Query the ChromaDB collection for relevant documents."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

def read_prompt_file(filename: str) -> str:
    """Read prompt template from file"""
    with open(filename, 'r') as file:
        return file.read().strip()

def generate_search_query(original_query: str) -> str:
    """Generate an optimized search query from the original user query"""
    template = read_prompt_file(EMBEDDING_PROMPT)
    prompt = template.format(
        generation_model=generation_model,
        embedding_model=embedding_model,
        question=original_query
    )
    
    llm = OllamaLLM(model=generation_model, base_url=ollama_url)
    return llm.invoke(prompt).strip()

def generate_final_response(original_query: str, context: str) -> str:
    """Generate final response using original query and retrieved context"""
    template = read_prompt_file(GENERATION_PROMPT)
    prompt = template.format(
        generation_model=generation_model,
        embedding_model=embedding_model,
        context=context,
        question=original_query
    )
    
    llm = OllamaLLM(model=generation_model, base_url=ollama_url)
    return llm.invoke(prompt)

def chat_loop():
    """Main chat loop"""
    # Initialize ChromaDB client with configured path
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    
    # Initialize embedding function
    embedding = ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
    )
    
    # Get collection with embedding function
    collection = chroma_client.get_collection(
        name="movies_rag",
        embedding_function=embedding
    )
    
    print("Movie Chat Assistant Ready! (Type '/quit' to exit)")
    
    while True:
        user_query = input("\nEnter your question about movies: ").strip()
        if user_query.lower() == '/quit':
            break
            
        # Step 1: Generate optimized search query
        search_query = generate_search_query(user_query)
        #print("\n######## Generated Search Query ########")
        #print(search_query)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs, metadata = query_chromadb(collection, search_query)
        context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        #print("\n######## Retrieved Context ########")
        #print(context)
        
        # Step 3: Generate final response
        response = generate_final_response(user_query, context)
        #print("\n######## Final Response ########")
        print(response)

if __name__ == "__main__":
    chat_loop()