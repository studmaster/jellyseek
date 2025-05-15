from langchain_ollama import OllamaLLM, OllamaEmbeddings
import chromadb
import os
from typing import Tuple, List, Dict
from jellyseek.rag.config import (
    OLLAMA_BASE_URL, 
    EMBEDDING_MODEL, 
    GENERATION_MODEL, 
    EMBEDDING_PROMPT, 
    GENERATION_PROMPT,
    CHROMADB_PATH,
    JELLYFIN_DATA_PATH
)
from jellyseek.jellyfin_export.main import fetch_items, save_items
from jellyseek.rag.db_generator import generate_database
import json
from pathlib import Path

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

def check_for_updates() -> bool:
    """
    Fetch new data from Jellyfin and check for updates.
    Returns True if updates were found and applied.
    """
    print("\nChecking for updates...")
    
    # Fetch new items from Jellyfin
    new_items = fetch_items()
    if not new_items:
        print("Failed to fetch items from Jellyfin")
        return False
        
    # Load existing items
    existing_file = Path(JELLYFIN_DATA_PATH) / 'jellyfin_items.json'
    if existing_file.exists():
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_items = json.load(f)
            
        # Compare item counts
        new_count = len(new_items.get('Items', []))
        existing_count = len(existing_items.get('Items', []))
        
        if new_count <= existing_count:
            print("No new items found.")
            return False
            
        print(f"Found {new_count - existing_count} new items!")
    
    # Save new items and regenerate database
    save_items(new_items)
    print("Saved new items, regenerating database...")
    generate_database(force_update=True)
    return True

def chat_loop():
    """Main chat loop"""
    # Initialize ChromaDB client with configured path
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection_name = "movies_rag"
    
    # Initialize embedding function
    embedding = ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
    )
    
    # Check if collection exists
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding,
            metadata={"description": "Movies RAG collection"}
        )
        
        # If collection exists but is empty, ask to create database
        if collection.count() == 0:
            print("\nDatabase exists but contains no movies!")
            while True:
                choice = input("Would you like to create a new database? (y/N): ").strip().lower()
                if choice in ['y', 'n', '']:
                    break
                print("Invalid choice. Please enter 'y' or 'n'")
            
            if choice != 'y':
                print("Cannot proceed without movies in the database. Exiting...")
                return
                
            print("\nFetching movies from Jellyfin...")
            if not check_for_updates():
                print("Failed to create database. Exiting...")
                return
        else:
            print(f"\nFound existing database with {collection.count()} movies.")
            
    except Exception as e:
        print(f"\nError accessing database: {str(e)}")
        print("Cannot proceed. Exiting...")
        return
    
    print("\nMovie Chat Assistant Ready! (Type '/quit' to exit, '/update' to check for new movies)")
    
    while True:
        user_query = input("\nEnter your question about movies: ").strip()
        
        if user_query.lower() == '/quit':
            break
        elif user_query.lower() == '/update':
            if check_for_updates():
                # Refresh collection after update
                collection = chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=embedding
                )
            continue
            
        # Step 1: Generate optimized search query
        search_query = generate_search_query(user_query)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs, metadata = query_chromadb(collection, search_query)
        context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        
        # Step 3: Generate final response
        response = generate_final_response(user_query, context)
        print(response)

if __name__ == "__main__":
    chat_loop()