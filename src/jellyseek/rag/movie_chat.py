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
from jellyseek.rag.commands import create_command_handler
from jellyseek.rag.database import initialize_database, query_database
from jellyseek.rag.llm import generate_search_query, generate_response
import json
from pathlib import Path

def handle_empty_database(cmd_handler, collection, embedding, collection_name, chroma_client):
    print("\nDatabase exists but contains no movies!")
    if input("Would you like to create a new database? (y/N): ").strip().lower() != 'y':
        print("Cannot proceed without movies in the database. Exiting...")
        return False
        
    print("\nFetching movies from Jellyfin...")
    return cmd_handler.handle("/update",
        collection=collection,
        embedding=embedding,
        collection_name=collection_name,
        chroma_client=chroma_client
    )

def handle_command(cmd_handler, user_query, collection, embedding, collection_name, chroma_client):
    result = cmd_handler.handle(
        user_query,
        collection=collection,
        embedding=embedding,
        collection_name=collection_name,
        chroma_client=chroma_client
    )
    return result is not None and result and user_query == '/quit'

def handle_query(user_query, collection):
    search_query = generate_search_query(user_query)
    retrieved_docs, _ = query_database(collection, search_query)
    
    if not retrieved_docs:
        print("No relevant movies found.")
        return
        
    response = generate_response(user_query, "\n\n".join(retrieved_docs))
    print(f"\nAssistant: {response}")

def chat_loop():
    """Main chat loop"""
    # Initialize components
    chroma_client, collection_name, embedding = initialize_database()
    cmd_handler = create_command_handler()
    
    try:
        # Initialize collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding,
            metadata={"description": "Movies RAG collection"}
        )
        
        if collection.count() == 0:
            if not handle_empty_database(cmd_handler, collection, embedding, collection_name, chroma_client):
                return
        
        print(f"\nFound existing database with {collection.count()} movies.")
            
    except Exception as e:
        print(f"\nError accessing database: {str(e)}")
        return
    
    print("\nMovie Chat Assistant Ready! (Type '/help' for available commands)")
    
    # Main chat loop
    while True:
        user_query = input("\nEnter your question about movies: ").strip()
        
        if user_query.startswith('/'):
            if handle_command(cmd_handler, user_query, collection, embedding, collection_name, chroma_client):
                break
            continue
        
        handle_query(user_query, collection)

if __name__ == "__main__":
    chat_loop()