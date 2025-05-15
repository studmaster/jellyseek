from langchain_ollama import OllamaEmbeddings
import chromadb
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from collections import OrderedDict
import re, unicodedata, uuid
from config import (
    OLLAMA_BASE_URL, 
    EMBEDDING_MODEL, 
    GENERATION_MODEL, 
    CHROMADB_PATH,
    JELLYFIN_DATA_PATH
)

# Define the embedding model
embedding_model = EMBEDDING_MODEL
ollama_url = OLLAMA_BASE_URL

# ...existing ChromaDBEmbeddingFunction class...
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

def load_movie_json(json_file: Path):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out movies with any null or missing categories
    required_fields = ["Title", "Plot", "Genres", "Tags", "Actors", "PremiereDate"]
    data = [
        item for item in data
        if all(item.get(field) not in (None, "") for field in required_fields)
    ]

    # ---------- DEDUP SECTION ----------
    unique: "OrderedDict[str, dict]" = OrderedDict()   # keeps first occurrence
    for item in data:
        title = item.get("Title", "").strip()
        year  = ""
        if date_str := item.get("PremiereDate"):
            try:
                year = str(datetime.fromisoformat(date_str.rstrip("Z")).year)
            except ValueError:
                pass
        dedup_key = f"{slug(title)}:{year}"
        unique.setdefault(dedup_key, item)   # ignore later duplicates
    # -----------------------------------

    documents, ids, metadatas = [], [], []
    for item in unique.values():            # iterate **de-duplicated** list
        title = item.get("Title", "Unknown Title")
        plot  = item.get("Plot", "")
        year_from = ""             
        
        if date_str := item.get("PremiereDate"):
            try:
                year_from = str(datetime.fromisoformat(date_str.rstrip("Z")).year)
            except ValueError:
                pass

        doc_text = (
            f"Title: {title}\n"
            f"Year: {year_from}\n"
            f"Genres: {', '.join(item.get('Genres', []))}\n"
            f"Tags: {', '.join(item.get('Tags', []))}\n"
            f"Actors: {', '.join(item.get('Actors', [])[:5])}\n"
            f"Critic Rating: {item.get('CriticRating', 'Not Rated')}\n"
            f"Official Rating: {item.get('OfficialRating', 'Not Rated')}\n"
            f"Runtime: {item.get('RuntimeMinutes', 'Unknown')} minutes\n"
            f"Plot: {plot}"
        )

        documents.append(doc_text)
        ids.append(f"{slug(title)}_{year_from}" or uuid.uuid4().hex)
        raw_meta = {
            "title": title,
            "year": int(year_from) if year_from else 0,
            "genres": ", ".join(item.get("Genres", [])),
            "critic_rating": item.get("CriticRating"),
            "official_rating": item.get("OfficialRating"),
            "runtime_minutes": item.get("RuntimeMinutes")  # Added runtime to metadata
        }

        metadatas.append(clean_metadata(raw_meta))
    return documents, ids, metadatas


def slug(s: str) -> str:
    """lower-case, accent-fold, replace non-alphanum with underscores"""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def clean_metadata(d: dict) -> dict:
    """
    Return a copy containing only keys whose values are
    str | int | float | bool and not None.
    """
    out = {}
    for k, v in d.items():
        if v is None:
            continue                      # skip entirely
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)               # final safety net
    return out

def generate_database(force_update: bool = False):
    """Main function to generate the vector database"""
    # Initialize ChromaDB client with configured path
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    
    collection_name = "movies_rag"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        if not force_update:
            doc_count = collection.count()
            print(f"\nFound existing database with {doc_count} documents.")
            print("Options:")
            print("1. Delete existing database and create new one")
            print("2. Keep existing database and exit")
            
            while True:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice in ['1', '2']:
                    break
                print("Invalid choice. Please enter 1 or 2.")
            
            if choice == '2':
                print("Keeping existing database. Exiting...")
                return
        
        print("\nDeleting existing database...")
        chroma_client.delete_collection(collection_name)
        print("Database deleted successfully.")
    except ValueError:
        # Collection doesn't exist yet
        print("\nNo existing database found. Creating new database...")
    
    # Initialize embedding function
    embedding = ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
    )
    
    # Create new collection
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "A collection for movies rag"},
        embedding_function=embedding
    )
    
    # Load and process movies from configured data path
    json_path = Path(JELLYFIN_DATA_PATH) / 'jellyfin_items.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Movie data not found at: {json_path}. Please run the jellyfin_export script first.")
    
    documents, doc_ids, metadatas = load_movie_json(json_path)
    
    # Add to collection
    collection.add(documents=documents, ids=doc_ids, metadatas=metadatas)
    print(f"Successfully added {len(documents)} movies to the database.")

if __name__ == "__main__":
    generate_database()