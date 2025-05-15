from langchain_ollama import OllamaEmbeddings
import chromadb
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from collections import OrderedDict
import re, unicodedata, uuid
from jellyseek.rag.config import (
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
    try:
        with json_file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        data = raw_data.get("Items", [])
        if not data:
            raise ValueError("No movie items found in the JSON file")

        print(f"Found {len(data)} total items")

        # Create unique movie entries
        unique: "OrderedDict[str, dict]" = OrderedDict()
        for item in data:
            if not isinstance(item, dict):
                print(f"Skipping non-dict item: {type(item)}")
                continue
                
            # Use Name instead of Title for Jellyfin API
            title = item.get("Name")
            if not title:
                print(f"Skipping item without name: {item}")
                continue

            year = ""
            if date_str := item.get("PremiereDate"):
                try:
                    year = str(datetime.fromisoformat(date_str.rstrip("Z")).year)
                except ValueError:
                    pass
            
            key = f"{slug(str(title))}:{year}"
            unique[key] = item

        print(f"\nFound {len(unique)} unique movies after deduplication")

        documents, ids, metadatas = [], [], []
        for item in unique.values():
            # Use Name instead of Title throughout
            title = str(item.get("Name", "Unknown")).strip()
            plot = str(item.get("Overview", "No plot available"))  # Also changed Plot to Overview
            year_from = ""
            
            if date_str := item.get("PremiereDate"):
                try:
                    year_from = str(datetime.fromisoformat(date_str.rstrip("Z")).year)
                except ValueError:
                    pass

            doc_text = (
                f"Title: {title}\n"
                f"Year: {year_from or 'Unknown'}\n"
                f"Genres: {', '.join(map(str, item.get('Genres', []) or ['Unknown']))}\n"
                f"Tags: {', '.join(map(str, item.get('Tags', []) or ['None']))}\n"
                f"Actors: {', '.join(map(str, (item.get('Actors', []) or [])[:5]))}\n"
                f"Critic Rating: {item.get('CriticRating', 'Not Rated')}\n"
                f"Official Rating: {item.get('OfficialRating', 'Not Rated')}\n"
                f"Runtime: {item.get('RunTimeTicks', 'Unknown')} minutes\n"
                f"Plot: {plot}"
            )

            documents.append(doc_text)
            ids.append(f"{slug(title)}_{year_from or uuid.uuid4().hex}")
            
            raw_meta = {
                "title": title,
                "year": int(year_from) if year_from else 0,
                "genres": ", ".join(map(str, item.get("Genres", []))),
                "critic_rating": item.get("CriticRating"),
                "official_rating": item.get("OfficialRating"),
                "runtime_minutes": item.get("RunTimeTicks")  # Changed to RunTimeTicks
            }
            metadatas.append(clean_metadata(raw_meta))

        print(f"Successfully processed {len(documents)} movies")
        return documents, ids, metadatas

    except Exception as e:
        print(f"Error processing movie data: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

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
    try:
        # Initialize ChromaDB client with configured path
        chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
        
        collection_name = "movies_rag"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            if not force_update:
                doc_count = collection.count()
                print(f"\nFound existing database with {doc_count} documents.")
                while True:
                    choice = input("Do you want to delete the existing database and create a new one? (y/n): ").strip().lower()
                    if choice in ['y', 'n']:
                        break
                    print("Invalid choice. Please enter 'y' or 'n'.")
                
                if choice == 'n':
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
            raise FileNotFoundError(f"Movie data not found at: {json_path}")

        documents, doc_ids, metadatas = load_movie_json(json_path)
        if not documents:
            raise ValueError("No valid movie documents were generated")

        # Add to collection
        collection.add(documents=documents, ids=doc_ids, metadatas=metadatas)
        print(f"Successfully added {len(documents)} movies to the database.")
        return True

    except Exception as e:
        print(f"Error generating database: {str(e)}")
        return False

if __name__ == "__main__":
    generate_database()