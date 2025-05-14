# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import json
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict

import requests

# Define the LLM model to be used
embedding_model = "bge-large"
generation_model = "gemma3:27b-it-qat"

# Configure ChromaDB
# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define a custom embedding function for ChromaDB using Ollama
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

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=embedding_model,
        base_url="http://100.112.80.41:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Define a collection for the RAG workflow
collection_name = "movies_rag"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for movies rag"},
    embedding_function=embedding  # Use the custom embedding function
)

from collections import OrderedDict
import re, unicodedata, uuid

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
        year_from = ""             # reuse parsed year if you like

        # harvest year again (or cache during dedup, whichever you prefer)
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
            f"Plot: {plot}"
        )

        documents.append(doc_text)
        ids.append(f"{slug(title)}_{year_from}" or uuid.uuid4().hex)
        raw_meta = {
            "title": title,                          # str
            "year": int(year_from) if year_from else 0,   # int fallback
            "genres": ", ".join(item.get("Genres", [])),  # str
            "critic_rating": item.get("CriticRating"),    # may be None
            "official_rating": item.get("OfficialRating") # may be None
        }

        metadatas.append(clean_metadata(raw_meta))
    return documents, ids, metadatas

# Absolute path to ~/jellyseek/movie_summary.json
json_path = Path.home() / "jellyseek" / "movie_summary.json"
documents, doc_ids, metadatas = load_movie_json(json_path)

# Insert into Chroma (ids + metadatas)
collection.add(documents=documents, ids=doc_ids, metadatas=metadatas)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=10):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=generation_model)
    return llm.invoke(prompt)

def generate_search_query(original_query: str) -> str:
    """Generate an optimized search query from the original user query"""
    prompt = f"""Convert the following question into a clear, concise search query 
    that would help find relevant movie information. Focus on key terms and concepts.
    
    Question: {original_query}
    
    Search query:"""
    
    llm = OllamaLLM(model=generation_model)
    return llm.invoke(prompt).strip()

def generate_final_response(original_query: str, context: str) -> str:
    """Generate final response using original query and retrieved context"""
    prompt = f"""Based on the following movie information, answer the original question.
    Be specific and reference relevant details from the context.
    
    Context: {context}
    
    Original Question: {original_query}
    
    Answer:"""
    
    llm = OllamaLLM(model=generation_model)
    return llm.invoke(prompt)

def rag_pipeline(query_text: str):
    """
    Enhanced RAG pipeline with three-step process:
    1. Generate optimized search query
    2. Retrieve relevant documents
    3. Generate final response
    """
    # Step 1: Generate optimized search query
    search_query = generate_search_query(query_text)
    print("######## Generated Search Query ########")
    print(search_query)
    
    # Step 2: Retrieve relevant documents using the optimized query
    retrieved_docs, metadata = query_chromadb(search_query)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
    print("######## Retrieved Context ########")
    print(context)
    
    # Step 3: Generate final response using original query and context
    response = generate_final_response(query_text, context)
    return response

user_query = input("Enter your question about movies: ").strip()
result = rag_pipeline(user_query)
print("\n######## Final Response ########")
print(result)