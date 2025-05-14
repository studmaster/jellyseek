# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import json
from pathlib import Path
from datetime import datetime

# Define the LLM model to be used
llm_model = "bge-large"

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
        model=llm_model,
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

def load_movie_json(json_file: Path):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

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
        metadatas.append(
            {
                "title": title,
                "year": year_from,
                "genres": item.get("Genres", []),
                "critic_rating": item.get("CriticRating"),
                "official_rating": item.get("OfficialRating"),
            }
        )

    return documents, ids, metadatas

# Absolute path to ~/jellyseek/movie_summary.json
json_path = Path.home() / "jellyseek" / "movie_summary.json"
documents, doc_ids, metadatas = load_movie_json(json_path)

# Insert into Chroma (ids + metadatas)
collection.add(documents=documents, ids=doc_ids, metadatas=metadatas)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
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
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

result = rag_pipeline("Tell me about Wolverine’s trip to Japan.")
print(result)