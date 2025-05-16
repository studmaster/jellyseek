import chromadb
from langchain_ollama import OllamaEmbeddings
from jellyseek.rag.config import MOVIES_COLLECTION_NAME, CHROMADB_PATH, EMBEDDING_MODEL, OLLAMA_BASE_URL

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

def initialize_database():
    """Initialize ChromaDB client and collection"""
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection_name = MOVIES_COLLECTION_NAME
    
    embedding = ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    )
    
    # Check if collection exists first
    collections = chroma_client.list_collections()
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding
        )
        print(f"\nFound existing database with {collection.count()} movies.")
    else:
        # Collection doesn't exist, create it
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding,
            metadata={"description": "Movies RAG collection"}
        )
        print("\nCreated new database.")
    
    return chroma_client, collection_name, embedding, collection

def query_database(collection, query_text: str, n_results: int = 10):
    """Query the ChromaDB collection"""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    documents = [doc for sublist in results["documents"] for doc in sublist]
    metadatas = [meta for sublist in results["metadatas"] for meta in sublist]
    return documents, metadatas