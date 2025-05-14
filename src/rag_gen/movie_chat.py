from langchain_ollama import OllamaLLM
import chromadb
import os
from typing import Tuple, List, Dict

# Model configurations
embedding_model = "bge-large"
generation_model = "gemma3:27b-it-qat"

def query_chromadb(collection, query_text: str, n_results: int = 10) -> Tuple[List[str], List[Dict]]:
    """Query the ChromaDB collection for relevant documents."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

def generate_search_query(original_query: str) -> str:
    """Generate an optimized search query from the original user query"""
    prompt = f"""As an AI assistant using {generation_model}, help convert the following question 
    into a search query optimized for {embedding_model} embeddings. The query should be clear, 
    concise and focus on key terms that will help find relevant movie information.
    
    Question: {original_query}
    
    Search query:"""
    
    llm = OllamaLLM(model=generation_model)
    return llm.invoke(prompt).strip()

def generate_final_response(original_query: str, context: str) -> str:
    """Generate final response using original query and retrieved context"""
    prompt = f"""As an AI assistant using {generation_model}, analyze the following movie information 
    retrieved using {embedding_model} embeddings and answer the original question.
    Be specific and reference relevant details from the context.
    
    Context: {context}
    
    Original Question: {original_query}
    
    Answer:"""
    
    llm = OllamaLLM(model=generation_model)
    return llm.invoke(prompt)

def chat_loop():
    """Main chat loop"""
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
    collection = chroma_client.get_collection("movies_rag")
    
    print("Movie Chat Assistant Ready! (Type 'quit' to exit)")
    
    while True:
        user_query = input("\nEnter your question about movies: ").strip()
        if user_query.lower() in ('quit', 'exit'):
            break
            
        # Step 1: Generate optimized search query
        search_query = generate_search_query(user_query)
        print("\n######## Generated Search Query ########")
        print(search_query)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs, metadata = query_chromadb(collection, search_query)
        context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."
        print("\n######## Retrieved Context ########")
        print(context)
        
        # Step 3: Generate final response
        response = generate_final_response(user_query, context)
        print("\n######## Final Response ########")
        print(response)

if __name__ == "__main__":
    chat_loop()