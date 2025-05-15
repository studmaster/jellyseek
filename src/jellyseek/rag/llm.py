from langchain_ollama import OllamaLLM
from jellyseek.rag.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    EMBEDDING_PROMPT,
    GENERATION_PROMPT
)

def read_prompt_file(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read().strip()

def generate_search_query(original_query: str) -> str:
    template = read_prompt_file(EMBEDDING_PROMPT)
    prompt = template.format(
        generation_model=GENERATION_MODEL,
        embedding_model=EMBEDDING_MODEL,
        question=original_query
    )
    
    llm = OllamaLLM(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL)
    return llm.invoke(prompt).strip()

def generate_response(original_query: str, context: str) -> str:
    template = read_prompt_file(GENERATION_PROMPT)
    prompt = template.format(
        generation_model=GENERATION_MODEL,
        embedding_model=EMBEDDING_MODEL,
        context=context,
        question=original_query
    )
    
    llm = OllamaLLM(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL)
    return llm.invoke(prompt)