from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma3:27b-it-qat")

# Validate required environment variables
if not OLLAMA_BASE_URL:
    raise ValueError("OLLAMA_BASE_URL must be set in the environment variables")

# Optional: Set default models if not specified
if not EMBEDDING_MODEL:
    EMBEDDING_MODEL = "bge-large"
    print(f"Warning: Using default embedding model: {EMBEDDING_MODEL}")

if not GENERATION_MODEL:
    GENERATION_MODEL = "gemma3:27b-it-qat"
    print(f"Warning: Using default generation model: {GENERATION_MODEL}")