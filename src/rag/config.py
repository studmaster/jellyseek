from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get project root directory (where .env file is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma3:27b-it-qat")

# Prompt file paths - resolve relative to project root
EMBEDDING_PROMPT = os.path.join(PROJECT_ROOT, os.getenv("embeddings_prompt", "prompts/embeddings_prompt.txt"))
GENERATION_PROMPT = os.path.join(PROJECT_ROOT, os.getenv("generation_prompt", "prompts/generation_prompt.txt"))

# Validate required environment variables
if not OLLAMA_BASE_URL:
    raise ValueError("OLLAMA_BASE_URL must be set in the environment variables")

# Validate prompt files exist
if not os.path.exists(EMBEDDING_PROMPT):
    raise FileNotFoundError(f"Embedding prompt file not found at: {EMBEDDING_PROMPT}")
if not os.path.exists(GENERATION_PROMPT):
    raise FileNotFoundError(f"Generation prompt file not found at: {GENERATION_PROMPT}")

# Optional: Set default models if not specified
if not EMBEDDING_MODEL:
    EMBEDDING_MODEL = "bge-large"
    print(f"Warning: Using default embedding model: {EMBEDDING_MODEL}")

if not GENERATION_MODEL:
    GENERATION_MODEL = "gemma3:27b-it-qat"
    print(f"Warning: Using default generation model: {GENERATION_MODEL}")