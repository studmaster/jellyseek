# JellySeek

A RAG (Retrieval Augmented Generation) implementation for Jellyfin that provides AI-powered movie recommendations from your personal library.

## Description

JellySeek connects to your Jellyfin server to provide intelligent movie recommendations using:
- ChromaDB for vector storage
- Ollama for embeddings and text generation
- RAG for context-aware responses

## Prerequisites

- Python 3.8 or higher
- Jellyfin server with API access
- Ollama server running locally or remotely
- Access to your Jellyfin API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jellyseek.git
cd jellyseek
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Copy the example configuration:
```bash
cp .env.example .env
```

4. Configure your `.env` file:
```bash
# Jellyfin Configuration
JELLYFIN_SERVER_URL=http://localhost:8096   # Your Jellyfin server URL
JELLYFIN_SERVER_API_KEY=your_api_key        # Your Jellyfin API key

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434      # Your Ollama server URL
EMBEDDING_MODEL=bge-large                   # Model for text embeddings
GENERATION_MODEL=gemma3:27b-it-qat         # Model for text generation

# Prompt Templates
embeddings_prompt=prompts/embeddings_prompt.txt
generation_prompt=prompts/generation_prompt.txt

# Data Storage Paths
CHROMADB_PATH=~/.local/share/jellyseek/chromadb   # ChromaDB storage
JELLYFIN_DATA_PATH=~/.local/share/jellyseek/data  # Exported data storage
```

## Usage

1. Run JellySeek:
```bash
jellyseek
```

2. Available commands in the chat:
- `/help` - Show available commands
- `/update` - Update movie database from Jellyfin
- `/quit` - Exit the application

## First Run

On first run, JellySeek will:
1. Create necessary directories
2. Fetch your movie library from Jellyfin
3. Generate embeddings and create the vector database
4. Start the chat interface

## Configuration Details

### Jellyfin Setup
1. Get your API key from Jellyfin:
   - Admin Dashboard → Advanced → API Keys → New API Key
2. Note your Jellyfin server URL (including port)

### Ollama Setup
1. Install Ollama following instructions at: https://ollama.ai
2. Pull required models:
```bash
ollama pull bge-large
ollama pull gemma3:27b-it-qat
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.