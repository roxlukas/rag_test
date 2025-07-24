# Very simple RAG implementation (Retrieveal - Augmented Generation)

Purely for educational purposes - how to make a simple RAG

## **Architecture Overview**

1. **Document Loading**: Scrapes nginx.org documentation automatically
2. **Embedding**: Uses Ollama instance with `nomic-embed-text` model
3. **Vector Storage**: ChromaDB for persistent vector storage
4. **Retrieval**: Semantic search to find relevant docs
5. **Generation**: Uses Ollama chat model for responses

## **Key Features**

- **Automatic Documentation Scraping**: Crawls nginx.org/en/docs/
- **Chunking Strategy**: Splits docs into 1000-word chunks with 200-word overlap
- **Persistent Storage**: ChromaDB data persists between restarts
- **RESTful API**: FastAPI with automatic documentation
- **Interactive Client**: Command-line interface for easy testing
- **Health Monitoring**: Built-in health checks and logging

## **Quick Start**

1. **Save all the files** in a directory called `nginx-rag/`
2. **Make the setup script executable**: `chmod +x setup.sh`
3. **Run the setup**: `./setup.sh`

## **Required Ollama Models**

Make sure you have these models pulled:
```bash
ollama pull nomic-embed-text  # For embeddings
ollama pull qwen3:8b          # For chat (or adjust CHAT_MODEL)
```

## **Configuration**

- Ollama at `192.168.1.250:11434` (adjust as needed)
- Embedding model: `nomic-embed-text`
- Chat model: `qwen3:8b` (adjust in docker-compose.yml if needed)

## **How It Works**

1. **Loading Phase**: 
   - Scrapes nginx documentation
   - Chunks text into manageable pieces
   - Generates embeddings via Ollama
   - Stores in ChromaDB

2. **Query Phase**:
   - Embeds the question
   - Finds similar document chunks
   - Combines relevant context
   - Generates answer via Ollama

## **Testing Examples**

Try these queries:
- "How do I configure SSL in nginx?"
- "What's the difference between proxy_pass and fastcgi_pass?"
- "How do I set up load balancing?"
- "How do I configure nginx for serving static files?"

## **Customization Options**

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in main.py
- Change embedding/chat models in docker-compose.yml
- Modify the scraping logic to include/exclude certain pages
- Tune the number of retrieved chunks (`n_results`)

The system will automatically load the documentation on first startup (takes a few minutes), then persist the vector database for fast subsequent startups.
