#!/bin/bash

# Nginx Documentation RAG Setup Script
# This script sets up and runs the RAG system

set -e  # Exit on any error

echo "🚀 Setting up Nginx Documentation RAG System"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create project directory
PROJECT_DIR="nginx-rag"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
    echo "📁 Created project directory: $PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Create data directory for persistent storage
mkdir -p data

echo "📋 Project structure:"
echo "├── main.py"
echo "├── requirements.txt"
echo "├── Dockerfile"
echo "├── docker-compose.yml"
echo "├── client.py"
echo "└── data/ (for ChromaDB persistence)"

# Check Ollama connectivity
echo ""
echo "🔍 Checking Ollama connectivity..."
OLLAMA_URL="http://192.168.1.250:11434"

if curl -s "$OLLAMA_URL/api/tags" > /dev/null; then
    echo "✅ Ollama is accessible at $OLLAMA_URL"
    
    # Check available models
    echo "📦 Available models:"
    curl -s "$OLLAMA_URL/api/tags" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for model in data.get('models', []):
        print(f'  - {model[\"name\"]}')
except:
    print('  Could not parse model list')
"
else
    echo "⚠️  Warning: Cannot connect to Ollama at $OLLAMA_URL"
    echo "   Make sure Ollama is running and accessible"
    echo "   You might need to adjust the OLLAMA_BASE_URL in docker-compose.yml"
fi

echo ""
echo "🔧 Next steps:"
echo "1. Make sure you have the required Ollama models:"
echo "   ollama pull nomic-embed-text"
echo "   ollama pull qwen2.5:8b  # Your optimized choice for quality/speed"
echo ""
echo "2. Build and run the RAG system:"
echo "   docker-compose up --build"
echo ""
echo "3. Wait for the system to load nginx documentation (this may take a few minutes)"
echo ""
echo "4. Test the system:"
echo "   python3 client.py"
echo "   # Or make direct HTTP requests to http://localhost:8000"
echo ""
echo "5. Example queries to try:"
echo "   - 'How do I configure SSL in nginx?'"
echo "   - 'What is the difference between proxy_pass and fastcgi_pass?'"
echo "   - 'How do I set up load balancing?'"

# Function to start the system
start_system() {
    echo ""
    echo "🚀 Starting the RAG system..."
    docker-compose up --build -d
    
    echo "⏳ Waiting for system to be ready..."
    sleep 10
    
    # Wait for health check
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ System is ready!"
            break
        fi
        echo "   Still starting... ($i/30)"
        sleep 5
    done
    
    # Show logs
    echo ""
    echo "📋 Recent logs:"
    docker-compose logs --tail=20
    
    echo ""
    echo "🎉 RAG system is running!"
    echo "   - API: http://localhost:8000"
    echo "   - Health check: http://localhost:8000/health"
    echo "   - Documentation: http://localhost:8000/docs"
}

# Function to show usage examples
show_examples() {
    echo ""
    echo "💡 Usage Examples:"
    echo "=================="
    echo ""
    echo "1. Interactive client:"
    echo "   python3 client.py"
    echo ""
    echo "2. Single query:"
    echo "   python3 client.py \"How do I configure SSL?\""
    echo ""
    echo "3. Direct HTTP API:"
    echo "   curl -X POST http://localhost:8000/query \\"
    echo "        -H \"Content-Type: application/json\" \\"
    echo "        -d '{\"question\": \"How do I set up nginx reverse proxy?\"}'"
    echo ""
    echo "4. Check system status:"
    echo "   curl http://localhost:8000/health"
}

# Ask user what they want to do
echo ""
read -p "🤔 Do you want to start the system now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    start_system
    show_examples
else
    echo "👍 System is ready to start when you are!"
    echo "   Run: docker-compose up --build"
    show_examples
fi