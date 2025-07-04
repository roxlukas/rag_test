import os
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.config import Settings
import hashlib
import time
from typing import List, Dict, Any
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = "http://192.168.1.250:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Good for embeddings
CHAT_MODEL = "qwen2.5:8b"  # Using Qwen3:8B for optimal quality/speed balance
COLLECTION_NAME = "nginx_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class RAGSystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="/app/data/chroma_db")
        self.collection = None
        self.setup_collection()
    
    def setup_collection(self):
        """Initialize ChromaDB collection"""
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            logger.info(f"Loaded existing collection: {COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {COLLECTION_NAME}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                }
                async with session.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["embedding"]
                    else:
                        logger.error(f"Embedding API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    async def scrape_nginx_docs(self, base_url: str = "https://nginx.org/en/docs/") -> List[Dict[str, str]]:
        """Scrape nginx documentation pages"""
        visited_urls = set()
        documents = []
        
        async def fetch_page(session, url):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
        
        async def extract_links(session, url, html):
            soup = BeautifulSoup(html, 'html.parser')
            links = set()
            
            # Extract all links that are within the docs section
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only include nginx.org docs URLs
                if (full_url.startswith('https://nginx.org/en/docs/') and 
                    full_url not in visited_urls and 
                    not full_url.endswith('.pdf')):
                    links.add(full_url)
            
            return links
        
        async def process_page(session, url):
            if url in visited_urls:
                return []
            
            visited_urls.add(url)
            logger.info(f"Processing: {url}")
            
            html = await fetch_page(session, url)
            if not html:
                return []
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract main content
            content_div = soup.find('div', {'id': 'content'}) or soup.find('div', class_='content')
            if content_div:
                # Remove navigation and other non-content elements
                for elem in content_div.find_all(['nav', 'header', 'footer', 'script', 'style']):
                    elem.decompose()
                
                text_content = content_div.get_text(separator=' ', strip=True)
                # Clean up whitespace
                text_content = re.sub(r'\s+', ' ', text_content)
                
                if len(text_content) > 100:  # Only include substantial content
                    title = soup.find('title')
                    page_title = title.get_text() if title else url.split('/')[-1]
                    
                    documents.append({
                        'url': url,
                        'title': page_title,
                        'content': text_content
                    })
            
            # Find more links to process
            new_links = await extract_links(session, url, html)
            return list(new_links)
        
        async with aiohttp.ClientSession() as session:
            urls_to_process = [base_url]
            processed_count = 0
            max_pages = 50  # Limit to prevent infinite crawling
            
            while urls_to_process and processed_count < max_pages:
                current_url = urls_to_process.pop(0)
                new_urls = await process_page(session, current_url)
                urls_to_process.extend(new_urls)
                processed_count += 1
                
                # Rate limiting
                await asyncio.sleep(0.5)
        
        logger.info(f"Scraped {len(documents)} documents")
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    async def load_documents(self):
        """Load and process nginx documentation"""
        logger.info("Starting document loading...")
        
        # Check if collection already has documents
        if self.collection.count() > 0:
            logger.info(f"Collection already contains {self.collection.count()} documents")
            return
        
        # Scrape documents
        documents = await self.scrape_nginx_docs()
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = await self.get_embedding(chunk)
                if not embedding:
                    continue
                
                # Create unique ID
                chunk_id = hashlib.md5(f"{doc['url']}_{i}".encode()).hexdigest()
                
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                all_metadatas.append({
                    'url': doc['url'],
                    'title': doc['title'],
                    'chunk_index': i
                })
                all_ids.append(chunk_id)
                
                # Process in batches
                if len(all_chunks) >= 10:
                    self.collection.add(
                        documents=all_chunks,
                        embeddings=all_embeddings,
                        metadatas=all_metadatas,
                        ids=all_ids
                    )
                    logger.info(f"Added batch of {len(all_chunks)} chunks")
                    all_chunks, all_embeddings, all_metadatas, all_ids = [], [], [], []
        
        # Add remaining chunks
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
        
        logger.info(f"Document loading complete. Total chunks: {self.collection.count()}")
    
    async def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [
            {
                'content': doc,
                'metadata': meta,
                'distance': dist
            }
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate response using Ollama"""
        prompt = f"""Based on the following nginx documentation context, please answer the question.

Context:
{context}

Question: {query}

Please provide a helpful answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
                async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["response"]
                    else:
                        return f"Error generating response: {response.status}"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Main RAG query function"""
        # Search for relevant documents
        similar_docs = await self.search_similar(question, n_results=3)
        
        if not similar_docs:
            return {
                "answer": "I couldn't find relevant information in the nginx documentation.",
                "sources": []
            }
        
        # Prepare context
        context = "\n\n".join([doc['content'] for doc in similar_docs])
        
        # Generate response
        answer = await self.generate_response(question, context)
        
        # Prepare sources
        sources = [
            {
                "title": doc['metadata']['title'],
                "url": doc['metadata']['url'],
                "relevance_score": 1 - doc['distance']  # Convert distance to similarity
            }
            for doc in similar_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(similar_docs)
        }

# FastAPI app
app = FastAPI(title="Nginx Documentation RAG", version="1.0.0")
rag_system = RAGSystem()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: int

@app.on_event("startup")
async def startup_event():
    """Load documents on startup"""
    await rag_system.load_documents()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = await rag_system.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "collection_size": rag_system.collection.count() if rag_system.collection else 0
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Nginx Documentation RAG System",
        "endpoints": {
            "query": "/query",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)