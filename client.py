#!/usr/bin/env python3
"""
RAG Client Script for testing the Nginx Documentation RAG system
"""

import requests
import json
import sys
import time
from typing import Dict, Any

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the RAG system is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"question": question},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("🚀 Nginx Documentation RAG Client")
        print("=" * 50)
        
        # Check health first
        print("Checking system health...")
        health = self.health_check()
        if "error" in health:
            print(f"❌ System not healthy: {health['error']}")
            return
        
        print(f"✅ System healthy! Collection size: {health.get('collection_size', 'unknown')}")
        print("\nEnter your questions about nginx (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("🔄 Searching and generating response...")
                start_time = time.time()
                
                result = self.query(question)
                
                end_time = time.time()
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                print(f"\n📝 Answer (took {end_time - start_time:.2f}s):")
                print("-" * 30)
                print(result['answer'])
                
                if result.get('sources'):
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['title']}")
                        print(f"   URL: {source['url']}")
                        print(f"   Relevance: {source['relevance_score']:.3f}")
                
                print(f"\n🔍 Used {result.get('context_used', 0)} context chunks")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

def main():
    """Main function"""
    client = RAGClient()
    
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}")
        result = client.query(question)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        
        print(f"\nAnswer: {result['answer']}")
        
        if result.get('sources'):
            print(f"\nSources:")
            for source in result['sources']:
                print(f"- {source['title']}: {source['url']}")
    else:
        # Interactive mode
        client.interactive_mode()

if __name__ == "__main__":
    main()