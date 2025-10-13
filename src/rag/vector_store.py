import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from pathlib import Path
import json

class VectorStore:
    def __init__(
        self, 
        persist_directory: str = "data/embeddings/chroma",
        collection_name: str = "personal_documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store with ChromaDB
        
        Args:
            persist_directory: Where to store the database
            collection_name: Name of the collection
            embedding_model: Sentence transformer model to use
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("Embedding model loaded")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of dicts with 'text', 'metadata', 'chunk_id'
            batch_size: Number of documents to process at once
        """
        print(f"Adding {len(chunks)} documents to vector store...")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract components
            texts = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            ids = [f"doc_{i + j}" for j in range(len(batch))]
            
            # Generate embeddings
            embeddings = self.embed_texts(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        print(f"Successfully added {len(chunks)} documents")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of dicts with 'text', 'metadata', 'distance'
        """
        # Generate query embedding
        query_embedding = self.embed_texts([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }
    
    def clear_collection(self):
        """Delete all documents from the collection"""
        self.client.delete_collection(self.collection.name)
        print(f"Cleared collection: {self.collection.name}")

if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()
    
    # Load processed chunks
    with open("data/processed/chunks.json", 'r') as f:
        chunks = json.load(f)
    
    # Add to vector store
    vector_store.add_documents(chunks)
    
    # Test search
    results = vector_store.search("career thoughts", n_results=3)
    
    print("\nSearch results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Source: {result['metadata'].get('source', 'unknown')}")