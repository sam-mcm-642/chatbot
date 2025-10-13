from typing import List, Dict
from .vector_store import VectorStore

class RAGRetriever:
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.3
    ) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            n_results: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold (0-1, lower distance = higher similarity)
        
        Returns:
            Formatted context string
        """
        # Search for relevant documents
        results = self.vector_store.search(query, n_results=n_results)
        
        # Filter by similarity threshold
        # Note: ChromaDB returns distances (lower = more similar)
        filtered_results = [r for r in results if r['distance'] < (1 - min_similarity)]
        
        if not filtered_results:
            return ""
        
        # Format context
        context_parts = []
        for i, result in enumerate(filtered_results, 1):
            source = result['metadata'].get('source', 'Unknown')
            date = result['metadata'].get('date', '')
            
            context_part = f"[Context {i}"
            if date:
                context_part += f" - {date}"
            context_part += f"]:\n{result['text']}\n"
            
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        return context
    
    def retrieve_with_metadata(
        self,
        query: str,
        n_results: int = 3,
        metadata_filter: Dict = None
    ) -> List[Dict]:
        """
        Retrieve documents with full metadata
        
        Args:
            query: User query
            n_results: Number of results
            metadata_filter: Filter by metadata (e.g., {'source': 'day_one'})
        
        Returns:
            List of result dicts
        """
        return self.vector_store.search(
            query,
            n_results=n_results,
            filter_metadata=metadata_filter
        )
    
    def retrieve_by_date_range(
        self,
        query: str,
        start_date: str,
        end_date: str,
        n_results: int = 5
    ) -> str:
        """
        Retrieve context filtered by date range
        
        Args:
            query: Search query
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            n_results: Number of results
        
        Returns:
            Formatted context string
        """
        # Note: This requires date metadata in your chunks
        # ChromaDB filtering syntax
        metadata_filter = {
            "$and": [
                {"date": {"$gte": start_date}},
                {"date": {"$lte": end_date}}
            ]
        }
        
        results = self.vector_store.search(
            query,
            n_results=n_results,
            filter_metadata=metadata_filter
        )
        
        # Format similar to retrieve_context
        context_parts = []
        for i, result in enumerate(results, 1):
            date = result['metadata'].get('date', '')
            context_parts.append(f"[{date}]: {result['text']}")
        
        return "\n\n".join(context_parts)

if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store)
    
    # Test retrieval
    context = retriever.retrieve_context("What have I learned about productivity?")
    print("Retrieved context:")
    print(context)