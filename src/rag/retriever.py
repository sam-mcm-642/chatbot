# from typing import Optional
# from .vector_store import VectorStore
# from .retrieval_config import RetrievalConfig
# from .retrieval_strategy import (
#     RetrievalStrategy,
#     StandardRetrievalStrategy,
#     GroupedRetrievalStrategy,
#     AdaptiveRetrievalStrategy,
#     RetrievalResult
# )

# class RAGRetriever:
#     """
#     Clean interface for RAG retrieval.
#     Strategy pattern allows swapping retrieval approaches without changing client code.
#     """
    
#     def __init__(
#         self,
#         vector_store: VectorStore,
#         config: Optional[RetrievalConfig] = None,
#         strategy: Optional[RetrievalStrategy] = None
#     ):
#         """
#         Initialize retriever with configuration and strategy.
        
#         Args:
#             vector_store: The vector store to query
#             config: Retrieval configuration (uses defaults if not provided)
#             strategy: Retrieval strategy (uses standard if not provided)
#         """
#         self.vector_store = vector_store
#         self.config = config or RetrievalConfig()
#         self.config.validate()
        
#         self.strategy = strategy or StandardRetrievalStrategy(vector_store)
    
#     def retrieve_context(
#         self,
#         query: str,
#         **kwargs
#     ) -> Optional[str]:
#         """
#         Retrieve context for a query using the configured strategy.
        
#         Args:
#             query: The search query
#             **kwargs: Strategy-specific parameters
            
#         Returns:
#             Context string or None if no relevant context found
#         """
#         import time
#         import random
#         debug_id = random.randint(1000, 9999)
        
#         print(f"\n{'='*80}")
#         print(f"[{debug_id}] retrieve_context() CALLED")
#         print(f"[{debug_id}] Query: '{query}'")
#         print(f"[{debug_id}] Kwargs: {kwargs}")
#         print(f"[{debug_id}] self object id: {id(self)}")
#         print(f"[{debug_id}] vector_store object id: {id(self.vector_store)}")
#         print(f"{'='*80}\n")
#         result = self.strategy.retrieve(query, self.config, **kwargs)
#         return result.context if result else None
    
#     def retrieve(
#         self,
#         query: str,
#         **kwargs
#     ) -> Optional[RetrievalResult]:
#         """
#         Retrieve full result object with metadata.
#         Useful for debugging and analysis.
#         """
#         return self.strategy.retrieve(query, self.config, **kwargs)
    
#     def set_strategy(self, strategy: RetrievalStrategy):
#         """
#         Change retrieval strategy at runtime.
#         """
#         self.strategy = strategy
    
#     def update_config(self, **kwargs):
#         """
#         Update configuration parameters.
#         """
#         import dataclasses
#         self.config = dataclasses.replace(self.config, **kwargs)
#         self.config.validate()


import re
from typing import Optional, List, Dict
from datetime import datetime
from .vector_store import VectorStore

class RAGRetriever:
    def __init__(self, vector_store: VectorStore):
        """Initialize RAG retriever"""
        self.vector_store = vector_store
    
    def _extract_year_filter(self, query: str) -> Optional[str]:
        """
        Extract year from query like "2025 targets", "from 2024", etc.
        Returns the year as a string, or None if no year found.
        """
        # Look for 4-digit years (2020-2029)
        year_pattern = r'\b(202[0-9])\b'
        match = re.search(year_pattern, query)
        
        if match:
            return match.group(1)
        
        # Check for relative time
        query_lower = query.lower()
        current_year = datetime.now().year
        
        if any(word in query_lower for word in ['this year', 'current year']):
            return str(current_year)
        
        if any(word in query_lower for word in ['last year', 'previous year']):
            return str(current_year - 1)
        
        if any(word in query_lower for word in ['recent', 'latest', 'current']):
            return str(current_year)
        
        return None
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.3,
        max_context_length: int = 3000,
        **kwargs
    ) -> Optional[str]:
        """
        Retrieve context with automatic year filtering
        """
        import time
        import random
        debug_id = random.randint(1000, 9999)
        
        print(f"\n{'='*80}")
        print(f"[{debug_id}] retrieve_context() CALLED")
        print(f"[{debug_id}] Query: '{query}'")
        print(f"[{debug_id}] Kwargs: {kwargs}")
        print(f"[{debug_id}] self object id: {id(self)}")
        print(f"[{debug_id}] vector_store object id: {id(self.vector_store)}")
        
        # Extract year filter
        year = self._extract_year_filter(query)
        
        metadata_filter = None
        if year:
            print(f"[{debug_id}] ✓ Detected year: {year}")
            metadata_filter = {
                "$and": [
                    {"week_date": {"$gte": f"{year}-01-01"}},
                    {"week_date": {"$lte": f"{year}-12-31"}}
                ]
            }
            print(f"[{debug_id}] Applying filter: {metadata_filter}")
        else:
            print(f"[{debug_id}] No year detected, searching all documents")
        
        print(f"{'='*80}\n")
        
        # Search with filter
        results = self.vector_store.search(
            query,
            n_results=n_results,
            filter_metadata=metadata_filter
        )
        
        print(f"[{debug_id}] Raw search returned {len(results)} results")
        if results:
            print(f"[{debug_id}] First result week: {results[0]['metadata'].get('week_date', 'unknown')}")
        
        # Filter by similarity
        filtered_results = [
            r for r in results 
            if r['distance'] < (1 - min_similarity)
        ]
        
        print(f"[{debug_id}] After similarity filter: {len(filtered_results)} results")
        
        if not filtered_results:
            return None
        
        # Group by week
        weeks_data = {}
        for result in filtered_results:
            week = result['metadata'].get('week_date', 'unknown')
            if week not in weeks_data:
                weeks_data[week] = []
            weeks_data[week].append(result['text'])
        
        # Sort weeks (most recent first)
        sorted_weeks = sorted(
            weeks_data.keys(),
            reverse=True,
            key=lambda x: x if x != 'unknown' else '0000-00-00'
        )
        
        print(f"[{debug_id}] Weeks found: {sorted_weeks[:5]}")  # Show first 5
        
        # Build context
        context_parts = []
        current_length = 0
        
        for week in sorted_weeks:
            week_texts = '\n'.join(weeks_data[week])
            
            if week != 'unknown':
                week_section = f"=== Week of {week} ===\n{week_texts}\n"
            else:
                week_section = f"{week_texts}\n"
            
            if current_length + len(week_section) > max_context_length:
                if len(context_parts) == 0:
                    context_parts.append(week_section)
                break
            
            context_parts.append(week_section)
            current_length += len(week_section)
        
        context = '\n'.join(context_parts)
        
        # Add summary
        if year:
            summary = f"[Showing {len(sorted_weeks)} weeks from {year}]\n\n"
        else:
            summary = f"[Showing {len(sorted_weeks)} weeks]\n\n"
        
        return summary + context