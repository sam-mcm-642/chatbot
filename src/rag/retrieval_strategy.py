from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """
    Standardized retrieval result format.
    Makes it easy to test and swap implementations.
    """
    context: str
    metadata: Dict
    num_sources: int
    total_length: int
    sources_used: List[str]

class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.
    Allows different approaches without changing client code.
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        config: 'RetrievalConfig',
        **kwargs
    ) -> Optional[RetrievalResult]:
        """
        Retrieve context for a query.
        
        Args:
            query: The search query
            config: Retrieval configuration
            **kwargs: Strategy-specific parameters
            
        Returns:
            RetrievalResult or None if no relevant context found
        """
        pass

class StandardRetrievalStrategy(RetrievalStrategy):
    """
    Standard semantic search retrieval.
    No heuristics, no magic - just configurable semantic search.
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        config: 'RetrievalConfig',
        n_results: Optional[int] = None,
        max_context_length: Optional[int] = None,
        min_similarity: Optional[int] = None
    ) -> Optional[RetrievalResult]:
        """
        Retrieve using standard semantic search.
        All parameters are optional and fall back to config.
        """
        # Use provided values or fall back to config
        n_results = n_results or config.default_n_results
        max_context_length = max_context_length or config.default_max_context_length
        min_similarity = min_similarity or config.min_similarity_threshold
        
        # Execute search
        results = self.vector_store.search(query, n_results=n_results)
        
        # Filter by similarity
        filtered = [
            r for r in results
            if r['distance'] < (1 - min_similarity)
        ]
        
        if not filtered:
            return None
        
        # Build context
        context_parts = []
        current_length = 0
        sources_used = []
        
        for result in filtered:
            text = result['text']
            
            # Apply chunk truncation if configured
            if config.max_chunk_length and len(text) > config.max_chunk_length:
                text = text[:config.max_chunk_length] + "..."
            
            # Check length constraint
            if current_length + len(text) > max_context_length:
                if len(context_parts) == 0:
                    # Include at least one result
                    context_parts.append(text)
                    current_length += len(text)
                break
            
            context_parts.append(text)
            current_length += len(text)
            
            # Track source
            source = result['metadata'].get('filename', 'unknown')
            if source not in sources_used:
                sources_used.append(source)
        
        context = '\n\n'.join(context_parts)
        
        # Add summary if configured
        if config.include_result_count_summary:
            context = f"[Retrieved {len(context_parts)} relevant passages]\n\n{context}"
        
        return RetrievalResult(
            context=context,
            metadata={'strategy': 'standard'},
            num_sources=len(context_parts),
            total_length=len(context),
            sources_used=sources_used
        )

class GroupedRetrievalStrategy(RetrievalStrategy):
    """
    Retrieval that groups results by a metadata field.
    Generic - works with any grouping field, not just dates.
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        config: 'RetrievalConfig',
        group_by_field: Optional[str] = None,
        **kwargs
    ) -> Optional[RetrievalResult]:
        """
        Retrieve and group by metadata field.
        
        Args:
            group_by_field: Metadata field to group by (e.g., "week_date", "category")
        """
        group_by_field = group_by_field or config.group_by_metadata_field
        
        if not group_by_field:
            # Fall back to standard retrieval
            standard = StandardRetrievalStrategy(self.vector_store)
            return standard.retrieve(query, config, **kwargs)
        
        # Retrieve results
        n_results = kwargs.get('n_results', config.default_n_results)
        results = self.vector_store.search(query, n_results=n_results)
        
        # Group by field
        groups = {}
        for result in results:
            group_value = result['metadata'].get(group_by_field, 'unknown')
            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(result)
        
        # Format grouped results
        context_parts = []
        current_length = 0
        max_length = kwargs.get('max_context_length', config.default_max_context_length)
        
        # Sort groups (you can make this configurable too)
        sorted_groups = sorted(groups.keys(), reverse=True)
        
        for group_value in sorted_groups:
            group_results = groups[group_value]
            
            # Format group header
            if config.include_metadata_headers:
                header = f"=== {group_by_field}: {group_value} ==="
                group_text = [header]
            else:
                group_text = []
            
            # Add group contents
            for result in group_results:
                text = result['text']
                if config.max_chunk_length and len(text) > config.max_chunk_length:
                    text = text[:config.max_chunk_length] + "..."
                group_text.append(text)
            
            full_group = '\n'.join(group_text)
            
            if current_length + len(full_group) > max_length:
                if len(context_parts) == 0:
                    context_parts.append(full_group)
                break
            
            context_parts.append(full_group)
            current_length += len(full_group)
        
        context = '\n\n'.join(context_parts)
        
        if config.include_result_count_summary:
            context = f"[Showing {len(context_parts)} groups by {group_by_field}]\n\n{context}"
        
        return RetrievalResult(
            context=context,
            metadata={'strategy': 'grouped', 'grouped_by': group_by_field},
            num_sources=len(context_parts),
            total_length=len(context),
            sources_used=list(groups.keys())
        )

class AdaptiveRetrievalStrategy(RetrievalStrategy):
    """
    Adjusts retrieval parameters based on result quality.
    Generic adaptation logic, not hard-coded heuristics.
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.base_strategy = StandardRetrievalStrategy(vector_store)
    
    def retrieve(
        self,
        query: str,
        config: 'RetrievalConfig',
        **kwargs
    ) -> Optional[RetrievalResult]:
        """
        Adaptively adjust retrieval based on initial results.
        """
        if not config.enable_adaptive_retrieval:
            return self.base_strategy.retrieve(query, config, **kwargs)
        
        # Initial probe
        probe_results = self.vector_store.search(query, n_results=5)
        
        if not probe_results:
            return None
        
        # Analyze quality
        best_distance = probe_results[0]['distance']
        avg_distance = sum(r['distance'] for r in probe_results) / len(probe_results)
        
        # Adjust parameters based on quality
        adjusted_config = config
        
        if best_distance > 0.6:  # Poor matches
            # Lower similarity threshold to get more results
            adjusted_config = dataclasses.replace(
                config,
                min_similarity_threshold=max(
                    0.1,
                    config.min_similarity_threshold - config.adaptive_similarity_adjustment
                )
            )
        
        if avg_distance < 0.4:  # Very good matches
            # Can afford to be more selective
            adjusted_config = dataclasses.replace(
                config,
                min_similarity_threshold=min(
                    0.5,
                    config.min_similarity_threshold + config.adaptive_similarity_adjustment
                )
            )
        
        return self.base_strategy.retrieve(query, adjusted_config, **kwargs)