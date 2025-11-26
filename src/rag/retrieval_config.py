from dataclasses import dataclass
from typing import Optional

@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval strategies.
    Centralized, type-safe, and easily adjustable.
    """
    # Retrieval parameters
    default_n_results: int = 5
    max_n_results: int = 20
    min_similarity_threshold: float = 0.3
    
    # Context window management
    default_max_context_length: int = 2000
    max_context_length: int = 10000
    min_context_length: int = 500
    
    # Chunk handling
    max_chunk_length: Optional[int] = None  # None = no truncation
    chunk_overlap_handling: str = "keep_all"  # "keep_all", "deduplicate", "merge"
    
    # Formatting
    include_metadata_headers: bool = True
    include_result_count_summary: bool = True
    group_by_metadata_field: Optional[str] = None  # e.g., "week_date"
    
    # Adaptive behavior
    enable_adaptive_retrieval: bool = False
    adaptive_similarity_adjustment: float = 0.1
    
    def validate(self):
        """Ensure configuration values are sensible"""
        assert 0 <= self.min_similarity_threshold <= 1
        assert self.min_context_length <= self.max_context_length
        assert self.default_n_results <= self.max_n_results