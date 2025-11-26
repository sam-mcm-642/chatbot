"""
Debug script to inspect the RAG system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

def inspect_vector_store():
    """
    Comprehensive inspection of what's in the vector store
    """
    print("="*80)
    print("RAG SYSTEM DIAGNOSTICS")
    print("="*80)
    
    # Load vector store
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    # Get statistics
    stats = vector_store.get_collection_stats()
    print(f"\nTotal documents in vector store: {stats['total_documents']}")
    
    if stats['total_documents'] == 0:
        print("\n❌ PROBLEM: Vector store is empty!")
        print("You need to run: python scripts/process_weekly_targets.py")
        return
    
    # Sample some documents to see what's actually stored
    print("\n" + "="*80)
    print("SAMPLE DOCUMENTS IN VECTOR STORE")
    print("="*80)
    
    # Get a few random documents
    test_results = vector_store.search("weekly targets", n_results=10)
    
    print(f"\nFound {len(test_results)} results for 'weekly targets'")
    
    for i, result in enumerate(test_results, 1):
        print(f"\n--- Document {i} ---")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Text preview: {result['text'][:200]}...")
    
    # Check for weekly targets specifically
    print("\n" + "="*80)
    print("WEEKLY TARGETS DOCUMENTS")
    print("="*80)
    
    # Try to find all weekly targets
    weekly_results = vector_store.collection.get(
        where={"doc_type": "weekly_targets"}
    )
    
    if weekly_results and weekly_results['ids']:
        print(f"\nFound {len(weekly_results['ids'])} weekly target documents")
        
        # Group by week
        weeks = {}
        for metadata in weekly_results['metadatas']:
            week = metadata.get('week_date', 'unknown')
            weeks[week] = weeks.get(week, 0) + 1
        
        print("\nDocuments by week:")
        for week in sorted(weeks.keys()):
            print(f"  {week}: {weeks[week]} chunks")
    else:
        print("\n❌ PROBLEM: No weekly_targets documents found!")
        print("Check if doc_type metadata was set correctly during processing")
    
    return vector_store

def test_retrieval_queries():
    """
    Test various retrieval queries to see what gets returned
    """
    print("\n" + "="*80)
    print("TESTING RETRIEVAL QUERIES")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    retriever = RAGRetriever(vector_store)
    
    test_queries = [
        "weekly targets",
        "all my weekly targets",
        "fitness goals",
        "gym sessions",
        "what were my targets",
        "show me my goals"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Test raw search
        raw_results = vector_store.search(query, n_results=5)
        print(f"Raw search returned {len(raw_results)} results")
        
        if raw_results:
            print(f"Best match distance: {raw_results[0]['distance']:.4f}")
            print(f"Best match week: {raw_results[0]['metadata'].get('week_date', 'unknown')}")
        
        # Test retriever
        context = retriever.retrieve_context(query, n_results=5)
        
        if context:
            print(f"Retriever returned {len(context)} characters of context")
            print(f"Context preview: {context[:200]}...")
        else:
            print("❌ Retriever returned NO context!")

def test_similarity_thresholds():
    """
    Test different similarity thresholds to see what's being filtered out
    """
    print("\n" + "="*80)
    print("TESTING SIMILARITY THRESHOLDS")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    query = "What were my weekly targets?"
    results = vector_store.search(query, n_results=10)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} results\n")
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        filtered = [r for r in results if r['distance'] < (1 - threshold)]
        print(f"Threshold {threshold}: {len(filtered)} results pass")
        
        if filtered:
            print(f"  Best: distance={filtered[0]['distance']:.4f}, week={filtered[0]['metadata'].get('week_date', 'unknown')}")

def check_embedding_quality():
    """
    Check if embeddings are being generated correctly
    """
    print("\n" + "="*80)
    print("CHECKING EMBEDDING QUALITY")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    # Test if similar queries return similar results
    similar_queries = [
        "fitness goals",
        "gym targets",
        "exercise plans"
    ]
    
    print("\nTesting semantic similarity:")
    for query in similar_queries:
        results = vector_store.search(query, n_results=3)
        print(f"\n'{query}':")
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. Distance: {r['distance']:.4f}, Week: {r['metadata'].get('week_date', 'unknown')}")
        else:
            print("  No results!")

if __name__ == "__main__":
    print("\n🔍 Starting RAG System Diagnostics\n")
    
    # Run all diagnostics
    vector_store = inspect_vector_store()
    
    if vector_store and vector_store.get_collection_stats()['total_documents'] > 0:
        test_retrieval_queries()
        test_similarity_thresholds()
        check_embedding_quality()
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)