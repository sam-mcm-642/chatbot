"""
Diagnose why retrieval might be returning same results for different queries
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.vector_store import VectorStore
import numpy as np

def test_embedding_generation():
    """
    Test if embeddings are actually different for different queries
    """
    print("="*80)
    print("TEST 1: EMBEDDING GENERATION")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    test_queries = [
        "fitness goals",
        "career planning",
        "budget management",
        "weekly targets",
        "gym sessions"
    ]
    
    embeddings = []
    for query in test_queries:
        emb = vector_store.embed_texts([query])[0]
        embeddings.append(emb)
        print(f"\nQuery: '{query}'")
        print(f"  Embedding shape: {np.array(emb).shape}")
        print(f"  First 5 values: {emb[:5]}")
        print(f"  Sum: {sum(emb):.4f}")
        print(f"  Mean: {np.mean(emb):.4f}")
    
    # Check if embeddings are different
    print("\n" + "="*80)
    print("EMBEDDING SIMILARITY MATRIX")
    print("="*80)
    
    embeddings_array = np.array(embeddings)
    
    # Compute cosine similarity between all pairs
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings_array)
    
    print("\nCosine similarity between queries:")
    print("(1.0 = identical, 0.0 = completely different)")
    print()
    print("        ", end="")
    for i, q in enumerate(test_queries):
        print(f"{q[:8]:>10}", end="")
    print()
    
    for i, q1 in enumerate(test_queries):
        print(f"{q1[:8]:>10}", end="")
        for j, q2 in enumerate(test_queries):
            print(f"{similarity_matrix[i,j]:>10.3f}", end="")
        print()
    
    # Check if any non-diagonal entries are suspiciously high
    max_off_diagonal = 0
    for i in range(len(test_queries)):
        for j in range(len(test_queries)):
            if i != j:
                max_off_diagonal = max(max_off_diagonal, similarity_matrix[i,j])
    
    print(f"\nMax similarity between different queries: {max_off_diagonal:.3f}")
    
    if max_off_diagonal > 0.95:
        print("❌ PROBLEM: Queries are producing nearly identical embeddings!")
        print("   The embedding model may not be working correctly.")
        return False
    else:
        print("✓ Embeddings are appropriately different")
        return True

def test_actual_search_results():
    """
    Test if ChromaDB is actually returning different results for different queries
    """
    print("\n" + "="*80)
    print("TEST 2: ACTUAL SEARCH RESULTS")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    test_queries = [
        "fitness and gym",
        "work and career",
        "weekly targets"
    ]
    
    all_results = {}
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = vector_store.search(query, n_results=5)
        
        all_results[query] = results
        
        print(f"Returned {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. Distance: {r['distance']:.4f}")
            print(f"     Week: {r['metadata'].get('week_date', 'unknown')}")
            print(f"     Text: {r['text'][:80]}...")
    
    # Check if results are identical across queries
    print("\n" + "="*80)
    print("COMPARING RESULT SETS")
    print("="*80)
    
    queries_list = list(all_results.keys())
    
    for i in range(len(queries_list)):
        for j in range(i+1, len(queries_list)):
            q1 = queries_list[i]
            q2 = queries_list[j]
            
            # Get document IDs or text snippets to compare
            docs1 = set([r['text'][:100] for r in all_results[q1]])
            docs2 = set([r['text'][:100] for r in all_results[q2]])
            
            overlap = len(docs1 & docs2)
            overlap_pct = (overlap / len(docs1)) * 100
            
            print(f"\nOverlap between '{q1}' and '{q2}':")
            print(f"  {overlap}/{len(docs1)} documents ({overlap_pct:.0f}%)")
            
            if overlap_pct > 80:
                print(f"  ❌ WARNING: {overlap_pct:.0f}% overlap is too high!")
    
    return True

def test_chromadb_configuration():
    """
    Check if ChromaDB is configured correctly
    """
    print("\n" + "="*80)
    print("TEST 3: CHROMADB CONFIGURATION")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    collection = vector_store.collection
    
    print(f"\nCollection name: {collection.name}")
    print(f"Collection metadata: {collection.metadata}")
    print(f"Number of documents: {collection.count()}")
    
    # Get a few random documents to check they have embeddings
    sample = collection.peek(limit=3)
    
    print("\nSample documents:")
    for i in range(len(sample['ids'])):
        print(f"\n  Document {i+1}:")
        print(f"    ID: {sample['ids'][i]}")
        if sample['embeddings'] and len(sample['embeddings']) > i:
            emb = sample['embeddings'][i]
            print(f"    Embedding length: {len(emb)}")
            print(f"    Embedding sum: {sum(emb):.4f}")
        else:
            print(f"    ❌ NO EMBEDDING!")
    
    return True

def test_vector_store_search_method():
    """
    Directly test the vector_store.search method with debug output
    """
    print("\n" + "="*80)
    print("TEST 4: VECTOR STORE SEARCH METHOD")
    print("="*80)
    
    from rag.vector_store import VectorStore
    
    # Monkey-patch the search method to add logging
    original_search = VectorStore.search
    
    def debug_search(self, query, n_results=5, filter_metadata=None):
        print(f"\n  DEBUG: search() called with query='{query}'")
        
        # Generate embedding
        query_embedding = self.embed_texts([query])[0]
        print(f"  DEBUG: Generated embedding, sum={sum(query_embedding):.4f}")
        
        # Call ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        print(f"  DEBUG: ChromaDB returned {len(results['documents'][0])} results")
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    VectorStore.search = debug_search
    
    # Now test
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    
    print("\nTesting search with different queries:")
    
    queries = ["fitness", "career", "budget"]
    for query in queries:
        print(f"\n--- Testing: '{query}' ---")
        results = vector_store.search(query, n_results=3)
        print(f"Top result: {results[0]['text'][:80]}...")
    
    # Restore original method
    VectorStore.search = original_search

if __name__ == "__main__":
    print("\n🔍 Diagnosing Retrieval Bug\n")
    
    embeddings_ok = test_embedding_generation()
    
    if embeddings_ok:
        test_actual_search_results()
        test_chromadb_configuration()
        test_vector_store_search_method()
    else:
        print("\n❌ CRITICAL: Fix embedding generation first")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)