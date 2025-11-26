"""
Test the retriever directly to see if IT'S caching or filtering incorrectly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

def test_retriever_with_different_queries():
    """
    Test if the retriever returns different results for different queries
    """
    print("="*80)
    print("TESTING RETRIEVER DIRECTLY")
    print("="*80)
    
    vector_store = VectorStore(persist_directory="data/embeddings/chroma")
    retriever = RAGRetriever(vector_store)
    
    test_queries = [
        "fitness and gym sessions",
        "work and career planning",
        "budget and finances",
        "weekly targets overview"
    ]
    
    results = {}
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        context = retriever.retrieve_context(query)
        
        if context:
            print(f"Context length: {len(context)} characters")
            print(f"First 300 chars:\n{context[:300]}")
            print(f"...\nLast 200 chars:\n{context[-200:]}")
            
            # Store for comparison
            results[query] = context
        else:
            print("❌ No context returned!")
            results[query] = None
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARING RETRIEVER OUTPUTS")
    print("="*80)
    
    queries_list = list(results.keys())
    
    for i in range(len(queries_list)):
        for j in range(i+1, len(queries_list)):
            q1 = queries_list[i]
            q2 = queries_list[j]
            
            context1 = results[q1]
            context2 = results[q2]
            
            if context1 and context2:
                # Compare if contexts are identical
                if context1 == context2:
                    print(f"\n❌ IDENTICAL: '{q1}' and '{q2}' returned SAME context!")
                else:
                    # Calculate overlap
                    lines1 = set(context1.split('\n'))
                    lines2 = set(context2.split('\n'))
                    overlap = len(lines1 & lines2)
                    overlap_pct = (overlap / len(lines1)) * 100 if lines1 else 0
                    
                    print(f"\n✓ Different: '{q1}' vs '{q2}'")
                    print(f"  Overlap: {overlap_pct:.0f}% of lines")

if __name__ == "__main__":
    test_retriever_with_different_queries()