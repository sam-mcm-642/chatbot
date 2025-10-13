"""
Process weekly targets from nested Craft export structure
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
import json

def process_craft_weekly_targets():
    """
    Process weekly targets from Craft export with nested structure
    Example: data/raw/Weekly targets/01/01/filename.md
    """
    
    # Your actual directory structure
    base_dir = "data/raw/Weekly targets"
    
    print("="*60)
    print("Processing Craft Weekly Targets")
    print("="*60)
    print(f"\nSearching in: {base_dir}")
    
    # Check if directory exists
    if not Path(base_dir).exists():
        print(f"❌ Directory not found: {base_dir}")
        print("Please check the path")
        return False
    
    # Find all markdown files (including nested)
    base_path = Path(base_dir)
    md_files = list(base_path.glob("**/*.md"))
    
    print(f"Found {len(md_files)} markdown files")
    
    if not md_files:
        print("❌ No markdown files found")
        return False
    
    # Show the structure
    print("\nFile structure:")
    for f in sorted(md_files)[:5]:  # Show first 5
        relative_path = f.relative_to(base_path)
        print(f"  📄 {relative_path}")
    if len(md_files) > 5:
        print(f"  ... and {len(md_files) - 5} more")
    
    # Process all files
    print("\n" + "="*60)
    print("Processing documents...")
    print("="*60)
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    
    for file_path in sorted(md_files):
        try:
            print(f"\nProcessing: {file_path.name}")
            chunks = processor.process_craft_weekly_targets(str(file_path))
            
            if chunks:
                # Show info about this week
                meta = chunks[0]['metadata']
                week_date = meta.get('week_date', 'unknown')
                completed = meta.get('completed_tasks', 0)
                total = meta.get('total_tasks', 0)
                rate = meta.get('completion_rate', 0)
                
                print(f"  Week: {week_date}")
                print(f"  Tasks: {completed}/{total} ({rate}%)")
                print(f"  Chunks created: {len(chunks)}")
                
                all_chunks.extend(chunks)
            else:
                print(f"  ⚠️  No chunks created (might not be a weekly target doc)")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if not all_chunks:
        print("\n❌ No chunks were created")
        return False
    
    # Save chunks
    output_path = Path("data/processed/weekly_targets_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\n✓ Saved {len(all_chunks)} chunks to {output_path}")
    
    # Add to vector store
    print("\n" + "="*60)
    print("Adding to Vector Store...")
    print("="*60)
    
    vector_store = VectorStore(
        persist_directory="data/embeddings/chroma",
        collection_name="personal_documents"
    )
    
    vector_store.add_documents(all_chunks)
    
    stats = vector_store.get_collection_stats()
    print(f"\n✓ Vector store now contains {stats['total_documents']} documents")
    
    # Test retrieval
    print("\n" + "="*60)
    print("Testing Retrieval...")
    print("="*60)
    
    test_query = "gym goals"
    results = vector_store.search(test_query, n_results=3)
    
    print(f"\nTest search: '{test_query}'")
    for i, result in enumerate(results, 1):
        week = result['metadata'].get('week_date', 'unknown')
        print(f"\n  Result {i} (Week: {week}):")
        print(f"  {result['text'][:150]}...")
    
    print("\n" + "="*60)
    print("✅ Processing Complete!")
    print("="*60)
    print("\nYour weekly targets are now available in RAG.")
    print("Try asking your chatbot:")
    print("  - 'What were my fitness goals?'")
    print("  - 'Show me my job search progress'")
    print("  - 'What targets did I set in October?'")
    
    return True

if __name__ == "__main__":
    success = process_craft_weekly_targets()
    sys.exit(0 if success else 1)