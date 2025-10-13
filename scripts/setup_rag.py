"""
Complete RAG setup pipeline
Run this to process your documents and set up RAG
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
import json

def setup_rag_pipeline(
    documents_dir: str = "data/raw/documents",
    file_extension: str = ".txt",
    output_chunks: str = "data/processed/chunks.json",
    vector_store_path: str = "data/embeddings/chroma"
):
    """
    Complete RAG setup pipeline
    
    1. Process documents into chunks
    2. Generate embeddings
    3. Store in vector database
    """
    
    print("="*60)
    print("RAG Pipeline Setup")
    print("="*60)
    
    # Step 1: Process documents
    print("\n1. Processing documents...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process_directory(documents_dir, file_extension)
    
    if not chunks:
        print("No documents found! Please add files to", documents_dir)
        return False
    
    # Save chunks
    Path(output_chunks).parent.mkdir(parents=True, exist_ok=True)
    with open(output_chunks, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_chunks}")
    
    # Step 2: Create vector store and add documents
    print("\n2. Creating vector store and generating embeddings...")
    vector_store = VectorStore(persist_directory=vector_store_path)
    vector_store.add_documents(chunks)
    
    # Step 3: Test retrieval
    print("\n3. Testing retrieval...")
    test_query = "test query"
    results = vector_store.search(test_query, n_results=3)
    
    print(f"\nTest search for: '{test_query}'")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Text: {result['text'][:100]}...")
        print(f"    Distance: {result['distance']:.4f}")
    
    # Stats
    stats = vector_store.get_collection_stats()
    print("\n" + "="*60)
    print("Setup Complete!")
    print(f"Total documents in vector store: {stats['total_documents']}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # Run the pipeline
    success = setup_rag_pipeline(
        documents_dir="data/raw/documents",
        file_extension=".txt"  # Change to .md, .json as needed
    )
    
    if success:
        print("\nRAG is ready to use!")
        print("Enable RAG in your chatbot by setting use_rag=True")
    else:
        print("\nSetup failed. Check the errors above.")