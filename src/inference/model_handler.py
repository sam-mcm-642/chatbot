import mlx.core as mx
from mlx_lm import load, generate
import gc
from typing import Optional
import sys
from pathlib import Path

# Add RAG imports
sys.path.append(str(Path(__file__).parent.parent))
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStore

class ChatbotModel:
    def __init__(
        self, 
        model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        use_rag: bool = False,
        vector_store_path: str = "data/embeddings/chroma"
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_rag = use_rag
        self.retriever = None
        
        if use_rag:
            print("Initializing RAG...")
            vector_store = VectorStore(persist_directory=vector_store_path)
            self.retriever = RAGRetriever(vector_store)
            print("RAG initialized")
    
    def load_model(self):
        """Load model using MLX"""
        try:
            print(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(
        self, 
        user_input: str, 
        conversation_history: list = None,
        max_length: int = 512,
        temperature: float = 0.7,
        use_rag: bool = None
    ) -> str:
        """Generate a response to user input"""
        
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please wait while I initialize..."
        
        # Determine if RAG should be used for this query
        use_rag_for_query = use_rag if use_rag is not None else self.use_rag
        
        try:
            # Build base prompt
            if conversation_history and len(conversation_history) > 1:
                context = ""
                for msg in conversation_history[-4:]:
                    if msg["role"] == "user":
                        context += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        context += f"Assistant: {msg['content']}\n"
                prompt = f"{context}User: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"
            
            # Add RAG context if enabled
            if use_rag_for_query and self.retriever:
                retrieved_context = self.retriever.retrieve_context(
                    user_input,
                    n_results=3
                )
                
                if retrieved_context:
                    print("DEBUG: Using RAG context")
                    # Insert context before the user query
                    prompt = f"Relevant information from your documents:\n{retrieved_context}\n\n{prompt}"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Generate
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_length
            )
            
            # Clean response
            if prompt in response:
                response = response[len(prompt):].strip()
            
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            return response if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def unload_model(self):
        """Free up memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        print("Model unloaded from memory")

# Global model instance
chatbot_model = ChatbotModel(use_rag=True)