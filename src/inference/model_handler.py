import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
import gc
from typing import Optional, List, Dict
import sys
from pathlib import Path
from rag.retrieval_config import RetrievalConfig
from rag.retrieval_strategy import StandardRetrievalStrategy, AdaptiveRetrievalStrategy

sys.path.append(str(Path(__file__).parent.parent))
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStore

class ChatbotModel:
    def __init__(
        self, 
        model_name: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
        use_rag: bool = False,
        vector_store_path: str = "data/embeddings/chroma"
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_rag = use_rag
        self.strategy = None
        self.retriever = None
        self.rag_config = None
        self.prompt_cache = None  # For conversation state
        
        if use_rag:
            print("Initializing RAG...")
            vector_store = VectorStore(persist_directory=vector_store_path)
            config = RetrievalConfig(enable_adaptive_retrieval=True)
            strategy = AdaptiveRetrievalStrategy(vector_store)
            # retriever = RAGRetriever(vector_store, config=config, strategy=strategy)
            self.retriever = RAGRetriever(vector_store)
            print("RAG initialized")
    
    def load_model(self):
        """Load model using MLX"""
        try:
            print(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            
            # Initialize prompt cache for conversation
            self.prompt_cache = make_prompt_cache(self.model)
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def build_messages(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict]] = None,
        rag_context: Optional[str] = None
    ) -> List[Dict]:
        """
        Build properly formatted message list for chat models.
        """
        messages = []
        
        # System message with or without RAG context
        if rag_context:
            system_content = (
                "You are a helpful personal assistant. Use the provided context "
                "from the user's documents to give accurate responses. "
                "Be concise and natural.\n\n"
                f"Context:\n{rag_context}"
            )
        else:
            system_content = (
                "You are a helpful personal assistant. "
                "Be concise and natural in your responses."
            )
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history (last 3 exchanges = 6 messages)
        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    # def generate_response(
    #     self, 
    #     user_input: str, 
    #     conversation_history: Optional[List[Dict]] = None,
    #     max_tokens: int = 256,
    #     temperature: float = 0.7,
    #     top_p: float = 0.9,
    #     min_p: float = 0.0,
    #     use_rag: Optional[bool] = None
    # ) -> str:
    #     """
    #     Generate a response using proper MLX-LM API.
    #     """
        
    #     if self.model is None or self.tokenizer is None:
    #         return "Model not loaded. Please wait while I initialize..."
        
    #     use_rag_for_query = use_rag if use_rag is not None else self.use_rag
        
    #     try:
    #         # Retrieve context if RAG is enabled
    #         rag_context = None
    #         if use_rag_for_query and self.retriever:
    #             rag_context = self.retriever.retrieve_context(
    #                 user_input,
    #                 n_results=10
    #             )
    #             if rag_context:
    #                 print("DEBUG: Retrieved RAG context")
            
    #         # Build properly formatted messages
    #         messages = self.build_messages(
    #             user_input=user_input,
    #             conversation_history=conversation_history,
    #             rag_context=rag_context
    #         )
            
    #         # Apply chat template - this is the standard way
    #         prompt = self.tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )
            
    #         print(f"DEBUG: Prompt length: {len(prompt)} characters")
            
    #         # Create sampler with temperature and other sampling params
    #         # This is the PROPER way to control temperature, top_p, etc.
    #         sampler = make_sampler(
    #             temp=temperature,
    #             top_p=top_p,
    #             min_p=min_p,
    #             min_tokens_to_keep=1
    #         )
            
    #         # Generate using the proper MLX-LM API
    #         response = generate(
    #             self.model,
    #             self.tokenizer,
    #             prompt=prompt,
    #             max_tokens=max_tokens,
    #             sampler=sampler,  # Pass the sampler here
    #             prompt_cache=self.prompt_cache,  # Use cache for conversation state
    #             verbose=False
    #         )
            
    #         print(f"DEBUG: Generated {len(response)} characters")
            
    #         return response.strip() if response else "I'm not sure how to respond to that."
            
    #     except Exception as e:
    #         error_msg = f"Error generating response: {str(e)}"
    #         print("="*60)
    #         print(error_msg)
    #         import traceback
    #         traceback.print_exc()
    #         print("="*60)
    #         return "I encountered an error generating a response. Please try
    #         again."
    
    def generate_response(
        self, 
        user_input: str, 
        conversation_history: Optional[List[Dict]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.0,
        use_rag: Optional[bool] = None
    ) -> str:
        """Generate a response using proper MLX-LM API."""
        
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please wait while I initialize..."
        
        use_rag_for_query = use_rag if use_rag is not None else self.use_rag
        
        try:
            # Retrieve context if RAG is enabled
            rag_context = None
            if use_rag_for_query and self.retriever:
                print(f"\n{'='*80}")
                print(f"DEBUG: Retrieving for query: '{user_input}'")
                print(f"{'='*80}")
                
                rag_context = self.retriever.retrieve_context(
                    user_input,
                    n_results=5
                )
                
                if rag_context:
                    print(f"✓ Retrieved {len(rag_context)} characters of context")
                    print(f"Context preview (first 200 chars):\n{rag_context[:200]}...")
                    print(f"Context preview (last 200 chars):\n...{rag_context[-200:]}")
                else:
                    print("❌ No context retrieved")
                print(f"{'='*80}\n")
            
            # Build properly formatted messages
            messages = self.build_messages(
                user_input=user_input,
                conversation_history=conversation_history,
                rag_context=rag_context
            )
            
            # Debug: Print what we're sending to the model
            print(f"\n{'='*80}")
            print(f"DEBUG: Messages being sent to model")
            print(f"{'='*80}")
            for i, msg in enumerate(messages):
                print(f"\nMessage {i} ({msg['role']}):")
                content = msg['content']
                if len(content) > 300:
                    print(f"{content[:150]}...\n...\n...{content[-150:]}")
                else:
                    print(content)
            print(f"{'='*80}\n")
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"Final prompt length: {len(prompt)} characters")
            
            # Create sampler
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                min_p=min_p,
                min_tokens_to_keep=1
            )
            
            # Generate
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=self.prompt_cache,  # ← POTENTIAL ISSUE
                verbose=False
            )
            
            print(f"Generated response length: {len(response)} characters")
            print(f"Response: {response[:200]}...")
            
            return response.strip() if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print("="*60)
            print(error_msg)
            import traceback
            traceback.print_exc()
            print("="*60)
            return "I encountered an error generating a response. Please try again."
    
    def reset_cache(self):
        """Reset the prompt cache for a new conversation"""
        if self.model:
            self.prompt_cache = make_prompt_cache(self.model)
            print("Conversation cache reset")
    
    def unload_model(self):
        """Free up memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.prompt_cache:
            del self.prompt_cache
        gc.collect()
        print("Model unloaded from memory")


# Global model instance
chatbot_model = ChatbotModel(use_rag=True)