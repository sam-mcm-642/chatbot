import mlx.core as mx
from mlx_lm import load, generate
import gc
from typing import Optional

class ChatbotModel:
    def __init__(self, model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model using MLX"""
        try:
            print(f"Loading model: {self.model_name}")
            print("Using MLX (optimized for Apple Silicon)")
            
            # MLX loads model and tokenizer together
            self.model, self.tokenizer = load(self.model_name)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_response(
        self, 
        user_input: str, 
        conversation_history: list = None,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate a response to user input"""
        
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please wait while I initialize..."
        
        try:
            # Format conversation history if provided
            if conversation_history:
                # Build chat format
                messages = []
                for msg in conversation_history[-6:]:  # Last 6 messages
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                messages.append({"role": "user", "content": user_input})
                
                # Format using chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Simple prompt without history
                prompt = user_input
            
            print(f"DEBUG: Generating response for: '{user_input[:50]}...'")
            
            # Generate using MLX
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_length,
                verbose=False
            )
            
            # MLX returns the full text including prompt, extract only the response
            # Remove the prompt from the response
            if prompt in response:
                response = response[len(prompt):].strip()
            
            print(f"DEBUG: Response generated successfully")
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return "I'm having trouble generating a response right now. Please try again."
    
    def unload_model(self):
        """Free up memory by unloading the model"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        gc.collect()
        print("Model unloaded from memory")

# Global model instance
chatbot_model = ChatbotModel()