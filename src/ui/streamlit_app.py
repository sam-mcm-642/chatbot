import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference.model_handler import chatbot_model

def initialize_chat():
    """Initialize chat session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your personal chatbot powered by MLX on Apple Silicon. What would you like to talk about?"}
        ]
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

def check_model_status():
    """Check if model is actually loaded"""
    return hasattr(chatbot_model, 'model') and chatbot_model.model is not None

def main():
    st.set_page_config(
        page_title="Personal Chatbot",
        page_icon="💭",
        layout="wide"
    )
    
    # Header
    st.title("💭 Personal Chatbot")
    st.markdown("*Your reflective AI companion - Powered by MLX*")
    
    # Initialize chat
    initialize_chat()
    
    # Check actual model status
    model_actually_loaded = check_model_status()
    
    # Load model if needed (synchronous - MLX is fast enough)
    if not model_actually_loaded and not st.session_state.model_loaded:
        with st.spinner("Loading model for the first time... This may take a minute."):
            success = chatbot_model.load_model()
            st.session_state.model_loaded = success
            if success:
                st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model status
        st.subheader("Model Status")
        if model_actually_loaded:
            st.success("Model ready (MLX optimized)")
            model_ready = True
        else:
            st.error("Model not loaded")
            if st.button("Load Model"):
                with st.spinner("Loading..."):
                    success = chatbot_model.load_model()
                    st.session_state.model_loaded = success
                    if success:
                        st.rerun()
            model_ready = False
        
        # Generation settings
        if model_ready:
            st.subheader("Generation Settings")
            temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
            max_tokens = st.slider("Max Response Length", 50, 1000, 512, 50)
        else:
            temperature = 0.7
            max_tokens = 512
        
        # Chat management
        st.subheader("Chat Management")
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. What would you like to talk about?"}
            ]
            st.rerun()
        
        # Export chat
        if len(st.session_state.messages) > 1:
            chat_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="Export Chat",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )
        
        # Development info
        st.subheader("Development Status")
        st.write("✅ MLX Integration")
        st.write("✅ Apple Silicon Optimized")
        st.write("✅ Base LLM Working")
        st.write("⏳ Personal Data Import")
        st.write("⏳ Fine-tuning")
        st.write("⏳ RAG Integration")
        
        # System info
        st.subheader("System Info")
        import platform
        st.text(f"Arch: {platform.machine()}")
        st.text(f"Python: {platform.python_version()}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What's on your mind?", disabled=not model_ready):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot_model.generate_response(
                    user_input=prompt,
                    conversation_history=st.session_state.messages[:-1],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()