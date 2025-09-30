# Personal AI Chatbot

A privacy-focused personal assistant chatbot optimized for Apple Silicon, designed to provide reflective conversations and eventually integrate with personal documents for context-aware interactions.

## Overview

This project implements a conversational AI assistant using MLX (Apple's machine learning framework) for optimal performance on M-series chips. The chatbot runs entirely locally, ensuring complete privacy for personal data and conversations.

## Features

### Current Implementation
- Local LLM inference using MLX-optimized Llama 3.2-1B-Instruct
- Streamlit-based chat interface
- Conversation history management
- Adjustable generation parameters (temperature, max length)
- Chat export functionality
- Native Apple Silicon optimization

### Planned Features
- Personal data integration (Day One journal entries, Craft documents)
- Retrieval-Augmented Generation (RAG) for document-aware responses
- Knowledge graph construction from personal data
- Entity extraction and relationship mapping
- Fine-tuning on personal writing style
- Temporal analysis of thought patterns
- Advanced query capabilities (e.g., "How has my thinking about X evolved?")

## Tech Stack

- **LLM Framework**: MLX / MLX-LM
- **Base Model**: Llama 3.2-1B-Instruct (4-bit quantized)
- **UI**: Streamlit
- **Language**: Python 3.11
- **Environment**: Conda (ARM64)
- **Platform**: macOS (Apple Silicon)

Future integrations:
- Vector database: ChromaDB or FAISS
- Graph processing: NetworkX
- Embeddings: sentence-transformers

## Project Structure