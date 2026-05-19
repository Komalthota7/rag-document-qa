# rag-document-qa
  AI-powered document question answering using RAG and Claude API
# RAG-Based Document Question Answering System

An AI-powered question answering system built with Python, 
ChromaDB, Sentence Transformers, and Anthropic Claude API.

## What It Does
- Load any text document into a vector database
- Ask questions in plain English
- Get accurate answers grounded in the document content
- Eliminates hallucination by restricting answers to retrieved context

## Tech Stack
- Python
- Anthropic Claude API (claude-sonnet-4-20250514)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)

## How to Run

### 1. Install dependencies
pip install anthropic chromadb sentence-transformers

### 2. Set your API key
set ANTHROPIC_API_KEY=your-api-key-here

### 3. Add your document
Replace document.txt with your own text file

### 4. Run the app
python rag.py

## How It Works
1. Document is loaded and split into chunks
2. Each chunk is converted into an embedding
3. Embeddings are stored in ChromaDB vector database
4. User question is converted into an embedding
5. Most similar chunks are retrieved from the database
6. Claude generates an answer based strictly on retrieved chunks
