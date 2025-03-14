# SimpleRAG: A Simple Lightweight Retrieval-Augmented Generation System

## Project Overview
SimpleRAG is a compact yet powerful implementation of a Retrieval-Augmented Generation (RAG) system designed to extract information from PDF documents and answer natural language queries. This project demonstrates practical knowledge of NLP, vector embeddings, similarity search, and language model integration.

## Features
- **PDF Document Processing**: Extract and chunk text from PDF documents
- **Vector Embeddings**: Generate high-quality semantic embeddings using Sentence Transformers
- **Vector Search**: Fast similarity search using FAISS (Facebook AI Similarity Search)
- **Contextual Question Answering**: Generate accurate responses using a language model enhanced with retrieved context

## Technical Stack
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Language Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (compact LLM for generation)
- **Vector Database**: FAISS with L2 distance metric for efficient similarity search
- **Document Processing**: PyPDF2 for PDF text extraction
- **Text Processing**: Custom chunking strategy based on paragraph boundaries

## How It Works
1. **Document Indexing**:
   - Load PDF documents with PyPDF2
   - Split text into chunks based on paragraph boundaries
   - Generate embeddings for each chunk using Sentence Transformers
   - Store embeddings in a FAISS index for efficient retrieval

2. **Query Processing**:
   - Encode user questions using the same Sentence Transformer
   - Retrieve the most similar document chunks using vector similarity search
   - Compile relevant chunks into a context window

3. **Answer Generation**:
   - Construct a prompt containing retrieved context and user question
   - Generate a natural language response using TinyLlama
   - Return the answer to the user

## Code Architecture
The solution is implemented as a single class with a clean, modular design:
- `__init__`: Initialize models and storage
- `load_pdf`: Process PDF documents and build the search index
- `query`: Handle user questions and generate answers

## Performance Considerations
- Uses a lightweight embedding model (MiniLM) with 384-dimensional vectors for optimal speed/quality balance
- Implements FAISS for sub-linear time complexity in similarity search
- Employs TinyLlama (1.1B parameters) for efficient generation while maintaining reasonable quality
- Simple chunking strategy that balances context preservation with retrieval granularity

## Potential Improvements
- Implement more sophisticated text chunking strategies
- Add support for multiple document formats (DOC, TXT, HTML, etc.)
- Enable incremental indexing for large document collections
- Incorporate reranking to improve retrieval precision
- Add caching mechanisms to improve response time for repeated queries
- Implement metadata filtering for more targeted retrievals

## Usage Example
```python
# Initialize the RAG system
rag = SimpleRAG()

# Load a PDF document
rag.load_pdf('document_input.pdf')

# Ask questions about the document
question = "What is the main topic of the document?"
answer = rag.query(question)
print(f"Answer: {answer}")
```

## Dependencies
```
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
transformers>=4.30.0
torch>=2.0.0
PyPDF2>=3.0.0
```
---

*This project demonstrates practical knowledge of modern NLP techniques, information retrieval, and language model integration - key skills for AI/ML engineering roles focused on natural language processing and information systems.*
