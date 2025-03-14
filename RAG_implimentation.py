from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import PyPDF2
import textwrap

class SimpleRAG:
    def __init__(self):
        # Initialize embedding model
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize LLM for answer generation
        self.llm = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        
        # Storage for documents and index
        self.documents = []
        self.index = None
        
    def load_pdf(self, pdf_path):
        """Load and chunk PDF into text segments"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
                
        # Simple chunking by paragraphs
        chunks = text.split('\n\n')
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        self.documents = chunks
        
        # Create embeddings and index
        embeddings = self.embed_model.encode(chunks)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
    def query(self, question, k=3):
        """Query the document and generate answer"""
        # Get question embedding
        question_embedding = self.embed_model.encode([question])
        
        # Find similar chunks
        distances, indices = self.index.search(question_embedding.astype('float32'), k)
        
        # Get relevant chunks
        relevant_chunks = [self.documents[i] for i in indices[0]]
        
        # Create prompt
        prompt = f"""Use the following context to answer the question.
        
Context:
{' '.join(relevant_chunks)}

Question: {question}

Answer:"""
        
        # Generate answer
        response = self.llm(prompt, max_length=800, num_return_sequences=1)
        
        return response[0]['generated_text']

# Example usage
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # Load a PDF
    rag.load_pdf('example.pdf')
    
    # Ask a question
    question = "What is the main topic of the document?"
    answer = rag.query(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {textwrap.fill(answer, width=80)}")