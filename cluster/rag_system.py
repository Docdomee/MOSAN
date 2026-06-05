import os
import glob
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentChunker:
    """Chunks markdown documents into logical paragraphs or sections."""
    
    @staticmethod
    def chunk_markdown(file_path: str, chunk_size: int = 500) -> List[str]:
        """
        Reads a markdown file and splits it into chunks.
        Uses a basic paragraph split to maintain semantic coherence.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newline (paragraphs)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            chunks = []
            current_chunk = ""
            
            for p in paragraphs:
                if len(current_chunk) + len(p) < chunk_size:
                    current_chunk += p + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = p + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return chunks
            
        except Exception as e:
            print(f"[DocumentChunker] Error processing {file_path}: {e}")
            return []

class TfIdfRetriever:
    """
    A lightweight, zero-dependency (other than sklearn) RAG engine.
    Uses TF-IDF and Cosine Similarity to find relevant document chunks.
    Includes simple modification-time caching.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # Save exported json alongside the reports
        self.export_path = os.path.join(os.path.dirname(self.data_dir), "rag_dataset.json") 
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.chunks_data = [] # List of dicts: {"text": chunk, "source": file_path}
        
        # Cache tracking
        self.last_indexed_files: Dict[str, float] = {}
        
    def _get_current_files_mtime(self) -> Dict[str, float]:
        """Gets all markdown files and their modification times."""
        if not os.path.exists(self.data_dir):
            return {}
            
        md_files = glob.glob(os.path.join(self.data_dir, "*.md"))
        return {f: os.path.getmtime(f) for f in md_files}
        
    def _needs_reindex(self) -> bool:
        """Checks if files have changed since last index."""
        current_files = self._get_current_files_mtime()
        
        if not current_files and not self.last_indexed_files:
            return False # Nothing to index
            
        if set(current_files.keys()) != set(self.last_indexed_files.keys()):
            return True # Files added or removed
            
        for f, mtime in current_files.items():
            if mtime > self.last_indexed_files.get(f, 0):
                return True # file modified
                
        return False
        
    def index_documents(self, force: bool = False):
        """Builds the TF-IDF index if needed."""
        if not force and not self._needs_reindex():
            return
            
        self.chunks_data = []
        current_files = self._get_current_files_mtime()
        
        for file_path in current_files.keys():
            chunks = DocumentChunker.chunk_markdown(file_path)
            for chunk in chunks:
                self.chunks_data.append({
                    "text": chunk,
                    "source": os.path.basename(file_path)
                })
                
        if not self.chunks_data:
            self.tfidf_matrix = None
            self.last_indexed_files = current_files
            return
            
        # Fit TF-IDF
        texts = [c["text"] for c in self.chunks_data]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.last_indexed_files = current_files
        print(f"[TfIdfRetriever] Indexed {len(current_files)} files into {len(self.chunks_data)} chunks.")
        
        # Export chunks for LLM Fine-Tuning Analysis as requested
        try:
            import json
            with open(self.export_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_data, f, indent=4)
        except Exception as e:
            print(f"[TfIdfRetriever] Failed to export rag_dataset.json: {e}")
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the index for the query and returns the top_k chunks.
        """
        self.index_documents() # Will auto-skip if no changes
        
        if self.tfidf_matrix is None or not self.chunks_data:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.0: # Only return actual matches
                result = self.chunks_data[idx].copy()
                result["score"] = round(score, 4)
                results.append(result)
                
        return results
