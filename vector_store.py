"""
Vector Database Management using ChromaDB
Handles dynamic knowledge base expansion and retrieval
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector database for knowledge base"""
    
    def __init__(self, persist_directory: str, collection_name: str = "knowledge_base"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized at {persist_directory}")
    
    def add_documents(self, documents: List[Dict[str, str]], 
                     metadata: Optional[List[Dict]] = None) -> int:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents with 'text' and 'id' fields
            metadata: Optional metadata for each document
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Generate embeddings
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Generate IDs if not provided
        ids = [doc.get('id', self._generate_id(doc['text'])) for doc in documents]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{"source": "unknown", "added_at": datetime.now().isoformat()} 
                       for _ in documents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return len(documents)
    
    def search(self, query: str, n_results: int = 5, 
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents with scores
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'id': results['ids'][0][i] if results['ids'] else ''
                })
        
        return formatted_results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name
        }
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


class KnowledgeBaseUpdater:
    """Handles periodic updates to the knowledge base"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.last_update_file = Path(vector_store.persist_directory) / "last_update.txt"
    
    def should_update(self, interval_hours: int = 24) -> bool:
        """Check if update is needed based on time interval"""
        if not self.last_update_file.exists():
            return True
        
        try:
            with open(self.last_update_file, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
            
            hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
            return hours_since_update >= interval_hours
        except Exception as e:
            logger.error(f"Error checking update status: {e}")
            return True
    
    def mark_updated(self):
        """Mark that an update has occurred"""
        with open(self.last_update_file, 'w') as f:
            f.write(datetime.now().isoformat())
        logger.info("Update marked complete")
    
    def add_new_documents(self, documents: List[Dict], 
                         metadata: Optional[List[Dict]] = None) -> int:
        """Add new documents and mark as updated"""
        count = self.vector_store.add_documents(documents, metadata)
        if count > 0:
            self.mark_updated()
        return count
