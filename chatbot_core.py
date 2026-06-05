"""
Main Chatbot Integration Module
Integrates all components: vector store, multi-modal, medical Q&A, scientific chatbot, sentiment analysis
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from config import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME, DATA_DIR, DATASETS_DIR,
    LLM_MODEL, OLLAMA_BASE_URL, UPDATE_INTERVAL_HOURS
)
from vector_store import VectorStore, KnowledgeBaseUpdater
from multi_modal import MultiModalAssistant
from sentiment_analyzer import SentimentAnalyzer
from medical_qa import MedicalQA
from scientific_chatbot import ScientificChatbot

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. LLM features will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPChatbot:
    """Main NLP Chatbot integrating all features"""
    
    def __init__(self):
        logger.info("Initializing NLP Chatbot...")
        
        # Initialize vector store
        self.vector_store = VectorStore(CHROMA_PERSIST_DIR, COLLECTION_NAME)
        self.knowledge_updater = KnowledgeBaseUpdater(self.vector_store)
        
        # Initialize multi-modal assistant (lazy loading)
        self.multi_modal = None
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize medical Q&A
        self.medical_qa = MedicalQA(self.vector_store, DATA_DIR)
        
        # Initialize scientific chatbot
        self.scientific_chatbot = ScientificChatbot(self.vector_store, DATASETS_DIR)
        
        # Initialize LLM if available
        self.llm = None
        if OLLAMA_AVAILABLE:
            try:
                self.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
                logger.info(f"LLM initialized: {LLM_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize Ollama: {e}")
        
        # Conversation history
        self.conversation_history = []
        
        # Load initial data
        self._load_initial_data()
        
        logger.info("NLP Chatbot initialized successfully")
    
    def _load_initial_data(self):
        """Load initial data into knowledge bases"""
        logger.info("Loading initial data...")
        
        # Load medical Q&A data
        med_count = self.medical_qa.load_medquad_data()
        logger.info(f"Loaded {med_count} medical Q&A pairs")
        
        # Load scientific papers
        sci_count = self.scientific_chatbot.load_arxiv_data()
        logger.info(f"Loaded {sci_count} scientific papers")
    
    def initialize_multi_modal(self):
        """Initialize multi-modal assistant (lazy loading)"""
        if self.multi_modal is None:
            logger.info("Initializing multi-modal assistant...")
            self.multi_modal = MultiModalAssistant()
    
    def process_message(self, message: str, mode: str = "general") -> Dict:
        """
        Process a user message based on the selected mode
        
        Args:
            message: User message
            mode: Processing mode (general, medical, scientific, image)
            
        Returns:
            Response dictionary
        """
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(message)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'text': message,
            'timestamp': datetime.now().isoformat(),
            'sentiment': sentiment
        })
        
        # Process based on mode
        if mode == "medical":
            response = self._process_medical_query(message)
        elif mode == "scientific":
            response = self._process_scientific_query(message)
        elif mode == "image":
            response = self._process_image_query(message)
        else:
            response = self._process_general_query(message)
        
        # Add sentiment-aware response
        emotional_response = self.sentiment_analyzer.get_emotional_response(sentiment)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'text': response.get('answer', ''),
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'sentiment_response': emotional_response
        })
        
        # Combine response with emotional context
        response['sentiment'] = sentiment
        response['emotional_context'] = emotional_response
        
        return response
    
    def _process_general_query(self, query: str) -> Dict:
        """Process general query using vector search and LLM"""
        # Search knowledge base
        results = self.vector_store.search(query, n_results=3)
        
        if results and results[0]['distance'] < 0.5:
            # Good match found
            answer = results[0]['text']
            confidence = 1 - results[0]['distance']
            
            # Use LLM to enhance if available
            if self.llm and confidence < 0.8:
                try:
                    context = "\n".join([r['text'] for r in results])
                    prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide a helpful answer based on the context."
                    enhanced_answer = self.llm.invoke(prompt)
                    answer = enhanced_answer
                except Exception as e:
                    logger.warning(f"LLM enhancement failed: {e}")
            
            return {
                'answer': answer,
                'confidence': confidence,
                'source': results[0]['metadata'].get('source', 'knowledge_base'),
                'related_info': [r['text'] for r in results[1:]],
                'status': 'success'
            }
        else:
            # No good match, use LLM if available
            if self.llm:
                try:
                    answer = self.llm.invoke(query)
                    return {
                        'answer': answer,
                        'confidence': 0.5,
                        'source': 'llm',
                        'status': 'success'
                    }
                except Exception as e:
                    logger.warning(f"LLM failed: {e}")
            
            return {
                'answer': "I don't have specific information about that in my knowledge base. Could you rephrase your question or try a different mode?",
                'confidence': 0.0,
                'source': 'none',
                'status': 'no_match'
            }
    
    def _process_medical_query(self, query: str) -> Dict:
        """Process medical query"""
        return self.medical_qa.answer_medical_question(query)
    
    def _process_scientific_query(self, query: str) -> Dict:
        """Process scientific query"""
        # Check if it's a concept explanation request
        if "explain" in query.lower() or "what is" in query.lower():
            concept = query.replace("explain", "").replace("what is", "").strip()
            return self.scientific_chatbot.explain_concept(concept)
        else:
            # Search for papers
            papers = self.scientific_chatbot.search_papers(query, n_results=5)
            if papers:
                return {
                    'answer': f"Found {len(papers)} relevant papers:\n\n" + 
                             "\n\n".join([f"- {p['metadata']['title']}" for p in papers]),
                    'papers': papers,
                    'status': 'success'
                }
            else:
                return {
                    'answer': "No relevant papers found in the database.",
                    'status': 'no_match'
                }
    
    def _process_image_query(self, query: str) -> Dict:
        """Process image-related query"""
        if self.multi_modal is None:
            self.initialize_multi_modal()
        
        return {
            'answer': "Please upload an image for analysis. I can describe images, answer questions about them, and classify visual content.",
            'status': 'awaiting_image',
            'capabilities': [
                'Image description and captioning',
                'Visual question answering',
                'Image classification',
                'Image-text similarity'
            ]
        }
    
    def analyze_image(self, image, query: Optional[str] = None) -> Dict:
        """
        Analyze an uploaded image
        
        Args:
            image: PIL Image object
            query: Optional question about the image
            
        Returns:
            Analysis result
        """
        if self.multi_modal is None:
            self.initialize_multi_modal()
        
        if query:
            result = self.multi_modal.answer_visual_question(image, query)
        else:
            context = self.multi_modal.extract_visual_context(image)
            result = context['caption']
        
        return {
            'answer': result,
            'status': 'success'
        }
    
    def update_knowledge_base(self, documents: List[Dict], 
                            metadata: Optional[List[Dict]] = None) -> int:
        """
        Update the knowledge base with new documents
        
        Args:
            documents: List of documents to add
            metadata: Optional metadata for documents
            
        Returns:
            Number of documents added
        """
        return self.knowledge_updater.add_new_documents(documents, metadata)
    
    def check_for_updates(self) -> bool:
        """
        Check if knowledge base needs updating
        
        Returns:
            True if update is needed
        """
        return self.knowledge_updater.should_update(UPDATE_INTERVAL_HOURS)
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        if not self.conversation_history:
            return {'message_count': 0, 'sentiment_trend': 'neutral'}
        
        sentiment_trend = self.sentiment_analyzer.track_conversation_sentiment(
            self.conversation_history
        )
        
        return {
            'message_count': len(self.conversation_history),
            'sentiment_trend': sentiment_trend['trend'],
            'modes_used': list(set(msg.get('mode', 'general') for msg in self.conversation_history))
        }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation cleared")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            'vector_db': vector_stats,
            'conversation_length': len(self.conversation_history),
            'medical_categories': self.medical_qa.get_medical_categories(),
            'llm_available': OLLAMA_AVAILABLE,
            'multi_modal_loaded': self.multi_modal is not None
        }
