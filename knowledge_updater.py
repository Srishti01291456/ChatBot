"""
Periodic Knowledge Base Update Mechanism
Automatically updates vector database with new information from specified sources
"""
import schedule
import time
import threading
from pathlib import Path
import json
import logging
from typing import List, Dict, Callable
from datetime import datetime

from config import DATA_DIR, DATASETS_DIR, UPDATE_INTERVAL_HOURS
from vector_store import VectorStore, KnowledgeBaseUpdater
from chatbot_core import NLPChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource:
    """Base class for data sources"""
    
    def fetch_new_data(self) -> List[Dict]:
        """Fetch new data from the source"""
        raise NotImplementedError
    
    def get_source_name(self) -> str:
        """Return the name of the data source"""
        raise NotImplementedError


class FileDataSource(DataSource):
    """Data source that reads from files"""
    
    def __init__(self, file_path: Path, source_name: str = "file"):
        self.file_path = file_path
        self.source_name = source_name
        self.last_modified = None
    
    def fetch_new_data(self) -> List[Dict]:
        """Check for file modifications and return new data"""
        if not self.file_path.exists():
            logger.warning(f"File not found: {self.file_path}")
            return []
        
        current_modified = self.file_path.stat().st_mtime
        
        if self.last_modified and current_modified <= self.last_modified:
            logger.info(f"No new data in {self.file_path}")
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.last_modified = current_modified
            logger.info(f"Fetched {len(data)} items from {self.file_path}")
            return data
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []
    
    def get_source_name(self) -> str:
        return self.source_name


class PeriodicUpdater:
    """Manages periodic updates to the knowledge base"""
    
    def __init__(self, chatbot: NLPChatbot):
        self.chatbot = chatbot
        self.data_sources: List[DataSource] = []
        self.update_thread = None
        self.running = False
        self.update_callbacks: List[Callable] = []
    
    def add_data_source(self, source: DataSource):
        """Add a data source to monitor"""
        self.data_sources.append(source)
        logger.info(f"Added data source: {source.get_source_name()}")
    
    def add_update_callback(self, callback: Callable):
        """Add a callback function to be called after updates"""
        self.update_callbacks.append(callback)
    
    def perform_update(self):
        """Perform a knowledge base update from all sources"""
        logger.info("Starting knowledge base update...")
        
        total_documents_added = 0
        update_results = []
        
        for source in self.data_sources:
            try:
                new_data = source.fetch_new_data()
                
                if new_data:
                    # Convert data to document format
                    documents = []
                    metadata = []
                    
                    for item in new_data:
                        if isinstance(item, dict):
                            text = item.get('text', str(item))
                            doc_id = item.get('id', '')
                            
                            documents.append({
                                'text': text,
                                'id': doc_id
                            })
                            
                            meta = {
                                'source': source.get_source_name(),
                                'updated_at': datetime.now().isoformat()
                            }
                            meta.update({k: v for k, v in item.items() if k not in ['text', 'id']})
                            metadata.append(meta)
                    
                    # Add to knowledge base
                    count = self.chatbot.update_knowledge_base(documents, metadata)
                    total_documents_added += count
                    
                    update_results.append({
                        'source': source.get_source_name(),
                        'documents_added': count,
                        'status': 'success'
                    })
                    
                    logger.info(f"Added {count} documents from {source.get_source_name()}")
                else:
                    update_results.append({
                        'source': source.get_source_name(),
                        'documents_added': 0,
                        'status': 'no_new_data'
                    })
            
            except Exception as e:
                logger.error(f"Error updating from {source.get_source_name()}: {e}")
                update_results.append({
                    'source': source.get_source_name(),
                    'documents_added': 0,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Call callbacks
        for callback in self.update_callbacks:
            try:
                callback(update_results, total_documents_added)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
        
        logger.info(f"Update complete. Total documents added: {total_documents_added}")
        
        return {
            'total_documents_added': total_documents_added,
            'results': update_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def start_scheduled_updates(self, interval_hours: int = None):
        """Start scheduled updates in a background thread"""
        if interval_hours is None:
            interval_hours = UPDATE_INTERVAL_HOURS
        
        if self.running:
            logger.warning("Updater already running")
            return
        
        self.running = True
        
        def run_scheduler():
            schedule.every(interval_hours).hours.do(self.perform_update)
            
            # Run once immediately
            self.perform_update()
            
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.update_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.update_thread.start()
        logger.info(f"Started scheduled updates (interval: {interval_hours} hours)")
    
    def stop_scheduled_updates(self):
        """Stop scheduled updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Stopped scheduled updates")
    
    def manual_update(self) -> Dict:
        """Trigger a manual update"""
        logger.info("Triggering manual update...")
        return self.perform_update()


class KnowledgeBaseManager:
    """High-level manager for knowledge base operations"""
    
    def __init__(self, chatbot: NLPChatbot):
        self.chatbot = chatbot
        self.updater = PeriodicUpdater(chatbot)
        self._setup_default_sources()
    
    def _setup_default_sources(self):
        """Setup default data sources"""
        # Medical data source
        medquad_file = DATA_DIR / "medquad_data.json"
        if medquad_file.exists():
            self.updater.add_data_source(FileDataSource(medquad_file, "medquad"))
        
        # Scientific papers source
        arxiv_file = DATASETS_DIR / "arxiv_metadata.json"
        if arxiv_file.exists():
            self.updater.add_data_source(FileDataSource(arxiv_file, "arxiv"))
    
    def start_auto_updates(self, interval_hours: int = None):
        """Start automatic periodic updates"""
        self.updater.start_scheduled_updates(interval_hours)
    
    def stop_auto_updates(self):
        """Stop automatic periodic updates"""
        self.updater.stop_scheduled_updates()
    
    def update_now(self) -> Dict:
        """Perform an immediate update"""
        return self.updater.manual_update()
    
    def add_custom_source(self, file_path: Path, source_name: str):
        """Add a custom data source"""
        self.updater.add_data_source(FileDataSource(file_path, source_name))
    
    def get_update_status(self) -> Dict:
        """Get the status of the updater"""
        return {
            'running': self.updater.running,
            'data_sources': [source.get_source_name() for source in self.updater.data_sources],
            'last_update': self.chatbot.knowledge_updater.last_update_file.exists()
        }


def create_sample_data_files():
    """Create sample data files for testing"""
    logger.info("Creating sample data files...")
    
    # Create sample medical data
    medquad_data = [
        {
            "id": "med_001",
            "text": "Question: What are the symptoms of COVID-19? Answer: Common symptoms of COVID-19 include fever, cough, fatigue, loss of taste or smell, and difficulty breathing.",
            "category": "infectious_disease",
            "focus": "symptoms"
        },
        {
            "id": "med_002",
            "text": "Question: How is influenza treated? Answer: Influenza is typically treated with rest, fluids, and over-the-counter medications to relieve symptoms. Antiviral drugs may be prescribed for high-risk patients.",
            "category": "infectious_disease",
            "focus": "treatment"
        }
    ]
    
    medquad_file = DATA_DIR / "medquad_data.json"
    with open(medquad_file, 'w', encoding='utf-8') as f:
        json.dump(medquad_data, f, indent=2)
    
    # Create sample arXiv data
    arxiv_data = [
        {
            "id": "arxiv_006",
            "text": "Title: Diffusion Models Beat GANs on Image Synthesis\nAbstract: We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models.",
            "categories": ["cs.CV", "cs.LG"],
            "published": "2021-07-01"
        },
        {
            "id": "arxiv_007",
            "text": "Title: LoRA: Low-Rank Adaptation of Large Language Models\nAbstract: We propose LoRA, an efficient adaptation method that freezes pre-trained weights and injects trainable rank decomposition matrices.",
            "categories": ["cs.CL", "cs.AI"],
            "published": "2021-10-06"
        }
    ]
    
    arxiv_file = DATASETS_DIR / "arxiv_metadata.json"
    with open(arxiv_file, 'w', encoding='utf-8') as f:
        json.dump(arxiv_data, f, indent=2)
    
    logger.info("Sample data files created successfully")


if __name__ == "__main__":
    # Test the updater
    create_sample_data_files()
    
    # Initialize chatbot
    chatbot = NLPChatbot()
    
    # Create manager
    manager = KnowledgeBaseManager(chatbot)
    
    # Perform manual update
    result = manager.update_now()
    print(f"Update result: {result}")
