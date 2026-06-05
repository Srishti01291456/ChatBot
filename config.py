"""
Configuration file for NLP Chatbot
No API keys required - uses open-source models only
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DATASETS_DIR = BASE_DIR / "datasets"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, VECTOR_DB_DIR, DATASETS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
IMAGE_MODEL = "Salesforce/blip-image-captioning-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# LLM Configuration (using Ollama)
LLM_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# Vector Database Configuration
CHROMA_PERSIST_DIR = str(VECTOR_DB_DIR / "chroma_db")
COLLECTION_NAME = "knowledge_base"

# Dataset configurations
MEDQUAD_URL = "https://github.com/abachaa/MedQuAD"
ARXIV_DATASET_PATH = DATASETS_DIR / "arxiv_metadata.json"

# Update configurations
UPDATE_INTERVAL_HOURS = 24  # Check for updates every 24 hours
MAX_DOCUMENTS_PER_UPDATE = 100

# UI Configuration
APP_TITLE = "Customer Service Chatbot"
APP_DESCRIPTION = "Learn to Build an AI-Powered Assistant"
