# Multi-Modal NLP Chatbot

A comprehensive, internship-ready NLP chatbot with multiple advanced features, built entirely using open-source models without any API keys.

## Features

### 🤖 Core Capabilities
- **Dynamic Knowledge Base**: Vector database (ChromaDB) with automatic updates
- **Multi-Modal AI**: Text and image understanding using CLIP and BLIP models
- **Sentiment Analysis**: Emotion detection and appropriate response generation
- **Context Management**: Maintains conversation history and context

### 🏥 Medical Q&A System
- Integration with MedQuAD dataset
- Medical entity recognition (symptoms, diseases, treatments)
- Retrieval-based question answering
- Specialized medical knowledge base

### 🔬 Scientific Domain Expert
- arXiv dataset integration for scientific papers
- Paper search and summarization
- Concept explanation and visualization
- Advanced NLP for information extraction

### 🔄 Automatic Knowledge Updates
- Periodic knowledge base updates
- Support for multiple data sources
- Custom document ingestion
- Configurable update intervals

## Architecture

```
NLP_BOT/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── chatbot_core.py          # Core chatbot integration
├── vector_store.py          # ChromaDB vector database management
├── multi_modal.py           # Multi-modal AI (CLIP/BLIP)
├── sentiment_analyzer.py    # Sentiment analysis
├── medical_qa.py           # Medical Q&A system
├── scientific_chatbot.py    # Scientific chatbot
├── knowledge_updater.py     # Periodic update mechanism
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (no API keys)
├── data/                   # Data directory
├── models/                 # Model cache directory
├── vector_db/             # Vector database storage
└── datasets/              # Dataset storage
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space

### Step 1: Clone/Download the Project
```bash
cd NLP_BOT
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Install NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 6: (Optional) Install Ollama for Enhanced LLM
For better responses, install Ollama:
```bash
# Download from https://ollama.ai
# After installation, run:
ollama pull llama3
```

## Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Chat Modes

1. **General Mode**: General knowledge and conversation
2. **Medical Mode**: Medical Q&A with MedQuAD dataset
3. **Scientific Mode**: Scientific research and paper analysis
4. **Image Mode**: Multi-modal image analysis

### Features in the UI

- **Chat Interface**: Interactive chat with message history
- **Sentiment Detection**: Real-time emotion analysis
- **Image Analysis**: Upload images for AI analysis
- **Knowledge Base Management**: Add custom knowledge
- **System Statistics**: View system performance metrics
- **Quick Actions**: Test different features

## Project Structure Details

### Core Modules

#### `config.py`
- Configuration settings for all components
- Model configurations
- Path management
- No API keys required

#### `vector_store.py`
- ChromaDB integration for vector storage
- Document addition and retrieval
- Knowledge base updater
- Automatic ID generation

#### `multi_modal.py`
- BLIP model for image captioning
- CLIP model for image-text understanding
- Visual question answering
- Image classification

#### `sentiment_analyzer.py`
- DistilBERT-based sentiment analysis
- Emotion detection (positive/negative/neutral)
- Sentiment-aware response generation
- Conversation sentiment tracking

#### `medical_qa.py`
- MedQuAD dataset integration
- Medical entity recognition
- Retrieval-based Q&A
- Category-based organization

#### `scientific_chatbot.py`
- arXiv dataset integration
- Paper search and summarization
- Concept explanation
- Keyword extraction

#### `chatbot_core.py`
- Main integration of all components
- Message processing pipeline
- Mode selection
- Conversation management

#### `knowledge_updater.py`
- Periodic update scheduling
- Multiple data source support
- Custom document ingestion
- Update status tracking

## Datasets

### MedQuAD Dataset
Download from: https://github.com/abachaa/MedQuAD
Place in: `data/medquad_data.json`

### arXiv Dataset
Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv
Place in: `datasets/arxiv_metadata.json`

**Note**: The system includes sample data for immediate testing. Replace with full datasets for production use.

## Customization

### Adding Custom Knowledge
1. Use the "Add Custom Knowledge" section in the UI
2. Or add documents programmatically:
```python
from chatbot_core import NLPChatbot

chatbot = NLPChatbot()
chatbot.update_knowledge_base(
    [{'text': 'Your custom text', 'id': 'custom_1'}],
    [{'source': 'custom', 'category': 'general'}]
)
```

### Configuring Update Intervals
Edit `config.py`:
```python
UPDATE_INTERVAL_HOURS = 24  # Change as needed
```

### Changing Models
Edit `config.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
```

## Performance Optimization

### Memory Management
- Models are loaded on-demand
- Vector database uses persistent storage
- Conversation history can be cleared

### GPU Acceleration
- Automatically uses CUDA if available
- Falls back to CPU if GPU not present
- Models cached in `models/` directory

### Batch Processing
- Supports batch document addition
- Efficient vector embeddings
- Optimized similarity search

## Troubleshooting

### Common Issues

**Issue**: "Module not found" errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Out of memory errors
**Solution**: 
- Close other applications
- Use CPU-only mode if GPU memory is limited
- Clear conversation history regularly

**Issue**: Slow response times
**Solution**:
- Install Ollama for faster LLM responses
- Use smaller embedding models
- Limit knowledge base size

**Issue**: Image analysis not working
**Solution**: Ensure PIL and torch are properly installed

## API-Free Design

This project is designed to work entirely without API keys:
- **Embeddings**: Sentence Transformers (local)
- **Sentiment**: DistilBERT (local)
- **Image AI**: BLIP and CLIP (local)
- **LLM**: Ollama (local, optional)
- **Vector DB**: ChromaDB (local)

## Evaluation Criteria

### Sentiment Analysis
- Accuracy of emotion detection
- Appropriate responses to different sentiments
- Conversation sentiment tracking

### Medical Q&A
- Retrieval accuracy
- Entity recognition precision
- Response relevance

### Scientific Chatbot
- Paper search relevance
- Summary quality
- Concept explanation clarity

### Multi-Modal
- Image caption accuracy
- Visual question answering
- Image-text similarity

## Future Enhancements

- [ ] Support for more datasets
- [ ] Advanced RAG implementation
- [ ] Voice input/output
- [ ] Real-time web scraping
- [ ] Multi-language support
- [ ] Advanced visualization

## License

This project is for educational and internship purposes.

## Contributing

This is an internship-ready project. Feel free to extend and customize for your requirements.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

## Demo and Page
<img width="936" height="576" alt="image" src="https://github.com/user-attachments/assets/5c0cdd5d-37c6-4863-ac72-de349ca27551" />

**Built with open-source technologies only - No API keys required!**
