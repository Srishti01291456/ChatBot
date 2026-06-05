"""
System Testing and Validation Script
Tests all components of the NLP Chatbot
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from config import DATA_DIR, DATASETS_DIR
from vector_store import VectorStore
from sentiment_analyzer import SentimentAnalyzer
from medical_qa import MedicalQA
from scientific_chatbot import ScientificChatbot
from knowledge_updater import create_sample_data_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_store():
    """Test vector database functionality"""
    print("\n" + "="*60)
    print("TESTING: Vector Store")
    print("="*60)
    
    try:
        vector_store = VectorStore("./test_vector_db", "test_collection")
        
        # Test adding documents
        docs = [
            {'text': 'This is a test document about artificial intelligence.', 'id': 'test1'},
            {'text': 'Machine learning is a subset of AI.', 'id': 'test2'}
        ]
        count = vector_store.add_documents(docs)
        print(f"✓ Added {count} documents")
        
        # Test search
        results = vector_store.search("artificial intelligence", n_results=2)
        print(f"✓ Found {len(results)} results")
        
        # Test stats
        stats = vector_store.get_collection_stats()
        print(f"✓ Collection stats: {stats}")
        
        # Clean up
        vector_store.clear_collection()
        print("✓ Vector store test passed")
        return True
        
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False


def test_sentiment_analyzer():
    """Test sentiment analysis"""
    print("\n" + "="*60)
    print("TESTING: Sentiment Analyzer")
    print("="*60)
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Test positive sentiment
        result = analyzer.analyze_sentiment("I love this product, it's amazing!")
        print(f"✓ Positive sentiment: {result['dominant_sentiment']} (confidence: {result['confidence']:.2f})")
        
        # Test negative sentiment
        result = analyzer.analyze_sentiment("This is terrible, I hate it.")
        print(f"✓ Negative sentiment: {result['dominant_sentiment']} (confidence: {result['confidence']:.2f})")
        
        # Test neutral sentiment
        result = analyzer.analyze_sentiment("The product is okay.")
        print(f"✓ Neutral sentiment: {result['dominant_sentiment']} (confidence: {result['confidence']:.2f})")
        
        # Test emotional response
        response = analyzer.get_emotional_response(result)
        print(f"✓ Emotional response: {response}")
        
        print("✓ Sentiment analyzer test passed")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment analyzer test failed: {e}")
        return False


def test_medical_qa():
    """Test medical Q&A system"""
    print("\n" + "="*60)
    print("TESTING: Medical Q&A System")
    print("="*60)
    
    try:
        # Create sample data
        create_sample_data_files()
        
        # Initialize
        vector_store = VectorStore("./test_vector_db", "medical_test")
        medical_qa = MedicalQA(vector_store, DATA_DIR)
        
        # Load data
        count = medical_qa.load_medquad_data()
        print(f"✓ Loaded {count} medical Q&A pairs")
        
        # Test question
        result = medical_qa.answer_medical_question("What are the symptoms of diabetes?")
        print(f"✓ Answer generated: {result['status']}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        
        # Test entity recognition
        entities = medical_qa.entity_recognizer.extract_entities(
            "The patient has fever and cough, possibly an infection."
        )
        print(f"✓ Entities detected: {entities}")
        
        # Clean up
        vector_store.clear_collection()
        print("✓ Medical Q&A test passed")
        return True
        
    except Exception as e:
        print(f"✗ Medical Q&A test failed: {e}")
        return False


def test_scientific_chatbot():
    """Test scientific chatbot"""
    print("\n" + "="*60)
    print("TESTING: Scientific Chatbot")
    print("="*60)
    
    try:
        # Initialize
        vector_store = VectorStore("./test_vector_db", "scientific_test")
        scientific_chatbot = ScientificChatbot(vector_store, DATASETS_DIR)
        
        # Load data
        count = scientific_chatbot.load_arxiv_data()
        print(f"✓ Loaded {count} scientific papers")
        
        # Test paper search
        results = scientific_chatbot.search_papers("transformer", n_results=3)
        print(f"✓ Found {len(results)} papers")
        
        # Test concept explanation
        explanation = scientific_chatbot.explain_concept("attention mechanism")
        print(f"✓ Concept explanation: {explanation['status']}")
        
        # Test paper summary
        summary = scientific_chatbot.get_paper_summary("Attention Is All You Need")
        print(f"✓ Paper summary: {summary['status']}")
        
        # Clean up
        vector_store.clear_collection()
        print("✓ Scientific chatbot test passed")
        return True
        
    except Exception as e:
        print(f"✗ Scientific chatbot test failed: {e}")
        return False


def test_integration():
    """Test full integration"""
    print("\n" + "="*60)
    print("TESTING: Full Integration")
    print("="*60)
    
    try:
        from chatbot_core import NLPChatbot
        
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = NLPChatbot()
        
        # Test general query
        print("\nTesting general query...")
        response = chatbot.process_message("What is artificial intelligence?", "general")
        print(f"✓ General query: {response['status']}")
        
        # Test medical query
        print("\nTesting medical query...")
        response = chatbot.process_message("What are the symptoms of diabetes?", "medical")
        print(f"✓ Medical query: {response['status']}")
        
        # Test scientific query
        print("\nTesting scientific query...")
        response = chatbot.process_message("Explain transformer architecture", "scientific")
        print(f"✓ Scientific query: {response['status']}")
        
        # Test system stats
        stats = chatbot.get_system_stats()
        print(f"✓ System stats retrieved")
        print(f"  Total documents: {stats['vector_db']['total_documents']}")
        print(f"  LLM available: {stats['llm_available']}")
        
        # Test conversation summary
        summary = chatbot.get_conversation_summary()
        print(f"✓ Conversation summary: {summary['message_count']} messages")
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("NLP CHATBOT SYSTEM TESTS")
    print("="*60)
    
    results = {
        'Vector Store': test_vector_store(),
        'Sentiment Analyzer': test_sentiment_analyzer(),
        'Medical Q&A': test_medical_qa(),
        'Scientific Chatbot': test_scientific_chatbot(),
        'Full Integration': test_integration()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nTo run the application:")
        print("  streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
