"""
Sentiment Analysis Module
Detects and responds to user emotions using open-source models
"""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment in text using open-source models"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        logger.info(f"Loading sentiment model: {model_name}")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True
        )
        logger.info("Sentiment analyzer loaded successfully")
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        try:
            results = self.sentiment_pipeline(text)[0]
            
            # Convert to dictionary
            sentiment_scores = {result['label']: result['score'] for result in results}
            
            # Get dominant sentiment
            dominant = max(sentiment_scores.items(), key=lambda x: x[1])
            
            return {
                'dominant_sentiment': dominant[0].lower(),
                'confidence': dominant[1],
                'scores': sentiment_scores,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'dominant_sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0},
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def get_emotional_response(self, sentiment: Dict) -> str:
        """
        Generate appropriate response based on sentiment
        
        Args:
            sentiment: Sentiment analysis result
            
        Returns:
            Appropriate response string
        """
        dominant = sentiment.get('dominant_sentiment', 'neutral')
        confidence = sentiment.get('confidence', 0.0)
        
        if dominant == 'positive' and confidence > 0.7:
            return "I'm glad to hear that! How can I help you further?"
        elif dominant == 'negative' and confidence > 0.7:
            return "I understand you're concerned. Let me try my best to help you with this."
        elif dominant == 'neutral':
            return "I'm here to help. Please let me know what you'd like to know."
        else:
            return "Thank you for your message. How can I assist you today?"
    
    def track_conversation_sentiment(self, messages: List[Dict]) -> Dict:
        """
        Track sentiment trends across a conversation
        
        Args:
            messages: List of message dictionaries with 'text' field
            
        Returns:
            Dictionary with sentiment trends
        """
        if not messages:
            return {'trend': 'neutral', 'average_scores': {}}
        
        sentiments = [self.analyze_sentiment(msg.get('text', '')) for msg in messages]
        
        # Calculate average scores
        avg_positive = sum(s.get('scores', {}).get('POSITIVE', 0) for s in sentiments) / len(sentiments)
        avg_negative = sum(s.get('scores', {}).get('NEGATIVE', 0) for s in sentiments) / len(sentiments)
        
        # Determine trend
        if avg_positive > avg_negative + 0.2:
            trend = 'improving'
        elif avg_negative > avg_positive + 0.2:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'average_scores': {
                'positive': avg_positive,
                'negative': avg_negative
            },
            'message_count': len(messages)
        }
