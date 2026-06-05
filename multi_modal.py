"""
Multi-modal AI Assistant
Handles text and image processing using CLIP and BLIP models
"""
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalAssistant:
    """Multi-modal AI assistant for text and image understanding"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize BLIP for image captioning
        logger.info("Loading BLIP model for image captioning...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        
        # Initialize CLIP for image-text matching
        logger.info("Loading CLIP model for image-text understanding...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        logger.info("Multi-modal models loaded successfully")
    
    def analyze_image(self, image: Image.Image, 
                     prompt: Optional[str] = None) -> Dict[str, str]:
        """
        Analyze an image and generate description
        
        Args:
            image: PIL Image object
            prompt: Optional text prompt for conditional captioning
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Generate caption
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            if prompt:
                inputs = self.blip_processor(image, text=prompt, return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            else:
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return {
                'caption': caption,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'caption': f"Error analyzing image: {str(e)}",
                'status': 'error'
            }
    
    def image_text_similarity(self, image: Image.Image, 
                            text: str) -> float:
        """
        Calculate similarity between image and text using CLIP
        
        Args:
            image: PIL Image object
            text: Text to compare
            
        Returns:
            Similarity score
        """
        try:
            inputs = self.clip_processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarity = logits_per_image.cpu().detach().numpy()[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def classify_image(self, image: Image.Image, 
                      labels: List[str]) -> Dict[str, float]:
        """
        Classify image against given labels using CLIP
        
        Args:
            image: PIL Image object
            labels: List of possible labels
            
        Returns:
            Dictionary of label -> probability
        """
        try:
            inputs = self.clip_processor(
                text=labels, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]
            
            return {label: float(prob) for label, prob in zip(labels, probs)}
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return {label: 0.0 for label in labels}
    
    def answer_visual_question(self, image: Image.Image, 
                             question: str) -> str:
        """
        Answer a question about an image
        
        Args:
            image: PIL Image object
            question: Question about the image
            
        Returns:
            Answer to the question
        """
        try:
            # Use the question as a prompt for conditional captioning
            prompt = f"Question: {question} Answer:"
            result = self.analyze_image(image, prompt)
            return result['caption']
        except Exception as e:
            logger.error(f"Error answering visual question: {e}")
            return f"Error answering question: {str(e)}"
    
    def extract_visual_context(self, image: Image.Image) -> Dict:
        """
        Extract comprehensive visual context from image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with visual context information
        """
        try:
            # Get basic caption
            basic_caption = self.analyze_image(image)['caption']
            
            # Try to identify common categories
            categories = ["indoor", "outdoor", "person", "animal", "vehicle", "food", 
                         "nature", "building", "technology", "art"]
            classification = self.classify_image(image, categories)
            
            # Get top categories
            top_categories = sorted(classification.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'caption': basic_caption,
                'detected_categories': top_categories,
                'image_size': image.size,
                'image_mode': image.mode
            }
        except Exception as e:
            logger.error(f"Error extracting visual context: {e}")
            return {
                'caption': "Could not analyze image",
                'detected_categories': [],
                'error': str(e)
            }
