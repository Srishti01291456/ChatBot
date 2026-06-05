"""
Medical Q&A System using MedQuAD Dataset
Implements retrieval mechanism and medical entity recognition
"""
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import logging
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalEntityRecognizer:
    """Basic medical entity recognition using regex patterns"""
    
    def __init__(self):
        # Common medical patterns
        self.symptoms_pattern = r'(?:pain|ache|fever|cough|headache|nausea|vomiting|fatigue|dizziness|shortness of breath|chest pain|swelling|rash|itching)'
        self.diseases_pattern = r'(?:diabetes|hypertension|cancer|arthritis|asthma|depression|anxiety|infection|inflammation|syndrome|disease)'
        self.treatments_pattern = r'(?:medication|surgery|therapy|treatment|vaccine|antibiotic|painkiller|exercise|diet)'
        self.body_parts_pattern = r'(?:heart|lung|liver|kidney|stomach|brain|skin|bone|muscle|blood)'
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted entities by category
        """
        entities = {
            'symptoms': [],
            'diseases': [],
            'treatments': [],
            'body_parts': []
        }
        
        # Extract symptoms
        symptoms = re.findall(self.symptoms_pattern, text, re.IGNORECASE)
        entities['symptoms'] = list(set([s.lower() for s in symptoms]))
        
        # Extract diseases
        diseases = re.findall(self.diseases_pattern, text, re.IGNORECASE)
        entities['diseases'] = list(set([d.lower() for d in diseases]))
        
        # Extract treatments
        treatments = re.findall(self.treatments_pattern, text, re.IGNORECASE)
        entities['treatments'] = list(set([t.lower() for t in treatments]))
        
        # Extract body parts
        body_parts = re.findall(self.body_parts_pattern, text, re.IGNORECASE)
        entities['body_parts'] = list(set([b.lower() for b in body_parts]))
        
        return entities


class MedicalQA:
    """Medical Question Answering system using MedQuAD dataset"""
    
    def __init__(self, vector_store: VectorStore, data_dir: Path):
        self.vector_store = vector_store
        self.data_dir = data_dir
        self.entity_recognizer = MedicalEntityRecognizer()
        self.collection_name = "medical_qa"
        
        # Initialize medical collection
        self.client = vector_store.client
        self.medical_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Medical QA system initialized")
        logger.info(f"Collection name: {self.collection_name}")
        
        # Check if collection has data
        try:
            count = self.medical_collection.count()
            logger.info(f"Medical collection has {count} documents")
        except Exception as e:
            logger.warning(f"Could not check collection count: {e}")
    
    def load_medquad_data(self, data_path: Optional[Path] = None) -> int:
        """
        Load MedQuAD dataset into vector store
        
        Args:
            data_path: Path to MedQuAD data file
            
        Returns:
            Number of documents loaded
        """
        if data_path is None:
            data_path = self.data_dir / "medquad_data.json"
        
        if not data_path.exists():
            logger.warning(f"MedQuAD data not found at {data_path}")
            # Create sample data for demonstration
            return self._create_sample_medical_data()
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            metadata = []
            
            for item in data:
                if 'question' in item and 'answer' in item:
                    doc_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                    documents.append({
                        'text': doc_text,
                        'id': item.get('id', '')
                    })
                    metadata.append({
                        'source': 'MedQuAD',
                        'category': item.get('category', 'general'),
                        'focus': item.get('focus', '')
                    })
            
            # Add to medical collection directly
            texts = [doc['text'] for doc in documents]
            embeddings = self.vector_store.embedding_model.encode(texts).tolist()
            ids = [doc['id'] for doc in documents]
            
            self.medical_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata
            )
            
            logger.info(f"Loaded {len(documents)} medical Q&A pairs")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error loading MedQuAD data: {e}")
            return self._create_sample_medical_data()
    
    def _create_sample_medical_data(self) -> int:
        """Create sample medical data for demonstration"""
        sample_data = [
            {
                'question': 'What are the common symptoms of diabetes?',
                'answer': 'Common symptoms of diabetes include increased thirst (polydipsia), frequent urination (polyuria), extreme hunger (polyphagia), unexplained weight loss, fatigue, blurred vision, slow-healing sores, frequent infections such as gum, skin, or vaginal infections, and numbness or tingling in the hands or feet.',
                'category': 'diabetes',
                'focus': 'symptoms'
            },
            {
                'question': 'How is diabetes treated?',
                'answer': 'Diabetes treatment includes lifestyle changes like healthy eating, regular physical activity, and weight loss. For type 1 diabetes, insulin therapy is essential. For type 2 diabetes, medications like metformin, sulfonylureas, or insulin may be prescribed. Blood sugar monitoring is crucial for all diabetes management.',
                'category': 'diabetes',
                'focus': 'treatment'
            },
            {
                'question': 'What are the symptoms of hypertension?',
                'answer': 'Hypertension often has no symptoms, which is why it is called the silent killer. When symptoms do occur, they may include headaches, shortness of breath, nosebleeds, dizziness, or chest pain. Regular blood pressure checks are important for early detection.',
                'category': 'hypertension',
                'focus': 'symptoms'
            },
            {
                'question': 'How is hypertension treated?',
                'answer': 'Hypertension treatment involves lifestyle modifications such as reducing sodium intake, maintaining a healthy weight, regular exercise, limiting alcohol, and quitting smoking. Medications may include diuretics, ACE inhibitors, angiotensin II receptor blockers (ARBs), calcium channel blockers, and beta blockers.',
                'category': 'hypertension',
                'focus': 'treatment'
            },
            {
                'question': 'What are the symptoms of heart attack?',
                'answer': 'Heart attack symptoms include chest pain or discomfort (feeling of pressure, squeezing, or fullness), pain or discomfort in the arms, back, neck, jaw, or stomach, shortness of breath, cold sweat, nausea, and lightheadedness. Women may experience different symptoms like unusual fatigue or sleep disturbances.',
                'category': 'heart_disease',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of asthma?',
                'answer': 'Asthma symptoms include shortness of breath, chest tightness or pain, wheezing when exhaling (common sign in children), trouble sleeping caused by shortness of breath, coughing or wheezing attacks that are worsened by respiratory viruses like cold or flu.',
                'category': 'asthma',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of flu?',
                'answer': 'Flu symptoms include fever over 100.4°F (38°C), aching muscles, chills and sweats, headache, dry persistent cough, fatigue and weakness, nasal congestion, and sore throat. Unlike cold symptoms which come on gradually, flu symptoms appear suddenly.',
                'category': 'influenza',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of COVID-19?',
                'answer': 'COVID-19 symptoms include fever or chills, cough, shortness of breath or difficulty breathing, fatigue, muscle or body aches, headache, new loss of taste or smell, sore throat, congestion or runny nose, nausea or vomiting, and diarrhea.',
                'category': 'covid',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of anemia?',
                'answer': 'Anemia symptoms include fatigue, weakness, pale or yellowish skin, irregular heartbeats, shortness of breath, dizziness or lightheadedness, chest pain, cold hands and feet, and headaches. Severe anemia can cause brittle nails, unusual food cravings, and sore tongue.',
                'category': 'anemia',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of migraine?',
                'answer': 'Migraine symptoms include severe throbbing or pulsing pain, usually on one side of the head, sensitivity to light, sound, and sometimes smell, nausea and vomiting, blurred vision, lightheadedness, sometimes followed by fainting. Some people experience aura before the headache.',
                'category': 'migraine',
                'focus': 'symptoms'
            },
            {
                'question': 'What is polio and what are its symptoms?',
                'answer': 'Polio (poliomyelitis) is a highly infectious viral disease that largely affects children under 5 years of age. The virus is transmitted by person-to-person spread mainly through the fecal-oral route or by contaminated water or food. Symptoms include fever, fatigue, headache, vomiting, stiffness in the neck, and pain in the limbs. In severe cases, it can cause permanent paralysis or death. Most people with polio do not show symptoms but can still spread the virus.',
                'category': 'polio',
                'focus': 'symptoms'
            },
            {
                'question': 'How is polio treated?',
                'answer': 'There is no cure for polio, but it can be prevented through vaccination. Treatment focuses on supportive care to manage symptoms and prevent complications. This may include bed rest, pain relievers, physical therapy to prevent muscle wasting and deformity, and mechanical ventilation for severe cases with respiratory paralysis. The polio vaccine is highly effective in preventing the disease.',
                'category': 'polio',
                'focus': 'treatment'
            },
            {
                'question': 'What are the symptoms of tuberculosis (TB)?',
                'answer': 'Tuberculosis symptoms include a bad cough lasting 3 weeks or longer, pain in the chest, coughing up blood or sputum, weakness or fatigue, weight loss, no appetite, chills, fever, and sweating at night. TB mainly affects the lungs but can also affect other parts of the body like the kidneys, spine, or brain.',
                'category': 'tuberculosis',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of malaria?',
                'answer': 'Malaria symptoms include fever, chills, headache, nausea and vomiting, muscle pain, fatigue, and sweating. These symptoms typically appear 10-15 days after being bitten by an infected mosquito. In severe cases, malaria can cause jaundice, seizures, mental confusion, coma, and death.',
                'category': 'malaria',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of dengue fever?',
                'answer': 'Dengue fever symptoms include high fever, severe headache, pain behind the eyes, joint and muscle pain, nausea, vomiting, swollen glands, and rash. Most people recover within a week, but severe dengue can cause bleeding, low blood pressure, and organ failure.',
                'category': 'dengue',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of pneumonia?',
                'answer': 'Pneumonia symptoms include fever, cough with phlegm, shortness of breath, chest pain, fatigue, confusion (especially in older adults), and lower than normal body temperature in older adults. Bacterial pneumonia typically causes more severe symptoms than viral pneumonia.',
                'category': 'pneumonia',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of hepatitis?',
                'answer': 'Hepatitis symptoms include jaundice (yellowing of skin and eyes), abdominal pain, dark urine, pale or clay-colored stools, fatigue, loss of appetite, nausea, vomiting, low-grade fever, and joint pain. Symptoms vary depending on the type of hepatitis (A, B, C, D, or E).',
                'category': 'hepatitis',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of typhoid fever?',
                'answer': 'Typhoid fever symptoms include sustained fever as high as 103-104°F (39-40°C), headache, stomach pain, weakness, loss of appetite, constipation or diarrhea, and a rose-colored rash on the abdomen. If untreated, typhoid can cause serious complications including intestinal perforation.',
                'category': 'typhoid',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of chickenpox?',
                'answer': 'Chickenpox symptoms include an itchy rash that turns into fluid-filled blisters, fever, headache, tiredness, and loss of appetite. The rash typically starts on the face, chest, and back before spreading to the rest of the body. Chickenpox is highly contagious and caused by the varicella-zoster virus.',
                'category': 'chickenpox',
                'focus': 'symptoms'
            },
            {
                'question': 'What are the symptoms of measles?',
                'answer': 'Measles symptoms include high fever, cough, runny nose, red watery eyes, and a characteristic red rash that starts on the face and spreads to the body. Measles can cause serious complications including pneumonia, encephalitis, and death. The MMR vaccine provides effective protection.',
                'category': 'measles',
                'focus': 'symptoms'
            }
        ]
        
        documents = []
        metadata = []
        
        for item in sample_data:
            doc_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append({
                'text': doc_text,
                'id': f"med_{hash(doc_text)}"
            })
            metadata.append({
                'source': 'sample_medical',
                'category': item['category'],
                'focus': item['focus']
            })
        
        # Add to medical collection directly
        texts = [doc['text'] for doc in documents]
        embeddings = self.vector_store.embedding_model.encode(texts).tolist()
        ids = [doc['id'] for doc in documents]
        
        self.medical_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata
        )
        
        logger.info(f"Created {len(documents)} sample medical Q&A pairs")
        return len(documents)
    
    def answer_medical_question(self, question: str, n_results: int = 3) -> Dict:
        """
        Answer a medical question using retrieval
        
        Args:
            question: Medical question
            n_results: Number of relevant answers to retrieve
            
        Returns:
            Dictionary with answer and context
        """
        try:
            # Extract entities from question
            entities = self.entity_recognizer.extract_entities(question)
            
            # Search for relevant Q&A pairs in medical collection
            query_embedding = self.vector_store.embedding_model.encode([question]).tolist()
            results = self.medical_collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            # Format response
            if results['documents'] and results['documents'][0]:
                best_doc = results['documents'][0][0]
                best_metadata = results['metadatas'][0][0] if results['metadatas'] and results['metadatas'][0] else {}
                best_distance = results['distances'][0][0] if results['distances'] and results['distances'][0] else 0.5
                
                answer = {
                    'question': question,
                    'answer': best_doc,
                    'confidence': 1 - best_distance,
                    'source': best_metadata.get('source', 'unknown'),
                    'entities_detected': entities,
                    'related_answers': results['documents'][0][1:] if len(results['documents'][0]) > 1 else [],
                    'status': 'success'
                }
            else:
                answer = {
                    'question': question,
                    'answer': "I don't have specific information about that in my medical database. Please consult a healthcare professional for accurate medical advice.",
                    'confidence': 0.0,
                    'entities_detected': entities,
                    'status': 'no_match'
                }
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering medical question: {e}")
            return {
                'question': question,
                'answer': "An error occurred while processing your question.",
                'status': 'error',
                'error': str(e)
            }
    
    def get_medical_categories(self) -> List[str]:
        """Get available medical categories"""
        try:
            # Get all unique categories from metadata
            results = self.medical_collection.get(include=['metadatas'])
            categories = set()
            for meta in results.get('metadatas', []):
                if 'category' in meta:
                    categories.add(meta['category'])
            return list(categories)
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
