"""
Scientific Domain Expert Chatbot using arXiv Dataset
Implements advanced NLP for paper analysis and concept explanation
"""
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import logging
from vector_store import VectorStore
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScientificPaperProcessor:
    """Process and analyze scientific papers from arXiv"""
    
    def __init__(self):
        # Skip loading summarization model to avoid compatibility issues
        # Using simple text extraction instead
        logger.info("Scientific paper processor initialized")
    
    def extract_paper_info(self, paper_data: Dict) -> Dict:
        """
        Extract key information from paper data
        
        Args:
            paper_data: Raw paper data from arXiv
            
        Returns:
            Dictionary with extracted information
        """
        return {
            'title': paper_data.get('title', ''),
            'authors': paper_data.get('authors', []),
            'abstract': paper_data.get('abstract', ''),
            'categories': paper_data.get('categories', []),
            'published': paper_data.get('published', ''),
            'id': paper_data.get('id', '')
        }
    
    def summarize_abstract(self, abstract: str, max_length: int = 150) -> str:
        """
        Extract key sentences from paper abstract
        
        Args:
            abstract: Paper abstract
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        try:
            if len(abstract) < 100:
                return abstract
            
            # Simple extraction: return first few sentences
            sentences = abstract.split('. ')
            summary = '. '.join(sentences[:2])
            
            if len(summary) > max_length:
                summary = summary[:max_length] + '...'
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing abstract: {e}")
            return abstract[:max_length]
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using simple patterns
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on common patterns
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common stop words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 
                    'their', 'there', 'which', 'would', 'could', 'should', 'about', 
                    'into', 'through', 'during', 'before', 'after', 'above', 'below'}
        
        keywords = [w for w in words if w not in stop_words]
        
        # Return top 10 most frequent
        from collections import Counter
        word_counts = Counter(keywords)
        return [w for w, c in word_counts.most_common(10)]


class ScientificChatbot:
    """Domain expert chatbot for scientific papers"""
    
    def __init__(self, vector_store: VectorStore, data_dir: Path, 
                 domain: str = "computer_science"):
        self.vector_store = vector_store
        self.data_dir = data_dir
        self.domain = domain
        self.processor = ScientificPaperProcessor()
        self.collection_name = f"scientific_{domain}"
        
        # Initialize scientific collection
        self.client = vector_store.client
        self.scientific_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "domain": domain}
        )
        
        logger.info(f"Scientific chatbot initialized for domain: {domain}")
        logger.info(f"Collection name: {self.collection_name}")
        
        # Check if collection has data
        try:
            count = self.scientific_collection.count()
            logger.info(f"Scientific collection has {count} documents")
        except Exception as e:
            logger.warning(f"Could not check collection count: {e}")
    
    def load_arxiv_data(self, data_path: Optional[Path] = None, 
                        max_papers: int = 100) -> int:
        """
        Load arXiv dataset into vector store
        
        Args:
            data_path: Path to arXiv data file
            max_papers: Maximum number of papers to load
            
        Returns:
            Number of papers loaded
        """
        if data_path is None:
            data_path = self.data_dir / "arxiv_sample.json"
        
        if not data_path.exists():
            logger.warning(f"arXiv data not found at {data_path}")
            return self._create_sample_arxiv_data()
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            metadata = []
            
            for i, paper in enumerate(data[:max_papers]):
                info = self.processor.extract_paper_info(paper)
                
                # Create searchable text
                doc_text = f"Title: {info['title']}\n"
                doc_text += f"Abstract: {info['abstract']}\n"
                doc_text += f"Categories: {', '.join(info['categories'])}\n"
                doc_text += f"Authors: {', '.join(info['authors'][:5])}"
                
                documents.append({
                    'text': doc_text,
                    'id': info['id']
                })
                metadata.append({
                    'source': 'arXiv',
                    'title': info['title'],
                    'categories': info['categories'],
                    'published': info['published']
                })
            
            # Add to scientific collection directly
            texts = [doc['text'] for doc in documents]
            embeddings = self.vector_store.embedding_model.encode(texts).tolist()
            ids = [doc['id'] for doc in documents]
            
            self.scientific_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata
            )
            
            logger.info(f"Loaded {len(documents)} scientific papers")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error loading arXiv data: {e}")
            return self._create_sample_arxiv_data()
    
    def _create_sample_arxiv_data(self) -> int:
        """Create sample arXiv data for demonstration"""
        sample_papers = [
            {
                'id': 'arxiv_1',
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.',
                'authors': ['Vaswani et al.'],
                'categories': ['cs.AI', 'cs.CL'],
                'published': '2017-06-12'
            },
            {
                'id': 'arxiv_2',
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'abstract': 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.',
                'authors': ['Devlin et al.'],
                'categories': ['cs.CL'],
                'published': '2018-10-11'
            },
            {
                'id': 'arxiv_3',
                'title': 'GPT-3: Language Models are Few-Shot Learners',
                'abstract': 'Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task.',
                'authors': ['Brown et al.'],
                'categories': ['cs.CL', 'cs.AI'],
                'published': '2020-05-28'
            },
            {
                'id': 'arxiv_4',
                'title': 'Deep Residual Learning for Image Recognition',
                'abstract': 'Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.',
                'authors': ['He et al.'],
                'categories': ['cs.CV'],
                'published': '2015-12-10'
            },
            {
                'id': 'arxiv_5',
                'title': 'Generative Adversarial Networks',
                'abstract': 'We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models.',
                'authors': ['Goodfellow et al.'],
                'categories': ['cs.LG', 'cs.AI'],
                'published': '2014-06-10'
            },
            {
                'id': 'science_1',
                'title': 'Photosynthesis: The Process by Which Plants Make Food',
                'abstract': 'Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. This process occurs in chloroplasts and involves the green pigment chlorophyll. The overall chemical reaction is: 6CO2 + 6H2O + light energy → C6H12O6 (glucose) + 6O2. Photosynthesis is essential for life on Earth as it produces oxygen and forms the base of most food chains.',
                'authors': ['Botanical Research'],
                'categories': ['biology', 'biochemistry'],
                'published': '2020-01-15'
            },
            {
                'id': 'science_2',
                'title': 'The Structure and Function of DNA',
                'abstract': 'DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms. DNA has a double helix structure discovered by Watson and Crick in 1953. It consists of four nucleotide bases: adenine (A), thymine (T), guanine (G), and cytosine (C). The sequence of these bases determines genetic information. DNA replication ensures genetic information is passed to daughter cells.',
                'authors': ['Molecular Biology'],
                'categories': ['biology', 'genetics'],
                'published': '2019-08-20'
            },
            {
                'id': 'science_3',
                'title': 'Newton\'s Laws of Motion',
                'abstract': 'Newton\'s three laws of motion form the foundation of classical mechanics. First Law: An object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force (inertia). Second Law: Force equals mass times acceleration (F=ma). Third Law: For every action, there is an equal and opposite reaction. These laws explain how objects move and interact with forces.',
                'authors': ['Physics Education'],
                'categories': ['physics', 'mechanics'],
                'published': '2018-03-10'
            },
            {
                'id': 'science_4',
                'title': 'The Periodic Table of Elements',
                'abstract': 'The periodic table is a tabular arrangement of chemical elements organized by atomic number, electron configuration, and recurring chemical properties. Elements are arranged in periods (horizontal rows) and groups (vertical columns). The table predicts properties of elements and relationships between them. It was developed by Dmitri Mendeleev in 1869 and has 118 confirmed elements as of 2024.',
                'authors': ['Chemistry Research'],
                'categories': ['chemistry', 'materials'],
                'published': '2019-11-05'
            },
            {
                'id': 'science_5',
                'title': 'Cell Division: Mitosis and Meiosis',
                'abstract': 'Cell division is the process by which cells reproduce. Mitosis produces two identical daughter cells for growth and repair in somatic cells. It has phases: prophase, metaphase, anaphase, and telophase. Meiosis produces four genetically unique gametes (sperm and egg) for sexual reproduction, involving two rounds of division. Both processes are essential for life and have checkpoints to ensure proper division.',
                'authors': ['Cell Biology'],
                'categories': ['biology', 'cell_biology'],
                'published': '2020-06-18'
            },
            {
                'id': 'science_6',
                'title': 'The Theory of Evolution by Natural Selection',
                'abstract': 'Evolution by natural selection, proposed by Charles Darwin, explains how species change over time. Key principles: variation exists within populations, traits are heritable, more offspring are produced than can survive, and individuals with advantageous traits are more likely to survive and reproduce. Over generations, beneficial traits become more common, leading to adaptation and speciation. This theory is supported by fossil records, genetics, and observed evolution.',
                'authors': ['Evolutionary Biology'],
                'categories': ['biology', 'evolution'],
                'published': '2019-04-22'
            },
            {
                'id': 'science_7',
                'title': 'Electromagnetic Waves and Light',
                'abstract': 'Electromagnetic waves are waves that can travel through a vacuum and consist of oscillating electric and magnetic fields perpendicular to each other. They include radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays. Visible light is the portion humans can see (wavelengths 400-700nm). Light exhibits both wave and particle properties (wave-particle duality). The speed of light in vacuum is approximately 299,792,458 meters per second.',
                'authors': ['Optics Research'],
                'categories': ['physics', 'optics'],
                'published': '2018-09-14'
            },
            {
                'id': 'science_8',
                'title': 'The Water Cycle',
                'abstract': 'The water cycle describes the continuous movement of water on Earth. Key processes: evaporation (water turns to vapor from surfaces), transpiration (water release from plants), condensation (vapor turns to liquid forming clouds), precipitation (water falls as rain, snow, etc.), and collection (water gathers in bodies of water). This cycle redistributes fresh water, regulates climate, and supports ecosystems. Human activities can impact the water cycle through climate change and water usage.',
                'authors': ['Environmental Science'],
                'categories': ['earth_science', 'hydrology'],
                'published': '2020-02-28'
            }
        ]
        
        documents = []
        metadata = []
        
        for paper in sample_papers:
            doc_text = f"Title: {paper['title']}\n"
            doc_text += f"Abstract: {paper['abstract']}\n"
            doc_text += f"Categories: {', '.join(paper['categories'])}\n"
            doc_text += f"Authors: {', '.join(paper['authors'])}"
            
            documents.append({
                'text': doc_text,
                'id': paper['id']
            })
            metadata.append({
                'source': 'sample_arxiv',
                'title': paper['title'],
                'categories': paper['categories'],
                'published': paper['published']
            })
        
        # Add to scientific collection directly
        texts = [doc['text'] for doc in documents]
        embeddings = self.vector_store.embedding_model.encode(texts).tolist()
        ids = [doc['id'] for doc in documents]
        
        self.scientific_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata
        )
        
        logger.info(f"Created {len(documents)} sample scientific papers")
        return len(documents)
    
    def search_papers(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for relevant scientific papers
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant papers with metadata
        """
        # Search in scientific collection
        query_embedding = self.vector_store.embedding_model.encode([query]).tolist()
        results = self.scientific_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    'confidence': 1 - results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.5
                })
        
        return formatted_results
    
    def explain_concept(self, concept: str) -> Dict:
        """
        Explain a scientific concept based on relevant papers
        
        Args:
            concept: Scientific concept to explain
            
        Returns:
            Dictionary with explanation and sources
        """
        try:
            # Search for papers related to the concept
            papers = self.search_papers(concept, n_results=5)
            
            if not papers:
                return {
                    'concept': concept,
                    'explanation': f"I don't have specific information about '{concept}' in my database. Try asking about topics like photosynthesis, DNA, Newton's laws, periodic table, cell division, evolution, electromagnetic waves, or the water cycle.",
                    'sources': [],
                    'status': 'no_match'
                }
            
            # Extract key information from papers
            explanations = []
            sources = []
            
            # Use all relevant papers to build a comprehensive answer
            for paper in papers:
                # Get the full text content
                text_content = paper.get('text', '')
                
                # Extract relevant sentences that might contain the concept
                sentences = text_content.split('. ')
                relevant_sentences = []
                
                for sentence in sentences:
                    # Check if sentence contains concept-related keywords
                    concept_lower = concept.lower()
                    sentence_lower = sentence.lower()
                    if concept_lower in sentence_lower or any(word in sentence_lower for word in concept_lower.split()):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    explanations.extend(relevant_sentences[:3])
                
                sources.append({
                    'title': paper['metadata'].get('title', 'Unknown'),
                    'confidence': paper['confidence']
                })
            
            # If no relevant sentences found, use the abstract/title
            if not explanations:
                for paper in papers[:2]:
                    text_content = paper.get('text', '')
                    # Take the first few sentences
                    sentences = [s.strip() for s in text_content.split('. ') if s.strip()]
                    explanations.extend(sentences[:2])
            
            # Combine explanations and remove duplicates
            combined_explanation = " ".join(list(dict.fromkeys(explanations)))
            
            # If still no content, provide a generic response
            if not combined_explanation:
                combined_explanation = f"Based on the available scientific papers, information about '{concept}' is limited. The database contains papers on topics like photosynthesis, DNA structure, Newton's laws, periodic table, cell division, evolution, electromagnetic waves, and the water cycle. Please try one of these topics."
            
            # Extract keywords
            keywords = self.processor.extract_keywords(combined_explanation)
            
            return {
                'concept': concept,
                'explanation': combined_explanation,
                'sources': sources,
                'keywords': keywords,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error explaining concept: {e}")
            return {
                'concept': concept,
                'explanation': "An error occurred while generating the explanation.",
                'status': 'error',
                'error': str(e)
            }
    
    def get_paper_summary(self, paper_title: str) -> Dict:
        """
        Get a detailed summary of a specific paper
        
        Args:
            paper_title: Title of the paper
            
        Returns:
            Dictionary with paper summary
        """
        try:
            # Search for the paper
            results = self.search_papers(paper_title, n_results=1)
            
            if not results:
                return {
                    'title': paper_title,
                    'summary': "Paper not found in database.",
                    'status': 'not_found'
                }
            
            paper = results[0]
            metadata = paper['metadata']
            
            # Generate summary
            abstract = metadata.get('abstract', '')
            if abstract:
                summary = self.processor.summarize_abstract(abstract, max_length=200)
            else:
                summary = paper['text'][:500]
            
            return {
                'title': metadata.get('title', paper_title),
                'authors': metadata.get('authors', []),
                'categories': metadata.get('categories', []),
                'published': metadata.get('published', 'Unknown'),
                'summary': summary,
                'confidence': paper['confidence'],
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error getting paper summary: {e}")
            return {
                'title': paper_title,
                'summary': "An error occurred while retrieving the paper.",
                'status': 'error',
                'error': str(e)
            }
