"""
Learn to Build a Customer Service Chatbot
Simple, Clean, and Interactive GUI
"""
import streamlit as st
from PIL import Image
from pathlib import Path
import logging
import pdfplumber

from chatbot_core import NLPChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Medical & Scientific AI",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean design
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2563eb;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        max-width: 85%;
    }
    .user-message {
        background-color: #3b82f6;
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f1f5f9;
        color: #1e293b;
        margin-right: auto;
    }
    .sentiment-badge {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
    }
    .sentiment-positive {
        background-color: #dcfce7;
        color: #166534;
    }
    .sentiment-negative {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .sentiment-neutral {
        background-color: #e2e8f0;
        color: #475569;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_chatbot():
    """Initialize chatbot session state"""
    if 'chatbot' not in st.session_state:
        with st.spinner("🤖 Initializing chatbot..."):
            st.session_state.chatbot = NLPChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "customer_service"


def display_chat_message(role, content, sentiment=None):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        sentiment_class = f"sentiment-{sentiment.get('dominant_sentiment', 'neutral')}" if sentiment else "sentiment-neutral"
        st.markdown(f"""
        <div class="chat-message assistant-message">
            {content}
            <div class="sentiment-badge {sentiment_class}">
                😊 {sentiment.get('dominant_sentiment', 'neutral').capitalize()}
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🏥 Medical & Scientific AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Medical Q&A and Scientific Research</div>', unsafe_allow_html=True)
    
    # Initialize chatbot
    initialize_chatbot()
    
    # Mode selection with tabs
    tab1, tab2, tab3 = st.tabs(["🏥 Medical Q&A", "🔬 Scientific Research", "� Document Analyzer"])
    
    with tab1:
        st.subheader("Medical Q&A")
        st.info("Ask medical questions powered by MedQuAD dataset")
        
        medical_query = st.text_input("Ask a medical question:", placeholder="e.g., What are the symptoms of diabetes?", key="medical_input")
        
        if st.button("Ask Medical Question", key="medical_button"):
            if medical_query:
                with st.spinner("Searching medical database..."):
                    try:
                        response = st.session_state.chatbot.process_message(medical_query, "medical")
                        st.success("Answer:")
                        st.write(response.get('answer', ''))
                        
                        if response.get('entities_detected'):
                            with st.expander("🔍 Detected Medical Entities"):
                                entities = response['entities_detected']
                                for category, items in entities.items():
                                    if items:
                                        st.write(f"**{category.capitalize()}:** {', '.join(items)}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.subheader("Scientific Research")
        st.info("Search scientific papers and explain concepts")
        
        scientific_query = st.text_input("Ask about science:", placeholder="e.g., Explain transformer architecture", key="science_input")
        
        if st.button("Search Papers", key="science_button"):
            if scientific_query:
                with st.spinner("Searching scientific database..."):
                    try:
                        response = st.session_state.chatbot.process_message(scientific_query, "scientific")
                        st.success("Response:")
                        st.write(response.get('answer', ''))
                        
                        if response.get('papers'):
                            with st.expander("📄 Related Papers"):
                                for paper in response['papers'][:3]:
                                    st.write(f"• {paper['metadata']['title']}")
                                    st.caption(f"Confidence: {paper['confidence']:.2f}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a query")
    
    with tab3:
        st.subheader("Document & Report Analyzer")
        st.info("Upload documents or reports for AI analysis and summarization")
        
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"], key="doc_upload")
        
        if uploaded_file:
            # Read document content
            try:
                content = ""
                file_type = uploaded_file.type
                
                if file_type == "text/plain":
                    content = uploaded_file.read().decode("utf-8")
                    st.success("TXT document loaded successfully!")
                
                elif file_type == "application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            content += page.extract_text() + "\n"
                    st.success("PDF document loaded successfully!")
                
                # Display content preview
                with st.expander("Preview Document Content"):
                    st.text_area("Content:", content, height=200, key="doc_content")
                
                # Analysis options
                analysis_type = st.selectbox(
                    "Select analysis type:",
                    ["Summarize", "Extract Key Points", "Sentiment Analysis", "Medical Entity Extraction", "Topic Extraction"],
                    key="analysis_type"
                )
                
                if st.button("Analyze Document", key="analyze_doc"):
                    with st.spinner("Analyzing document..."):
                        try:
                            if analysis_type == "Summarize":
                                # Simple summarization
                                sentences = [s.strip() for s in content.split('. ') if s.strip()]
                                if len(sentences) > 10:
                                    summary = '. '.join(sentences[:10]) + '...'
                                else:
                                    summary = '. '.join(sentences)
                                st.success("Summary:")
                                st.write(summary)
                            
                            elif analysis_type == "Extract Key Points":
                                # Extract sentences with keywords
                                keywords = ["important", "key", "main", "significant", "critical", "essential", "crucial", "vital", "major", "primary"]
                                key_points = [s.strip() for s in content.split('. ') if any(k in s.lower() for k in keywords) and s.strip()]
                                if not key_points:
                                    # Fallback: extract longest sentences
                                    sentences = [s.strip() for s in content.split('. ') if s.strip()]
                                    key_points = sorted(sentences, key=len, reverse=True)[:5]
                                st.success("Key Points:")
                                for point in key_points[:5]:
                                    st.write(f"• {point}")
                            
                            elif analysis_type == "Sentiment Analysis":
                                sentiment = st.session_state.chatbot.sentiment_analyzer.analyze_sentiment(content)
                                st.success("Sentiment Analysis:")
                                st.write(f"**Dominant Sentiment:** {sentiment['dominant_sentiment'].capitalize()}")
                                st.write(f"**Confidence:** {sentiment['confidence']:.2f}")
                                st.write("**Scores:**")
                                for label, score in sentiment['scores'].items():
                                    st.write(f"- {label}: {score:.2f}")
                            
                            elif analysis_type == "Medical Entity Extraction":
                                entities = st.session_state.chatbot.medical_qa.entity_recognizer.extract_entities(content)
                                st.success("Medical Entities:")
                                found_entities = False
                                for category, items in entities.items():
                                    if items:
                                        st.write(f"**{category.capitalize()}:** {', '.join(items)}")
                                        found_entities = True
                                if not found_entities:
                                    st.info("No medical entities detected in the document.")
                            
                            elif analysis_type == "Topic Extraction":
                                # Simple topic extraction using frequent words
                                words = content.lower().split()
                                # Filter common words
                                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'you', 'your', 'my', 'our', 'as', 'if', 'then', 'when', 'where', 'while', 'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'once', 'here', 'there', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'}
                                filtered_words = [w for w in words if len(w) > 3 and w not in common_words]
                                word_freq = {}
                                for word in filtered_words:
                                    word_freq[word] = word_freq.get(word, 0) + 1
                                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                                st.success("Top Topics/Keywords:")
                                for word, freq in top_words:
                                    st.write(f"• {word.capitalize()} (mentioned {freq} times)")
                        
                        except Exception as e:
                            st.error(f"Error analyzing document: {str(e)}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Alternative: Paste text directly
        st.divider()
        st.subheader("Or Paste Text Directly")
        pasted_text = st.text_area("Paste your text here:", height=150, key="pasted_text")
        
        if pasted_text:
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Summarize", "Extract Key Points", "Sentiment Analysis", "Medical Entity Extraction"],
                key="analysis_type_paste"
            )
            
            if st.button("Analyze Pasted Text", key="analyze_paste"):
                with st.spinner("Analyzing text..."):
                    try:
                        if analysis_type == "Summarize":
                            sentences = pasted_text.split('. ')
                            summary = '. '.join(sentences[:5]) + '...'
                            st.success("Summary:")
                            st.write(summary)
                        
                        elif analysis_type == "Extract Key Points":
                            keywords = ["important", "key", "main", "significant", "critical", "essential"]
                            key_points = [s for s in pasted_text.split('. ') if any(k in s.lower() for k in keywords)]
                            st.success("Key Points:")
                            for point in key_points[:5]:
                                st.write(f"• {point}")
                        
                        elif analysis_type == "Sentiment Analysis":
                            sentiment = st.session_state.chatbot.sentiment_analyzer.analyze_sentiment(pasted_text)
                            st.success("Sentiment Analysis:")
                            st.write(f"**Dominant Sentiment:** {sentiment['dominant_sentiment'].capitalize()}")
                            st.write(f"**Confidence:** {sentiment['confidence']:.2f}")
                            st.write("**Scores:**")
                            for label, score in sentiment['scores'].items():
                                st.write(f"- {label}: {score:.2f}")
                        
                        elif analysis_type == "Medical Entity Extraction":
                            entities = st.session_state.chatbot.medical_qa.entity_recognizer.extract_entities(pasted_text)
                            st.success("Medical Entities:")
                            for category, items in entities.items():
                                if items:
                                    st.write(f"**{category.capitalize()}:** {', '.join(items)}")
                    
                    except Exception as e:
                        st.error(f"Error analyzing text: {str(e)}")
    
    # Footer with controls
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("�️ Clear Chat"):
            st.session_state.chatbot.clear_conversation()
            st.session_state.messages = []
            st.success("Chat cleared!")
    
    with col2:
        if st.button("📊 System Stats"):
            stats = st.session_state.chatbot.get_system_stats()
            st.json(stats)
    
    with col3:
        if st.button("➕ Add Knowledge"):
            with st.expander("Add Custom Knowledge"):
                custom_text = st.text_area("Enter information:", height=100)
                if st.button("Add to Database"):
                    if custom_text:
                        count = st.session_state.chatbot.update_knowledge_base(
                            [{'text': custom_text, 'id': f"custom_{len(custom_text)}"}],
                            [{'source': 'user_input'}]
                        )
                        st.success(f"Added {count} document(s)!")


if __name__ == "__main__":
    main()
