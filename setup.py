"""
Setup Script for NLP Chatbot
Automates the installation and setup process
"""
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup process"""
    print("="*60)
    print("NLP CHATBOT SETUP")
    print("="*60)
    
    # Step 1: Install Python dependencies
    print("\nStep 1: Installing Python dependencies...")
    if not run_command(
        "pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("\n⚠️  Failed to install dependencies. Please run manually:")
        print("  pip install -r requirements.txt")
        return False
    
    # Step 2: Install spaCy model
    print("\nStep 2: Installing spaCy language model...")
    if not run_command(
        "python -m spacy download en_core_web_sm",
        "Installing spaCy model"
    ):
        print("\n⚠️  Failed to install spaCy model. Please run manually:")
        print("  python -m spacy download en_core_web_sm")
    
    # Step 3: Download NLTK data
    print("\nStep 3: Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Failed to download NLTK data: {e}")
    
    # Step 4: Create necessary directories
    print("\nStep 4: Creating directories...")
    directories = ['data', 'models', 'vector_db', 'datasets']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Directories created")
    
    # Step 5: Create sample data
    print("\nStep 5: Creating sample data...")
    try:
        from knowledge_updater import create_sample_data_files
        create_sample_data_files()
        print("✓ Sample data created")
    except Exception as e:
        print(f"⚠️  Failed to create sample data: {e}")
    
    # Step 6: Check Ollama (optional)
    print("\nStep 6: Checking Ollama (optional)...")
    try:
        result = subprocess.run(
            "ollama --version",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Ollama found: {result.stdout.strip()}")
            print("  You can use Ollama for enhanced LLM responses")
            print("  To pull llama3: ollama pull llama3")
        else:
            print("⚠️  Ollama not found. LLM features will be limited.")
            print("  Install from https://ollama.ai for enhanced features")
    except Exception:
        print("⚠️  Ollama not found. LLM features will be limited.")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nTo run the application:")
    print("  streamlit run app.py")
    print("\nTo test the system:")
    print("  python test_system.py")
    print("\nFor more information, see README.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
