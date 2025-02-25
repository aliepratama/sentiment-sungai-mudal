"""
Setup script to initialize required dependencies for the sentiment analysis app.
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    
if __name__ == "__main__":
    install_requirements()
    download_nltk_data()
    print("Setup completed successfully!")
