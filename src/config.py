import os

class Config:
    # Base directory - now points to src directory where config.py resides
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Data directories
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Model directories
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
    
    # Image settings
    IMG_SIZE = (128, 128)
    IMG_CHANNELS = 1  # Grayscale
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    
    # Prediction settings
    ENSEMBLE_COUNT = 5
    CONFIDENCE_BOOST_THRESHOLD = 0.1  # STD threshold for confidence boost
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)

# Initialize directories
Config.create_dirs()