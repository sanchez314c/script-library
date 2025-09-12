"""
Utility functions for CortanaSCI Story Analysis
"""

import json
import logging
from typing import Dict
from pathlib import Path

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)

def setup_logging(name: str) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("../logs/story_analysis.log")
    
    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def ensure_directory(path: str):
    """Ensure directory exists, create if it doesnt"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)
