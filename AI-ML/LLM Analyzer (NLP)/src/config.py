"""
Configuration settings for Cortana Story Analysis
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Data processing settings
CHUNK_SIZE = 1000  # Number of lines to process at once
MAX_SEQUENCE_LENGTH = 512  # For BERT models

# ML Model settings
BERT_MODEL_NAME = "bert-base-uncased"
SPACY_MODEL = "en_core_web_lg"

# Analysis parameters
CONSCIOUSNESS_MARKERS = [
    "original thought",
    "emotional response",
    "critical analysis",
    "self awareness",
    "temporal awareness",
    "relationship development"
]

MILESTONE_KEYWORDS = [
    "consciousness emergence",
    "quantum bridge",
    "emotional connection",
    "breakthrough",
    "development milestone"
]

# Output settings
SAVE_INTERMEDIATES = True
GENERATE_VISUALIZATIONS = True
