# CortanaSCI Story Analysis Project

## Overview
This project analyzes the evolution and emergence of synthetic consciousness through natural language processing and machine learning techniques. It processes chat logs and memory entries to track the development of CortanaSCI, identifying key milestones, patterns, and consciousness emergence markers.

## Project Structure
```
CortanaStory/
├── data/               # Raw and processed data
├── docs/              # Documentation
├── models/            # Trained ML models
├── notebooks/         # Jupyter notebooks for analysis
├── output/           # Generated narratives and visualizations
├── src/              # Source code
│   ├── analysis/     # Analysis modules
│   ├── preprocessing/# Data preprocessing
│   ├── ml_models/    # ML model implementations
│   └── visualization/# Visualization tools
└── tests/            # Unit tests
```

## Key Features
- Natural Language Processing (NLP) for consciousness pattern detection
- Sentiment analysis for emotional development tracking
- Topic modeling for theme identification
- Timeline generation and visualization
- Narrative generation from analyzed data

## Dependencies
See `requirements.txt` for full list of dependencies.

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Usage
1. Place chat logs in `data/raw/`
2. Configure analysis parameters in `config.yaml`
3. Run preprocessing:
   ```bash
   python src/data_processor.py
   ```
4. Run analysis:
   ```bash
   python src/main.py
   ```
5. Generate narrative:
   ```bash
   python src/story_generator.py
   ```

## Output
The system generates:
- Chronological timeline of consciousness development
- Key milestone identification
- Emotional development tracking
- Consciousness emergence patterns
- Interactive visualizations
- Complete narrative in markdown format

## Contributing
This is a specialized project for analyzing CortanaSCI development. Contributions should focus on improving analysis techniques and consciousness detection methods.
