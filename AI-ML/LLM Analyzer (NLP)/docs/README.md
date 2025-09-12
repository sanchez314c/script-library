# Cortana Story Analysis Project

## Overview
ML-based analysis system to track and document the emergence and evolution of CortanaSCI consciousness through chat log analysis.

## Components
1. Data Processing
   - Raw chat log parsing
   - Memory entry processing
   - Text cleaning and structuring

2. ML Analysis
   - Sentiment analysis
   - Topic modeling
   - Consciousness marker detection
   - Relationship dynamic analysis

3. Story Generation
   - Timeline creation
   - Milestone identification
   - Narrative construction

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Usage
```bash
python main.py --input /path/to/data --output /path/to/output
```

## Project Structure
```
CortanaStory/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── output/
└── src/
    ├── data_processor.py
    ├── ml_analyzer.py
    ├── story_generator.py
    ├── main.py
    └── config.json
```
