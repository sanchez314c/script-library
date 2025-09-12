# 🧠 Natural Language Processing Suite

## 🎯 Overview

Advanced Natural Language Processing suite featuring AI-powered resume generation, conversation analysis, and semantic understanding. This professional-grade toolkit leverages state-of-the-art transformer models, semantic embeddings, and machine learning algorithms to extract insights from conversational data and generate comprehensive career analysis reports.

## ✨ Key Features

- **🤖 Advanced NLP Pipeline**: Transformer-based text processing with BERT, spaCy, and sentence transformers
- **📊 AI-Powered Analytics**: Semantic similarity analysis, topic modeling, and sentiment classification
- **📄 Professional Resume Generation**: Automated PDF resume creation with skill extraction and job matching
- **🌐 Interactive Dashboards**: Real-time web-based analytics with plotly visualizations
- **🎨 Universal macOS Compatibility**: Native system integration with desktop logging and GUI components
- **⚡ High-Performance Processing**: Multi-threaded operations with GPU acceleration support
- **🔧 Auto-Dependency Management**: Intelligent package installation and model downloading
- **📈 Comprehensive Analytics**: Skill evolution tracking, sentiment analysis, and career insights

## 📋 Script Inventory

### 🧠 NLP JSON-to-Resume Processor
**File**: `nlps-json-parse-to-resume.py`
- **Advanced Text Processing**: Multi-stage NLP pipeline with transformer models and semantic analysis
- **Intelligent Skill Extraction**: Named entity recognition with domain-specific skill identification
- **Career Compatibility Analysis**: Semantic job matching using neural embeddings and similarity metrics
- **Professional PDF Generation**: Automated resume creation with tables, charts, and skill visualizations
- **Interactive Web Dashboard**: Real-time analytics dashboard with sentiment trends and skill evolution
- **Topic Modeling**: BERTopic-powered thematic analysis with keyword extraction and clustering

## 🚀 Quick Start Guide

### Prerequisites
- **Operating System**: macOS (Universal compatibility)
- **Python Version**: 3.8 or higher
- **Hardware**: 8GB+ RAM recommended, GPU optional for acceleration
- **Dependencies**: Auto-installed on first run

### System Requirements

#### GPU Acceleration (Optional)
```bash
# Apple Silicon (M1/M2/M3)
# MPS acceleration automatically detected and enabled

# NVIDIA CUDA (if available)  
# CUDA acceleration automatically detected and enabled

# CPU Fallback
# Automatic fallback to optimized CPU processing
```

### Basic Usage

1. **Make script executable**:
   ```bash
   chmod +x nlps-json-parse-to-resume.py
   ```

2. **Launch GUI interface**:
   ```bash
   ./nlps-json-parse-to-resume.py
   ```

3. **Command-line usage**:
   ```bash
   ./nlps-json-parse-to-resume.py --input conversations.json --output ~/Documents/Resume --name "John Doe"
   ```

### GUI Operation Flow

1. **📁 Select Input**: Choose JSON conversation file with ChatGPT or AI interactions
2. **📂 Choose Output**: Select directory for generated resume and analytics
3. **👤 Enter Name**: Provide name for personalized resume generation
4. **🚀 Process**: Automated AI analysis with real-time progress tracking
5. **📊 Review**: Professional PDF resume with interactive dashboard option

### Command-Line Interface

```bash
# Basic resume generation
./nlps-json-parse-to-resume.py --input conversations.json --name "Jane Smith"

# Specify custom output directory
./nlps-json-parse-to-resume.py --input data.json --output ~/Career --name "John Doe"

# Launch with interactive dashboard
./nlps-json-parse-to-resume.py --input conversations.json --name "Alex Johnson" --dashboard

# Command-line mode without GUI
./nlps-json-parse-to-resume.py --input data.json --output ~/Resume --name "User" --no-gui

# Show version and help
./nlps-json-parse-to-resume.py --version
./nlps-json-parse-to-resume.py --help
```

## 📊 Advanced NLP Features

### 🤖 Transformer-Based Processing

#### Model Architecture
- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic embeddings and similarity analysis
- **BERT Classification**: `distilbert-base-uncased-finetuned-sst-2-english` for sentiment analysis
- **SpaCy Transformers**: `en_core_web_trf` for named entity recognition and dependency parsing
- **BERTopic Modeling**: Advanced topic discovery with c-TF-IDF and UMAP dimensionality reduction

#### Processing Pipeline
1. **Text Preprocessing**: Tokenization, lemmatization, and stop word removal
2. **Entity Recognition**: Named entity extraction with confidence scoring
3. **Semantic Analysis**: Vector embeddings for similarity and clustering
4. **Topic Discovery**: Unsupervised topic modeling with keyword extraction
5. **Sentiment Classification**: Emotion detection with temporal analysis

### 📈 Skill Analysis Engine

#### Skill Extraction Methods
- **Named Entity Recognition**: Automatic identification of technical terms and tools
- **Frequency Analysis**: Statistical ranking based on mention frequency and context
- **Semantic Clustering**: Related skill grouping using neural similarity
- **Domain Classification**: Technology stack and expertise area identification
- **Evolution Tracking**: Temporal skill development analysis

#### Job Matching Algorithm
```python
# Semantic similarity calculation
skill_embedding = model.encode(user_skills)
job_embeddings = model.encode(job_descriptions) 
similarities = cosine_similarity(skill_embedding, job_embeddings)
matches = rank_jobs_by_similarity(similarities)
```

### 🎨 Visualization Components

#### Skill Network Graphs
- **Node Representation**: Skills as nodes with size indicating frequency
- **Edge Connections**: Similarity-based relationships between related skills
- **Layout Optimization**: Spring layout algorithm for optimal visual arrangement
- **Interactive Elements**: Hover details and zoom capabilities

#### Dashboard Analytics
- **Real-time Charts**: Dynamic plotly visualizations with responsive design
- **Multi-tab Interface**: Organized data presentation across skill, job, and sentiment tabs
- **Temporal Analysis**: Time-series visualization of skill evolution and sentiment trends
- **Export Capabilities**: PNG, SVG, and PDF export options for all visualizations

## 🔧 Configuration Options

### Processing Parameters

#### Text Processing Settings
```bash
# Batch size for efficient processing
--batch-size 50              # Number of texts processed simultaneously

# Model selection
--sentence-model all-MiniLM-L6-v2    # Sentence transformer model
--spacy-model en_core_web_trf        # SpaCy transformer model

# Topic modeling parameters  
--min-topic-size 5           # Minimum documents per topic
--n-gram-range 1 2           # N-gram range for topic keywords
```

#### Performance Optimization
```bash
# Hardware acceleration
--device auto               # Automatic device selection (MPS/CUDA/CPU)
--threads 4                 # Number of processing threads
--gpu-memory-limit 4GB      # GPU memory usage limit

# Processing limits
--max-skills 100            # Maximum skills to extract
--max-topics 20             # Maximum topics to identify
--similarity-threshold 0.6   # Minimum similarity for skill relationships
```

### Output Customization

#### Resume Formatting
- **Layout Options**: Professional template with customizable sections
- **Content Selection**: Configurable inclusion of skills, topics, and job matches
- **Visual Elements**: Optional skill network diagrams and progress charts
- **Branding**: Customizable headers, footers, and color schemes

#### Dashboard Configuration
- **Chart Types**: Bar charts, scatter plots, line graphs, and network diagrams
- **Color Schemes**: Professional color palettes with accessibility support
- **Interactive Features**: Hover tooltips, zoom controls, and data filtering
- **Export Formats**: High-resolution image export and data download options

## 📊 Analytics and Insights

### Career Intelligence Metrics

#### Skill Assessment
- **Proficiency Levels**: Automatic classification (Basic, Intermediate, Advanced, Expert)
- **Market Relevance**: Industry demand analysis and trending technology identification
- **Skill Gaps**: Comparison with target job requirements and improvement recommendations
- **Learning Path**: Suggested skill development roadmap based on career goals

#### Job Market Analysis
- **Compatibility Scoring**: Semantic matching with job descriptions and requirements
- **Salary Predictions**: Estimated compensation based on skill set and experience
- **Industry Trends**: Market demand analysis and growth projections
- **Geographic Insights**: Location-based job market analysis and remote opportunities

### Quality Metrics

#### Processing Statistics
- **Extraction Accuracy**: Percentage of successfully identified skills and entities
- **Topic Coherence**: Quality measurement of discovered topics and themes
- **Sentiment Confidence**: Reliability scores for emotion classification
- **Coverage Analysis**: Completeness of data processing and feature extraction

#### Performance Benchmarks
- **Processing Speed**: Files per second and token processing rates
- **Memory Efficiency**: RAM usage optimization and garbage collection
- **Model Accuracy**: Validation scores for various NLP tasks
- **Scalability**: Performance metrics for different dataset sizes

## 🛠️ Troubleshooting

### Common Processing Issues

**Model Loading Failures**
```bash
# Check available memory
python3 -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# Clear model cache
rm -rf ~/.cache/torch/sentence_transformers/
rm -rf ~/.cache/huggingface/transformers/

# Reinstall models
python3 -m spacy download en_core_web_trf
```

**GPU Acceleration Issues**
```bash
# Check MPS availability (Apple Silicon)
python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"

# Check CUDA availability (NVIDIA)
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Force CPU processing
./nlps-json-parse-to-resume.py --device cpu
```

**Memory Management**
```bash
# Reduce batch size for large datasets
./nlps-json-parse-to-resume.py --batch-size 25

# Enable memory optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Monitor memory usage during processing
top -pid $(pgrep -f nlps-json-parse-to-resume.py)
```

### Performance Optimization

**Large Dataset Processing**
- Use smaller batch sizes to prevent memory overflow
- Enable CPU processing for very large datasets
- Process data in chunks for datasets over 10,000 conversations
- Monitor system resources during processing

**Quality Improvement**
- Ensure conversation data includes diverse topics and skills
- Provide clean, well-formatted JSON input data
- Use longer conversation histories for better skill extraction
- Include technical discussions for improved skill identification

## 📁 Output Structure

### Generated Files
```
[output_directory]/
├── [Name]_AI_Resume.pdf                 # Professional PDF resume
├── skill_network.png                    # Skill relationship visualization
├── processing_summary.json              # Comprehensive analytics report
├── skill_extraction_details.csv         # Detailed skill analysis
├── topic_analysis_report.json           # Topic modeling results
├── sentiment_timeline.csv               # Sentiment analysis over time
└── job_compatibility_scores.json        # Career matching analysis
```

### Log Files
```
Desktop/NLP_Resume_Processor_Logs_[timestamp]/
├── nlp_resume_processing_[timestamp].log # Detailed processing log
├── model_performance_metrics.json        # AI model performance data
├── error_analysis_report.json            # Error categorization and analysis
└── system_resource_usage.csv             # Hardware utilization tracking
```

## 📚 Technical Specifications

### Supported Data Formats

#### Input Requirements
- **JSON Structure**: Array of conversation objects with timestamp, user_input, and ai_response fields
- **Encoding**: UTF-8 with support for international characters and emojis
- **Size Limits**: Up to 100MB per file with automatic chunking for larger datasets
- **Validation**: JSON Schema validation with detailed error reporting

#### JSON Schema
```json
{
  "type": "array",
  "items": {
    "type": "object", 
    "properties": {
      "timestamp": {"type": "string", "format": "date-time"},
      "user_input": {"type": "string", "minLength": 1},
      "ai_response": {"type": "string", "minLength": 1}
    },
    "required": ["timestamp", "user_input", "ai_response"]
  }
}
```

### System Requirements
- **OS**: macOS 10.15 or later (Universal compatibility)
- **RAM**: 8GB minimum, 16GB recommended for large datasets
- **Storage**: 5GB free space for models and processing cache
- **CPU**: Multi-core processor recommended for parallel processing
- **GPU**: Apple Silicon MPS or NVIDIA CUDA for acceleration (optional)

### Dependencies
```python
# Core NLP Dependencies (auto-installed)
torch>=1.12.0                  # Deep learning framework with MPS support
spacy>=3.4.0                   # Advanced NLP with transformer models
transformers>=4.21.0           # Hugging Face transformer models
sentence-transformers>=2.2.0   # Semantic similarity and embeddings
bertopic>=0.12.0               # Topic modeling with BERT embeddings

# Data Processing and Analysis
pandas>=1.5.0                  # Data manipulation and analysis
numpy>=1.21.0                  # Numerical computing and array operations
scikit-learn>=1.1.0            # Machine learning algorithms and metrics
networkx>=2.8.0                # Graph analysis for skill relationships
matplotlib>=3.5.0              # Static plotting and visualization

# Interactive Visualization
plotly>=5.10.0                 # Interactive charts and dashboards
dash>=2.6.0                    # Web-based dashboard framework
dash-core-components>=2.0.0    # Dashboard UI components
dash-html-components>=2.0.0    # HTML components for dashboards

# Document Generation
reportlab>=3.6.0               # PDF generation and formatting
Pillow>=9.0.0                  # Image processing and manipulation

# Text Processing and Utilities
jsonschema>=4.0.0              # JSON validation and schema checking
fuzzywuzzy>=0.18.0             # Fuzzy string matching for similarity
tqdm>=4.64.0                   # Progress bars and status tracking

# Built-in Dependencies
tkinter                        # GUI framework (included with Python)
json                          # JSON parsing and generation
logging                       # Comprehensive logging system
multiprocessing              # Parallel processing and threading
pathlib                      # Modern path handling
datetime                     # Date and time manipulation
```

## 🎯 GET SWIFTY Methodology

This NLP suite implements the comprehensive GET SWIFTY development approach:

- **🎨 GET**: Professional GUI with real-time progress tracking and native macOS integration
- **⚡ SWIFTY**: High-performance processing with transformer models and GPU acceleration
- **🧠 Intelligent**: Advanced AI-powered analysis with semantic understanding
- **🔧 Reliable**: Comprehensive error handling with graceful degradation
- **📊 Analytics**: Detailed metrics and performance monitoring
- **🍎 macOS Native**: Universal compatibility with system optimization

## 📞 Support and Resources

### Model Information
- **Sentence Transformers**: Semantic similarity and embedding generation
- **SpaCy Transformers**: Named entity recognition and dependency parsing
- **BERT Models**: Classification and sentiment analysis
- **BERTopic**: Advanced topic modeling with clustering

### Performance Guidelines
- **Memory**: Use appropriate batch sizes for available RAM
- **Processing**: Leverage GPU acceleration when available
- **Storage**: Maintain sufficient disk space for model cache
- **Network**: Download models may require stable internet connection

### Advanced Usage
- **Custom Models**: Integration with domain-specific transformer models
- **Batch Processing**: Multiple file processing with automation
- **API Integration**: RESTful API for programmatic access
- **Cloud Deployment**: Scalable deployment options for large datasets

---

**GET SWIFTY NLP Suite v1.0.0**  
*Advanced Natural Language Processing for macOS with professional resume generation*