# Data Scraping & Analysis Tools Collection

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/sanchez314c/Script.Library)
[![Python](https://img.shields.io/badge/python-3.6+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A powerful collection of web scraping and data extraction tools designed for research, academic content collection, and large-scale data analysis. Each tool is optimized for specific data collection tasks with intelligent filtering, multi-core processing, and comprehensive quality assessment.

## 🎯 Overview

This collection includes four specialized data scraping applications:

- **🔍 AI Content Scraper** - Intelligent AI research paper and documentation collector
- **📊 Data Analysis Scraper** - NLP-powered content analysis with link tree generation  
- **🧠 LLM Training Data Scraper** - High-quality text data collection for language models
- **📄 PDF Document Scraper** - Academic PDF harvesting with recursive crawling

## ✨ Key Features

- **🚀 Multi-core Processing** - Automatically utilizes all available CPU cores
- **🧠 Intelligent Content Filtering** - AI-powered relevance scoring and quality assessment
- **🔄 Auto-dependency Management** - Scripts install required packages automatically
- **📊 Progress Tracking** - Real-time GUI progress indicators for all operations
- **🍎 Native macOS Integration** - Uses native dialogs and desktop logging
- **🔒 Duplicate Detection** - SHA-256 content hashing prevents redundant downloads
- **⚡ Auto-resume Capability** - State persistence allows resuming interrupted sessions
- **📈 Comprehensive Analytics** - Detailed statistics and performance metrics

## 🚀 Quick Start

1. **Clone or download** this Data Scraping folder
2. **Run any script** - dependencies install automatically
3. **Select output directory** via native macOS dialog
4. **Monitor progress** through GUI indicators
5. **Check Desktop** for log files after completion

### Example Usage

```bash
# Collect AI research content
python data-google-search-ai-scraper.py

# Analyze and extract data insights
python data-google-search-data-scraper.py  

# Gather LLM training data
python data-llm-raw-data-scraper.py

# Harvest academic PDFs
python data-pdf-document-scraper.py
```

## 📋 Requirements

- **Python 3.6+** (macOS includes this by default)
- **macOS 10.14+** (for optimal native dialog support)
- **Internet Connection** (for web scraping operations)

### Auto-installed Dependencies

All required packages are installed automatically:
- `requests` - HTTP library for web requests
- `beautifulsoup4` - HTML/XML parsing
- `google` - Google search integration
- `fake-useragent` - User agent rotation
- `nltk` - Natural language processing
- `tkinter` - GUI components (built into Python)

## 🛠️ Applications

### 🔍 AI Content Scraper
**File:** `data-google-search-ai-scraper.py`

Specialized scraper for collecting AI-related research content, papers, and documentation.

**Features:**
- Intelligent search query optimization for AI content
- Multi-format support (PDF, TXT, PY files)
- Academic content prioritization
- Duplicate detection via content hashing
- Recursive link following with depth control
- Quality-based content filtering

**Usage:**
```bash
python data-google-search-ai-scraper.py [options]

Options:
  --dest DIR        Output directory (default: GUI prompt)
  --threads NUM     Worker threads (default: CPU cores - 1)
  --depth NUM       Maximum crawl depth (default: 3)
  --quiet          Reduce console output
  --version        Show version information
```

**Search Specializations:**
- Large Language Model research
- GPT and transformer architectures
- Deep learning methodologies
- Neural network papers
- Natural language processing
- AI ethics and governance
- Machine learning advances

---

### 📊 Data Analysis Scraper
**File:** `data-google-search-data-scraper.py`

Advanced data extraction tool with natural language processing and comprehensive content analysis.

**Features:**
- NLP-powered sentence extraction and ranking
- Intelligent content quality assessment
- Link tree generation with metadata
- Multi-threaded content analysis
- Comprehensive statistics tracking
- JSON-structured output for further analysis

**Usage:**
```bash
python data-google-search-data-scraper.py [options]

Options:
  --dest DIR        Output directory (default: GUI prompt)
  --threads NUM     Worker threads (default: CPU cores - 1)
  --results NUM     Results per search query (default: 10)
  --output FILE     Output filename (default: link_tree_data.json)
  --quiet          Reduce console output
```

**Analysis Capabilities:**
- Sentence tokenization and cleaning
- Stop word removal and filtering
- Content relevance scoring
- Word frequency analysis
- Impact sentence identification
- Statistical summarization

**Output Structure:**
```json
{
  "metadata": {
    "timestamp": "2025-01-23 10:30:00",
    "num_queries": 10,
    "results_per_query": 10,
    "threads_used": 7
  },
  "queries": {
    "research_query": {
      "urls": ["url1", "url2", ...],
      "analysis": [
        {
          "url": "example.com",
          "sentences": ["extracted", "sentences"],
          "quality_score": 85,
          "timestamp": "2025-01-23 10:31:00"
        }
      ],
      "stats": {
        "total_urls": 10,
        "analyzed_urls": 8,
        "total_sentences": 150,
        "impactful_sentences": 45
      }
    }
  },
  "stats": {
    "total_queries": 10,
    "total_urls": 100,
    "analyzed_urls": 85,
    "total_sentences": 1500,
    "impactful_sentences": 450
  }
}
```

---

### 🧠 LLM Training Data Scraper
**File:** `data-llm-raw-data-scraper.py`

Specialized tool for collecting high-quality text data suitable for language model training.

**Features:**
- Advanced content quality assessment
- Configurable quality thresholds
- Intelligent text filtering and cleaning
- Vocabulary diversity analysis
- Content coherence scoring
- Optimized for LLM training datasets

**Usage:**
```bash
python data-llm-raw-data-scraper.py [options]

Options:
  --dest DIR        Output directory (default: GUI prompt)
  --threads NUM     Worker threads (default: CPU cores - 1)
  --results NUM     Results per search query (default: 10)
  --quality FLOAT   Quality threshold 0.0-1.0 (default: 0.5)
  --quiet          Reduce console output
```

**Quality Assessment Metrics:**
- **Sentence Length Analysis** - Optimal 10-30 words per sentence
- **Vocabulary Diversity** - Unique word ratio measurement
- **Content Coherence** - Logical flow and structure assessment
- **Relevance Scoring** - Topic relevance and focus evaluation
- **Noise Filtering** - Removal of low-quality content

**Quality Scoring Algorithm:**
```python
quality_score = (
    length_score * 0.3 +      # Sentence length optimization
    diversity_score * 0.4 +   # Vocabulary richness
    coherence_score * 0.3     # Content structure
)
```

**Output Files:**
- `llm_data_[query].json` - Extracted sentences with metadata
- `scraping_stats.json` - Collection statistics and metrics
- `Desktop/data-llm-raw-data-scraper.log` - Detailed operation log

---

### 📄 PDF Document Scraper
**File:** `data-pdf-document-scraper.py`

High-performance PDF harvesting tool optimized for academic and research document collection.

**Features:**
- Recursive website crawling with configurable depth
- Academic content prioritization
- PDF signature validation
- Intelligent filename sanitization
- Collision-free file naming
- Comprehensive state persistence

**Usage:**
```bash
python data-pdf-document-scraper.py [options]

Options:
  --dest DIR        Save directory (default: GUI prompt)
  --threads NUM     Worker threads (default: CPU cores - 1)
  --depth NUM       Maximum crawl depth (default: 4)
  --start-url URL   Starting URLs (can specify multiple)
  --quiet          Reduce console output
```

**Default Academic Sources:**
- arXiv.org AI papers
- NeurIPS conference proceedings
- OpenAI research publications
- Google Research publications
- Meta AI research papers
- Microsoft Research AI division

**Content Prioritization:**
1. **Direct PDF links** - Immediate download priority
2. **Academic indicators** - Papers, research, publications
3. **Domain reputation** - Trusted academic institutions
4. **Content relevance** - AI/ML related materials

**File Management:**
- Automatic duplicate detection
- Safe filename generation
- Collision-free naming (appends `_1`, `_2`, etc.)
- PDF signature verification
- Minimum size filtering (1KB+)

## ⚙️ Configuration

### Performance Tuning

All scripts automatically optimize for your system:

```python
# Automatic thread calculation
optimal_threads = max(1, multiprocessing.cpu_count() - 1)

# Rate limiting to respect websites
request_delay = 1.0  # seconds between requests
retry_attempts = 3   # number of retry attempts
```

### Quality Thresholds

Configurable quality settings for content filtering:

```python
# LLM Data Scraper quality thresholds
min_sentence_length = 5      # words
max_sentence_length = 1000   # characters
min_vocabulary_words = 3     # unique words
quality_threshold = 0.5      # 0.0-1.0 scale
```

### Search Optimization

Targeted search queries for different content types:

```python
# AI Content queries
ai_queries = [
    "LLM filetype:pdf OR filetype:txt OR filetype:py",
    "GPT transformer filetype:pdf OR filetype:txt OR filetype:py",
    "Deep Learning filetype:pdf OR filetype:txt OR filetype:py"
]

# Academic PDF queries
pdf_queries = [
    "artificial intelligence research papers",
    "machine learning methodology",
    "neural network architecture"
]
```

## 📊 Output Files & Structure

### Standard Output Structure

```
Output Directory/
├── data-files/
│   ├── ai_content_1.pdf
│   ├── research_paper_2.txt
│   └── algorithm_3.py
├── analysis/
│   ├── link_tree_data.json
│   ├── content_analysis.json
│   └── quality_metrics.json
├── logs/
│   ├── scraper_state.json
│   ├── download_state.json
│   └── scraping_stats.json
└── metadata/
    ├── file_metadata.json
    └── processing_summary.json
```

### Desktop Log Files

All scripts create log files on your Desktop:
- `data-google-search-ai-scraper.log`
- `data-google-search-data-scraper.log`
- `data-llm-raw-data-scraper.log`
- `data-pdf-document-scraper.log`

### State Persistence Files

- `scraper_state.json` - Visited URLs and downloaded files
- `download_state.json` - Hash database for duplicate detection
- `scraping_stats.json` - Performance metrics and statistics

## 🔧 Advanced Features

### Multi-threading Architecture

Each scraper uses optimal threading strategies:

```python
# CPU-bound tasks: cores - 1
cpu_threads = max(1, multiprocessing.cpu_count() - 1)

# I/O-bound tasks: cores * 2
io_threads = multiprocessing.cpu_count() * 2

# Dynamic thread allocation based on task type
with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
    futures = [executor.submit(process_url, url) for url in urls]
```

### Intelligent Content Filtering

Advanced filtering algorithms remove low-quality content:

```python
def assess_content_quality(sentences):
    # Length analysis
    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
    
    # Vocabulary diversity
    all_words = [word.lower() for s in sentences for word in s.split()]
    vocab_diversity = len(set(all_words)) / len(all_words)
    
    # Content coherence
    coherence = sum(1 for s in sentences if len(s.split()) > 10) / len(sentences)
    
    return weighted_score(length, diversity, coherence)
```

### State Management

Robust state persistence allows resuming operations:

```python
def save_state():
    state = {
        'visited_urls': list(visited_urls),
        'downloaded_files': list(downloaded_files),
        'content_hashes': list(content_hashes),
        'timestamp': datetime.now().isoformat(),
        'stats': processing_statistics
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
```

## 🐛 Troubleshooting

### Common Issues

**Rate Limiting:**
```bash
# If you encounter rate limiting errors
# Increase delays in scripts (edit request_delay values)
request_delay = 2.0  # Increase from default 1.0
```

**Memory Issues:**
```bash
# For large datasets, reduce thread count
python script.py --threads 2
```

**Network Timeouts:**
```bash
# Check internet connection and retry
# Scripts have built-in retry logic with exponential backoff
```

**Permission Errors:**
```bash
# Ensure output directory is writable
chmod 755 /path/to/output/directory
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Remove --quiet flag for detailed output
python script.py  # Full debug output
python script.py --quiet  # Minimal output
```

### Log Analysis

Check desktop log files for detailed information:

```bash
# View recent errors
grep -i error ~/Desktop/data-*-scraper.log

# Monitor real-time progress
tail -f ~/Desktop/data-*-scraper.log

# Search for specific issues
grep -i "timeout\|connection\|failed" ~/Desktop/data-*-scraper.log
```

## 📈 Performance Metrics

### Typical Performance

**AI Content Scraper:**
- Speed: ~50-100 files per hour
- Success Rate: 85-90%
- CPU Usage: 70-80% (multi-core)

**Data Analysis Scraper:**
- Speed: ~200-300 URLs per hour
- Analysis Rate: ~500 sentences per minute
- Memory Usage: <500MB typical

**LLM Data Scraper:**
- Speed: ~100-200 quality sentences per hour
- Quality Filtering: 60-70% pass rate
- Dataset Size: 10-50MB per query

**PDF Document Scraper:**
- Speed: ~20-50 PDFs per hour
- Success Rate: 80-85%
- Average File Size: 2-10MB per PDF

### Optimization Tips

1. **Adjust Thread Count:**
   ```bash
   # For faster processing (if system can handle it)
   python script.py --threads 8
   
   # For stability on older systems
   python script.py --threads 2
   ```

2. **Quality vs Speed Trade-off:**
   ```bash
   # Higher quality, slower processing
   python data-llm-raw-data-scraper.py --quality 0.8
   
   # Lower quality, faster processing  
   python data-llm-raw-data-scraper.py --quality 0.3
   ```

3. **Memory Management:**
   ```bash
   # Reduce concurrent processing for large datasets
   python script.py --threads 1 --results 5
   ```

## 🔗 Integration

### Workflow Integration

These scrapers work excellently together:

```bash
# 1. Collect AI research papers
python data-pdf-document-scraper.py --dest /research/pdfs/

# 2. Extract structured data from findings
python data-google-search-data-scraper.py --dest /research/analysis/

# 3. Gather high-quality training text
python data-llm-raw-data-scraper.py --dest /research/training_data/

# 4. Supplement with specific AI content
python data-google-search-ai-scraper.py --dest /research/ai_content/
```

### API Integration

Output files are structured for easy integration:

```python
# Load and process scraper results
import json

# Load link tree analysis
with open('link_tree_data.json', 'r') as f:
    analysis_data = json.load(f)

# Extract high-quality sentences
quality_sentences = []
for query_data in analysis_data['queries'].values():
    for analysis in query_data['analysis']:
        if analysis['quality_score'] > 70:
            quality_sentences.extend(analysis['sentences'])

# Use for downstream processing
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## 📞 Support

- **Issues:** Report bugs via GitHub Issues
- **Email:** sanchez314c@speedheathens.com
- **Documentation:** Check individual script headers for detailed usage

## 🔗 Related Projects

- [Audio Processing Scripts](../Audio/) - Audio manipulation and analysis tools
- [AI/ML Tools](../AI-ML/) - Machine learning and AI utilities
- [System Utilities](../System/) - System administration scripts

---

**Author:** sanchez314c@speedheathens.com  
**Version:** 1.0.0  
**Last Updated:** 2025-01-23  
**Platform:** macOS Universal  

*GET SWIFTY with Data! 🚀*