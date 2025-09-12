# AI-ML Tools Collection

A comprehensive collection of AI and Machine Learning tools designed for researchers, developers, and enthusiasts working with Large Language Models (LLMs) and natural language processing.

## 🚀 Scripts Overview

### 1. AI LLM Terminal Chat Interface
**File:** `ai-llm-terminal-chat-interface.py`

A universal terminal-based chat interface supporting multiple LLM providers with advanced features.

**Features:**
- Multi-provider support: OpenAI, Anthropic, Google AI, Perplexity, Groq, Mistral, Jamba, xAI, Ollama
- Interactive terminal interface with readline support
- Conversation history management and persistence
- Streaming responses for real-time interaction
- Secure credential management with GUI prompts
- Model switching and system prompt integration
- Token usage tracking

**Usage:**
```bash
python ai-llm-terminal-chat-interface.py [--model provider/model] [--system prompt_file]
```

**Examples:**
```bash
# Use OpenAI GPT-4o
python ai-llm-terminal-chat-interface.py --model openai/gpt-4o

# Use Anthropic Claude with custom system prompt
python ai-llm-terminal-chat-interface.py --model anthropic/claude-3-5-sonnet-20241022 --system my_prompt.txt

# Use local Ollama model
python ai-llm-terminal-chat-interface.py --model ollama/llama3.1
```

### 2. AI LLM Comparison Suite
**File:** `ai-llm-comparison-suite.py`

Comprehensive testing suite for comparing responses across multiple LLM providers with detailed analytics.

**Features:**
- Parallel processing for efficient comparisons
- Support for all major LLM providers
- Detailed performance metrics and analysis
- Response quality evaluation
- Statistical analysis and common phrase detection
- CSV and JSON export for further analysis
- Progress tracking with GUI

**Usage:**
```bash
python ai-llm-comparison-suite.py [--models MODEL1,MODEL2] [--prompt FILE] [--output FILE]
```

**Examples:**
```bash
# Compare specific models
python ai-llm-comparison-suite.py --models "openai/gpt-4o,anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro"

# Use prompt from file
python ai-llm-comparison-suite.py --prompt comparison_prompt.txt

# Save results to specific location
python ai-llm-comparison-suite.py --output my_comparison_results.json
```

### 3. AI Prompt Optimizer
**File:** `ai-prompt-optimizer.py`

Advanced NLP tool for analyzing and optimizing text prompts across multiple dimensions.

**Features:**
- Multi-dimensional optimization: technical clarity, emotional resonance, persuasiveness, educational effectiveness
- State-of-the-art NLP analysis using spaCy and Transformers
- Readability scoring and complexity metrics
- Sentiment analysis and emotional profiling
- Batch processing capabilities
- GPU acceleration support
- Comprehensive reporting with before/after analysis

**Usage:**
```bash
python ai-prompt-optimizer.py [--input FILE] [--mode MODE] [--output FILE] [--batch]
```

**Optimization Modes:**
- `technical`: Optimize for technical clarity and precision
- `emotional`: Enhance emotional resonance and engagement
- `persuasive`: Improve persuasive impact and call-to-action
- `educational`: Optimize for learning and comprehension
- `all`: Apply all optimization strategies (default)

**Examples:**
```bash
# Optimize single prompt for all dimensions
python ai-prompt-optimizer.py --input my_prompt.txt --mode all

# Optimize for persuasiveness only
python ai-prompt-optimizer.py --input sales_copy.txt --mode persuasive

# Batch process multiple files
python ai-prompt-optimizer.py --input prompts_folder/ --batch --output optimized_prompts/
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- macOS (native dialog support)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Setup
1. **spaCy Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK Data:**
   ```bash
   python -m nltk.downloader punkt vader_lexicon
   ```

3. **API Keys:**
   Create a `.env` file in the AI-ML directory with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_AI_API_KEY=your_google_ai_key_here
   PERPLEXITY_API_KEY=your_perplexity_key_here
   GROQ_API_KEY=your_groq_key_here
   MISTRAL_API_KEY=your_mistral_key_here
   AI21_API_KEY=your_ai21_key_here
   XAI_API_KEY=your_xai_key_here
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## 🔑 Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo, o1-preview, o1-mini | Full API support |
| **Anthropic** | Claude-3.5-Sonnet, Claude-3.5-Haiku, Claude-3-Opus | Streaming supported |
| **Google AI** | Gemini-1.5-Pro, Gemini-1.5-Flash, Gemini-1.0-Pro | Direct API integration |
| **Perplexity** | Llama-3.1-Sonar models, Llama-3.1 Instruct | Online search capabilities |
| **Groq** | Llama-3.1, Mixtral, Gemma models | High-speed inference |
| **Mistral** | Mistral-Large, Medium, Small, Codestral | European AI provider |
| **AI21 (Jamba)** | Jamba-1.5-Large, Jamba-1.5-Mini | Advanced reasoning |
| **xAI** | Grok-Beta, Grok-Vision-Beta | Elon Musk's AI models |
| **Ollama** | Llama3.1, Llama3.2, Mistral, CodeLlama, DeepSeek-Coder | Local deployment |

## 📊 Features & Capabilities

### Universal Features (All Scripts)
- **Cross-Platform Compatibility:** Designed for macOS with universal paths
- **Desktop Logging:** All logs automatically saved to `~/Desktop/script-name.log`
- **GUI Integration:** Native macOS dialogs for file selection and credentials
- **Environment Variables:** Support for `.env` files and environment variables
- **Error Handling:** Comprehensive error handling and user feedback
- **Progress Tracking:** Visual progress indicators for long-running operations

### Security Features
- Secure credential storage in user home directory
- Environment variable support for CI/CD pipelines
- API key masking in dialogs
- No hardcoded secrets in source code

### Performance Features
- Multi-threading for parallel processing
- GPU acceleration where available
- Memory-efficient processing for large datasets
- Optimized for Apple Silicon and Intel Macs

## 📝 Output Files

All scripts generate detailed log files on the desktop:
- `ai-llm-terminal-chat-interface.log`: Chat session logs and API interactions
- `ai-llm-comparison-suite.log`: Comparison results and performance metrics
- `ai-prompt-optimizer.log`: Optimization analysis and improvements

## 🔍 Use Cases

### Research & Development
- Compare model performance across providers
- Analyze prompt effectiveness and optimization
- Test different approaches to AI interaction
- Benchmark response quality and speed

### Content Creation
- Optimize marketing copy and sales materials
- Enhance educational content for clarity
- Improve technical documentation
- Create engaging social media content

### Business Applications
- Evaluate LLM providers for enterprise use
- Optimize customer service prompts
- Enhance chatbot personalities and responses
- Create standardized AI interaction workflows

## 🛠️ Customization

### Adding New Providers
Each script is designed with extensible provider architecture. To add a new LLM provider:

1. Create a new provider class inheriting from `ChatProvider`
2. Implement required methods: `get_available_models()`, `send_message()`, `stream_message()`
3. Add provider to the providers dictionary in the main class
4. Update documentation and model lists

### Configuration Options
- Modify default models in script configuration
- Adjust timeout values for API calls
- Customize optimization parameters
- Configure logging levels and formats

## 📄 License

These scripts are part of Jason Paul Michaels' Script Library. See the main repository for license information.

## 🤝 Contributing

For bug reports, feature requests, or contributions, please visit:
- **GitHub:** https://github.com/sanchez314c
- **Issues:** Create issues in the main Script Library repository

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com)
- [Google AI API](https://ai.google.dev)
- [Ollama Documentation](https://ollama.ai/docs)
- [spaCy Documentation](https://spacy.io)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

*Last Updated: January 23, 2025*
*Version: 1.0.0*