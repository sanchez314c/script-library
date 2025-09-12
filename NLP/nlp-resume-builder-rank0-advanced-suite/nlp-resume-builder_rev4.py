# Standard library imports
import os
# Configure environment variables before any other imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Prevent online model downloads

import json
from datetime import datetime
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gc
import psutil
import sys

# GUI imports
import tkinter as tk
from tkinter import filedialog

# Data processing and ML imports
import numpy as np
import pandas as pd
import torch
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Transformer and deep learning imports
from transformers import (
    pipeline,
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel
)

# Scikit-learn imports
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.decomposition import (
    LatentDirichletAllocation,
    TruncatedSVD
)
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Configure comprehensive logging
import logging
import logging.handlers

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler('ai_resume_generator.log')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(detailed_formatter)
console_handler.setFormatter(detailed_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def clear_system_memory():
    """Aggressive memory cleanup"""
    try:
        # Force Python garbage collection
        gc.collect()
        
        # Clear CUDA/MPS cache if available
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Force system memory release (Unix/Linux/MacOS)
        if sys.platform != 'win32':
            os.system('sync')  # Sync filesystem
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3')
            except:
                pass  # Might not have permission, that's OK
        
        # Get process
        process = psutil.Process()
        
        # Release memory back to OS (Unix/Linux/MacOS)
        if sys.platform != 'win32':
            os.system('purge')
        
        # Memory info logging
        memory_info = process.memory_info()
        total_memory = psutil.virtual_memory()
        logger.info(f"Memory Usage - RSS: {memory_info.rss / 1024 / 1024:.2f}MB, "
                   f"VMS: {memory_info.vms / 1024 / 1024:.2f}MB, "
                   f"System Available: {total_memory.available / 1024 / 1024:.2f}MB")
                   
    except Exception as e:
        logger.error(f"Error in memory cleanup: {e}")

def monitor_memory(threshold_gb=32):
    """Monitor memory usage and clean if needed"""
    try:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < threshold_gb:
            logger.warning(f"Low memory: {available_gb:.2f}GB available. Cleaning...")
            clear_system_memory()
    except Exception as e:
        logger.error(f"Error monitoring memory: {e}")

# Configure processing settings
NUM_CORES = max(1, int(cpu_count() * 0.8))  # Use 80% of available cores
CHUNK_SIZE = 32  # Optimized for GPU batch processing
BATCH_SIZE = 32  # Optimized for GPU efficiency
MAX_TEXT_LENGTH = 512  # Match BERT's token limit
MAX_TOKENS = 512  # BERT's maximum token limit
MAX_MEMORY_GB = 4  # Conservative memory limit for RX580
DEVICE_BATCH_SIZE = 32  # Process in GPU-optimized batches

# Initial memory cleanup
clear_system_memory()

# Comprehensive AI skills list
AI_SKILLS = [
    "machine learning", "deep learning", "neural networks", "natural language processing",
    "computer vision", "data analysis", "data visualization", "python programming",
    "model training", "hyperparameter tuning", "reinforcement learning", "unsupervised learning",
    "supervised learning", "tensorflow", "pytorch", "scikit-learn", "data preprocessing",
    "algorithm development", "artificial intelligence", "predictive modeling", "big data",
    "cloud computing", "database management", "feature engineering", "statistical analysis",
    "time series analysis", "anomaly detection", "image processing", "text mining",
    "transfer learning", "model evaluation", "optimization techniques", "MLOps",
    "data engineering", "GPU programming", "distributed computing", "quantum computing",
    "robotic process automation", "computer graphics", "game development", "mobile development",
    "web development", "system architecture", "network security", "blockchain technology"
]

# Job titles
JOB_TITLES = [
    "AI/ML Engineer", "AI Ethics Officer", "UI/UX Designer/Dev",
    "Natural Language Processing Engineer", "Researcher", "Research Scientist",
    "AI Engineer", "Data Mining and Analysis", "Data Scientist",
    "Business Intelligence Developer", "AI Data Analytics", "AI Consultant",
    "AI Sales", "AI Product Manager", "AI Safety Engineer",
    "Machine Learning Engineer", "Deep Learning Specialist", "Computer Vision Engineer",
    "MLOps Engineer", "AI Infrastructure Architect", "AI Solutions Architect"
]

class GPUManager:
    def __init__(self):
        self.available_gpus = self._get_available_gpus()
        logger.info(f"Available GPUs: {len(self.available_gpus)}")
        
    def _get_available_gpus(self):
        gpus = []
        try:
            # Check for MPS-capable GPUs
            if torch.backends.mps.is_available():
                # Assuming dual RX580s
                gpus = ['mps:0', 'mps:1']
                logger.info("Detected multiple MPS-capable GPUs")
        except Exception as e:
            logger.warning(f"Error detecting GPUs: {e}")
        return gpus or ['cpu']

class ParallelProcessor:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.device = self._setup_device()
        self.nlp = self._setup_nlp()
        
    def _setup_device(self):
        try:
            if 'mps' in self.gpu_id:
                return torch.device(self.gpu_id)
            return torch.device('cpu')
        except Exception as e:
            logger.warning(f"Error setting up device {self.gpu_id}: {e}")
            return torch.device('cpu')
    
    def _setup_nlp(self):
        """Initialize spaCy for this processor"""
        try:
            if not spacy.util.is_package('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
            return spacy.load('en_core_web_sm')
        except Exception as e:
            logger.error(f"Error loading spaCy model in processor: {e}")
            raise
            
    def process_chunk(self, conversations):
        # Set environment variable for this process
        os.environ['MPS_TARGET_GPU'] = self.gpu_id.split(':')[1] if 'mps' in self.gpu_id else ''
        
        generator = AIResumeGenerator(device=self.device, nlp=self.nlp)
        return generator.process_conversations_chunk(conversations)

# Initialize transformers with error handling
try:
    # Load models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    base_model = AutoModel.from_pretrained('bert-base-uncased', local_files_only=True)
    
    # Configure summarizer to run on CPU
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # Force CPU for summarization
    )
    
    sia = SentimentIntensityAnalyzer()
    logger.info("Base models successfully loaded")
except Exception as e:
    logger.error(f"Error initializing transformers: {e}")
    raise

class AIResumeGenerator:
    def __init__(self, device=None, nlp=None):
        logger.info("Initializing AIResumeGenerator")
        self.device = device if device is not None else self._setup_default_device()
        self.nlp = nlp if nlp is not None else self._setup_nlp()
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()
        
        # Initialize these after setting up NLP tools
        self.ai_skills = self.expand_keywords_with_synonyms(AI_SKILLS)
        self.job_titles = JOB_TITLES
        self.processed_data = defaultdict(list)
        self.temporal_data = defaultdict(dict)
        
        # Clear memory after initialization
        clear_system_memory()
        
        logger.debug(f"AIResumeGenerator initialized successfully on {self.device}")

    def expand_keywords_with_synonyms(self, keywords):
        """Expand keywords with their synonyms using WordNet"""
        try:
            expanded = set(keywords)
            for keyword in keywords:
                # Handle multi-word keywords
                words = keyword.split()
                for word in words:
                    try:
                        for syn in wordnet.synsets(word):
                            # Only add technical/relevant synonyms
                            for lemma in syn.lemmas():
                                candidate = lemma.name().replace('_', ' ')
                                # Filter out non-technical or irrelevant synonyms
                                if (len(candidate) > 2 and  # Avoid short abbreviations
                                    not any(char.isdigit() for char in candidate) and  # No numbers
                                    candidate.isalnum()):  # Only alphanumeric
                                    expanded.add(candidate)
                    except Exception as word_error:
                        logger.debug(f"Error processing word '{word}': {word_error}")
                        continue
                        
            # Remove duplicates and sort
            result = sorted(list(expanded))
            logger.debug(f"Expanded {len(keywords)} keywords to {len(result)} terms")
            return result
            
        except Exception as e:
            logger.error(f"Error in keyword expansion: {e}")
            return list(keywords)  # Return original keywords if expansion fails

    def _setup_default_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps:0')
        return torch.device('cpu')

    def _setup_nlp(self):
        """Initialize spaCy for this instance"""
        try:
            if not spacy.util.is_package('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
            return spacy.load('en_core_web_sm')
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise

    def _setup_tokenizer(self):
        """Initialize tokenizer for this instance"""
        try:
            return AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise

    def _setup_model(self):
        """Setup model for this instance with proper device placement"""
        try:
            model = base_model.to(self.device)
            logger.info(f"Model successfully moved to {self.device}")
            return model
        except Exception as e:
            logger.error(f"Error setting up model on {self.device}: {e}")
            logger.info("Falling back to CPU")
            return base_model.to('cpu')

    def process_conversations_chunk(self, conversations):
        """Process a chunk of conversations for parallel processing"""
        try:
            logger.info(f"Processing chunk of {len(conversations)} conversations on {self.device}")
            local_processed_data = defaultdict(list)
            local_temporal_data = defaultdict(dict)
            
            # Process in batches for GPU efficiency
            for i in range(0, len(conversations), BATCH_SIZE):
                # Check memory every few batches
                if i % (BATCH_SIZE * 4) == 0:
                    monitor_memory(threshold_gb=32)
                    
                batch = conversations[i:i + BATCH_SIZE]
                try:
                    for conv in batch:
                        result = self._process_single_conversation(conv)
                        if result:
                            local_data, temporal_data = result
                            self._merge_results([(local_data, temporal_data)])
                            
                        # Release memory after each conversation
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
                
                # Clear GPU memory after each batch
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                    
            return self.processed_data, self.temporal_data
        except Exception as e:
            logger.error(f"Error processing conversation chunk: {e}")
            return defaultdict(list), defaultdict(dict)
        finally:
            # Ensure memory cleanup after chunk processing
            clear_system_memory()

    def _process_single_conversation(self, conversation):
        """Process individual conversation with enhanced error handling"""
        try:
            mapping = conversation.get('mapping', {})
            if not mapping and isinstance(conversation, dict):
                mapping = {'root': conversation}
            
            batch_texts = []
            batch_dates = []
            batch_metadata = []
            
            for node_id, node in mapping.items():
                try:
                    message = node.get('message', {})
                    if not message:
                        continue
                        
                    content = message.get('content', {})
                    text = self._extract_text_content(content)
                    
                    if text and text.strip():
                        timestamp = node.get('create_time')
                        date = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
                        
                        # Pre-process text before adding to batch
                        text = self._truncate_text(text)
                        batch_texts.append(text)
                        batch_dates.append(date)
                        batch_metadata.append({'node_id': node_id, 'timestamp': timestamp})
                        
                except Exception as node_error:
                    logger.warning(f"Error processing node {node_id}: {node_error}")
                    continue
            
            if batch_texts:
                local_processed_data = defaultdict(list)
                local_temporal_data = defaultdict(dict)
                self._analyze_content_batch(batch_texts, batch_dates, batch_metadata, 
                                         local_processed_data, local_temporal_data)
                return local_processed_data, local_temporal_data
                
            return None
            
        except Exception as e:
            logger.error(f"Error in single conversation processing: {str(e)}")
            return None
        finally:
            # Clean up memory after processing conversation
            gc.collect()

    def _extract_text_content(self, content):
        """Extract text content from various content formats"""
        try:
            if isinstance(content, dict):
                parts = content.get('parts', [])
                return ' '.join(str(part) for part in parts if part)
            elif isinstance(content, list):
                return ' '.join(str(part) for part in content if part)
            elif isinstance(content, str):
                return content
            else:
                return ''
        except Exception as e:
            logger.warning(f"Error extracting text content: {e}")
            return ''

    def _truncate_text(self, text):
        """Truncate text to fit within BERT token limits"""
        try:
            tokens = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=MAX_TOKENS-10,
                return_tensors="pt"
            )
            return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error in truncation: {e}")
            return text[:MAX_TEXT_LENGTH]

    def _analyze_content_batch(self, texts, dates, metadata, local_processed_data, local_temporal_data):
        """Analyze a batch of content with optimized GPU usage"""
        try:
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            
            # Process texts in GPU-optimized batches
            for i in range(0, len(texts), DEVICE_BATCH_SIZE):
                # Monitor memory usage
                if i % (DEVICE_BATCH_SIZE * 2) == 0:
                    monitor_memory(threshold_gb=32)
                    
                batch_texts = texts[i:i + DEVICE_BATCH_SIZE]
                batch_dates = dates[i:i + DEVICE_BATCH_SIZE]
                batch_meta = metadata[i:i + DEVICE_BATCH_SIZE]
                
                try:
                    # Tokenize entire batch at once
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_TOKENS,
                        padding=True
                    )
                    
                    # Move entire batch to GPU
                    if self.device.type == 'mps':
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate embeddings for entire batch
                    with torch.no_grad():
                        try:
                            outputs = self.model(**inputs)
                            embeddings = outputs.last_hidden_state.mean(dim=1)
                            
                            # Process batch results while keeping embeddings on GPU
                            for idx, (text, date, meta) in enumerate(zip(batch_texts, batch_dates, batch_meta)):
                                self._process_text_content(
                                    text, date, meta, 
                                    embeddings[idx:idx+1],
                                    local_processed_data, 
                                    local_temporal_data,
                                    device=self.device
                                )
                                
                                # Clean up after each item
                                gc.collect()
                            
                            # Clear GPU memory after batch processing
                            if self.device.type == 'mps':
                                torch.mps.empty_cache()
                                
                        except RuntimeError as e:
                            logger.warning(f"Runtime error in embedding generation: {e}")
                            continue
                            
                except Exception as batch_error:
                    logger.error(f"Error processing batch: {batch_error}")
                    continue

        except Exception as e:
            logger.error(f"Critical error in batch processing: {str(e)}")
            raise
        finally:
            # Ensure memory cleanup
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()

    def _process_text_content(self, text, date, meta, embedding, local_processed_data, local_temporal_data, device=None):
        """Process individual text content with GPU optimization"""
        try:
            # Extract skills
            skills = self._extract_skills(text)
            for skill in skills:
                month_key = date.strftime('%Y-%m')
                if skill not in local_temporal_data:
                    local_temporal_data[skill] = {}
                local_temporal_data[skill][month_key] = \
                    local_temporal_data[skill].get(month_key, 0) + 1

            # Generate summary for appropriate length content
            text_length = len(text.split())
            if 100 < text_length <= MAX_TEXT_LENGTH:
                try:
                    if device and device.type == 'mps':
                        # Use GPU for summarization
                        summary_inputs = self.tokenizer(
                            text, 
                            return_tensors="pt", 
                            max_length=MAX_TOKENS, 
                            truncation=True
                        )
                        summary_inputs = {k: v.to(device) for k, v in summary_inputs.items()}
                        with torch.no_grad():
                            summary_outputs = self.model(**summary_inputs)
                            summary = self.tokenizer.decode(
                                self.tokenizer.encode(
                                    text,
                                    max_length=130,
                                    truncation=True
                                ),
                                skip_special_tokens=True
                            )
                            
                            # Clean up GPU memory
                            if device.type == 'mps':
                                torch.mps.empty_cache()
                    else:
                        summary = summarizer(
                            text,
                            max_length=min(130, text_length),
                            min_length=30,
                            do_sample=False
                        )[0]['summary_text']
                except Exception as sum_error:
                    logger.warning(f"Error in summarization: {sum_error}")
                    summary = text[:200] + "..."
            else:
                summary = text[:200] + "..."

            # Process projects
            projects = self._extract_projects(text)
            if projects:
                for project in projects:
                    if device and device.type == 'mps':
                        project['embedding'] = embedding.cpu().numpy()
                    else:
                        project['embedding'] = embedding[0].numpy()
                    project['date'] = date
                    project['metadata'] = meta
                    local_processed_data['projects'].append(project)

            # Sentiment analysis
            try:
                if device and device.type == 'mps':
                    # Use GPU for sentiment analysis
                    sentiment_inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        max_length=MAX_TOKENS, 
                        truncation=True
                    )
                    sentiment_inputs = {k: v.to(device) for k, v in sentiment_inputs.items()}
                    with torch.no_grad():
                        sentiment_outputs = self.model(**sentiment_inputs)
                        sentiment_embedding = sentiment_outputs.last_hidden_state.mean(dim=1)
                        compound_score = sentiment_embedding.mean().item()
                        
                        # Clean up GPU memory
                        if device.type == 'mps':
                            torch.mps.empty_cache()
                else:
                    sentiment = sia.polarity_scores(text)
                    compound_score = sentiment['compound']

                if compound_score > 0.2:
                    achievement = {
                        'content': summary,
                        'full_content': text,
                        'sentiment': compound_score,
                        'date': date,
                        'metadata': meta,
                        'embedding': embedding.cpu().numpy() if device and device.type == 'mps' else embedding[0].numpy()
                    }
                    local_processed_data['achievements'].append(achievement)
            except Exception as sent_error:
                logger.warning(f"Error in sentiment analysis: {sent_error}")

            # Extract entities
            entities = self._extract_entities(text)
            if entities:
                local_processed_data['entities'].extend(entities)

        except Exception as e:
            logger.error(f"Error in text content processing: {e}")
        finally:
            # Clean up memory
            gc.collect()

    def _extract_skills(self, text):
        """Extract skills with context awareness"""
        try:
            skills = set()
            text_lower = text.lower()
            doc = self.nlp(text_lower)
            
            for skill in self.ai_skills:
                skill_lower = skill.lower()
                if skill_lower in text_lower:
                    for chunk in doc.noun_chunks:
                        if skill_lower in chunk.text:
                            skills.add(skill)
                            break
            
            return list(skills)
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return []
        finally:
            # Clean up spaCy doc object
            del doc
            gc.collect()

    def _extract_projects(self, text):
        """Extract project descriptions with context"""
        try:
            doc = self.nlp(text)
            projects = []
            
            project_keywords = [
                'project', 'developed', 'created', 'built', 'implemented', 
                'designed', 'architected', 'engineered', 'deployed'
            ]
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if any(keyword in sent_text for keyword in project_keywords):
                    context = self._get_sentence_context(doc, sent)
                    
                    projects.append({
                        'description': sent.text.strip(),
                        'context': context,
                        'entities': [(ent.text, ent.label_) for ent in sent.ents]
                    })
            
            return projects
        except Exception as e:
            logger.error(f"Error extracting projects: {str(e)}")
            return []
        finally:
            # Clean up spaCy doc object
            del doc
            gc.collect()

    def _extract_entities(self, text):
        """Extract named entities with context"""
        try:
            doc = self.nlp(text[:MAX_TEXT_LENGTH])
            
            relevant_types = {
                'ORG', 'PRODUCT', 'GPE', 'PERSON', 'DATE', 'WORK_OF_ART',
                'EVENT', 'LANGUAGE', 'LAW', 'MONEY', 'PERCENT'
            }
            
            entities = []
            for ent in doc.ents:
                if ent.label_ in relevant_types:
                    start_idx = max(0, ent.start - 3)
                    end_idx = min(len(doc), ent.end + 3)
                    context = doc[start_idx:end_idx].text
                    
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'context': context
                    })
            
            return entities
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return []
        finally:
            # Clean up spaCy doc object
            del doc
            gc.collect()

    def _get_sentence_context(self, doc, sent):
        """Get surrounding context for a sentence"""
        try:
            sentences = list(doc.sents)
            sent_index = sentences.index(sent)
            
            context = []
            if sent_index > 0:
                context.append(sentences[sent_index - 1].text)
            if sent_index < len(sentences) - 1:
                context.append(sentences[sent_index + 1].text)
            
            return ' '.join(context)
        except Exception as e:
            logger.error(f"Error getting sentence context: {str(e)}")
            return ""

    def _merge_results(self, results):
        """Merge batch processing results with duplicate handling"""
        try:
            for local_processed_data, local_temporal_data in results:
                # Merge processed data
                for key, values in local_processed_data.items():
                    if key == 'projects':
                        existing_descriptions = {p['description'] 
                                              for p in self.processed_data[key]}
                        unique_projects = [p for p in values 
                                         if p['description'] not in existing_descriptions]
                        self.processed_data[key].extend(unique_projects)
                    else:
                        self.processed_data[key].extend(values)
                
                # Merge temporal data
                for skill, timeline in local_temporal_data.items():
                    if skill not in self.temporal_data:
                        self.temporal_data[skill] = {}
                    for month_key, count in timeline.items():
                        self.temporal_data[skill][month_key] = \
                            self.temporal_data[skill].get(month_key, 0) + count
                            
                # Clean up merged data
                gc.collect()
                            
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            raise

def process_parallel(input_file, num_gpus=2):
    """Process conversations using multiple GPUs in parallel"""
    try:
        # Clear memory before starting
        clear_system_memory()
        
        # Load conversations in chunks to avoid memory spike
        conversations = []
        chunk_size = 1000  # Adjust based on available RAM
        
        logger.info("Loading conversations in chunks...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_size = len(data)
            
            for i in range(0, total_size, chunk_size):
                chunk = data[i:i + chunk_size]
                conversations.extend(chunk)
                
                # Monitor memory after each chunk
                monitor_memory(threshold_gb=32)
                
                logger.info(f"Loaded {len(conversations)}/{total_size} conversations")
        
        if not conversations:
            raise ValueError("Empty data loaded from JSON file")
        
        total_conversations = len(conversations)
        logger.info(f"Loaded {total_conversations} conversations")
        
        # Initialize GPU manager
        gpu_manager = GPUManager()
        available_gpus = gpu_manager.available_gpus[:num_gpus]
        
        if len(available_gpus) < 2 or total_conversations < 2:
            logger.warning("Falling back to single GPU processing")
            generator = AIResumeGenerator()
            generator.process_conversations_chunk(conversations)
            return generator.processed_data, generator.temporal_data
            
        # Calculate chunk size with minimum threshold
        num_chunks = min(len(available_gpus) * 2, total_conversations // 100 + 1)
        chunk_size = max(total_conversations // num_chunks, 50)
        
        logger.info(f"Processing with {num_chunks} chunks of approximately {chunk_size} conversations each")
        
        # Create chunks
        chunks = []
        for i in range(0, total_conversations, chunk_size):
            chunk = conversations[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
            
            # Monitor memory after creating each chunk
            if i % (chunk_size * 4) == 0:
                monitor_memory(threshold_gb=32)
        
        logger.info(f"Created {len(chunks)} chunks for processing")

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                gpu_id = available_gpus[chunk_idx % len(available_gpus)]
                processor = ParallelProcessor(gpu_id)
                futures.append(executor.submit(processor.process_chunk, chunk))
            
            # Collect results with progress tracking
            all_processed_data = defaultdict(list)
            all_temporal_data = defaultdict(dict)
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc="Processing on multiple GPUs"):
                try:
                    processed_data, temporal_data = future.result()
                    # Merge results
                    for key, value in processed_data.items():
                        all_processed_data[key].extend(value)
                    for skill, timeline in temporal_data.items():
                        if skill not in all_temporal_data:
                            all_temporal_data[skill] = {}
                        for month_key, count in timeline.items():
                            all_temporal_data[skill][month_key] = \
                                all_temporal_data[skill].get(month_key, 0) + count
                                
                    # Clean up after processing each chunk
                    gc.collect()
                    monitor_memory(threshold_gb=32)
                    
                except Exception as e:
                    logger.error(f"Error processing GPU chunk: {e}")
                    
        return all_processed_data, all_temporal_data
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise
    finally:
        # Ensure memory cleanup after processing
        clear_system_memory()

def main():
    try:
        # Clear memory before starting
        logger.info("Performing initial memory cleanup...")
        clear_system_memory()
        
        # Initialize Tkinter
        root = tk.Tk()
        root.withdraw()

        # Prompt for input file
        logger.info("Requesting input file selection...")
        input_file = filedialog.askopenfilename(
            title="Select conversations.json file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not input_file:
            logger.warning("No input file selected. Exiting...")
            return

        # Prompt for output directory
        logger.info("Requesting output directory selection...")
        output_dir = filedialog.askdirectory(
            title="Select output directory for resume files"
        )
        
        if not output_dir:
            logger.warning("No output directory selected. Exiting...")
            return

        # Process using multiple GPUs
        logger.info("Starting parallel processing on multiple GPUs...")
        processed_data, temporal_data = process_parallel(input_file, num_gpus=2)
        
        # Clean up memory before final processing
        clear_system_memory()
        
        # Initialize generator for final resume generation
        generator = AIResumeGenerator()
        generator.processed_data = processed_data
        generator.temporal_data = temporal_data
        
        logger.info("Generating resume components...")
        generator.generate_resume(output_dir)
        
        logger.info(f"Resume generation complete. Check '{output_dir}' for all components.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")
        logger.info("Check ai_resume_generator.log for details.")
    finally:
        # Ensure proper cleanup
        clear_system_memory()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    try:
        # Initial memory cleanup
        clear_system_memory()
        
        # Initialize NLTK data
        try:
            nltk.download([
                'punkt',
                'stopwords',
                'wordnet',
                'vader_lexicon'
            ], quiet=True)
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
        
        # Initialize spaCy if needed
        try:
            if not spacy.util.is_package('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
        except Exception as e:
            logger.error(f"Error initializing spaCy: {e}")
        
        freeze_support()
        main()
    except Exception as e:
        logger.error(f"Error in script execution: {e}")
    finally:
        # Final cleanup
        clear_system_memory()
