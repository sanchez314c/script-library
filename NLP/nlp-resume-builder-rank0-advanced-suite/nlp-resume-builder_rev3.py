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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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

# Configure processing settings
NUM_CORES = max(1, int(cpu_count() * 0.8))  # Use 80% of available cores
CHUNK_SIZE = 50  # Smaller chunks for better memory management
BATCH_SIZE = 1  # Process one at a time for MPS
MAX_TEXT_LENGTH = 512  # Match BERT's token limit
MAX_TOKENS = 512  # BERT's maximum token limit
MAX_MEMORY_GB = 4  # Conservative memory limit for RX580
DEVICE_BATCH_SIZE = 1  # Process one item at a time on device

# Initialize device for PyTorch with better error handling
device = torch.device('cpu')  # Default to CPU
try:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("MPS (Metal Performance Shaders) available - using RX580")
    else:
        logger.info("MPS not available, using CPU")
except:
    logger.info("Error checking MPS availability, falling back to CPU")

# Initialize transformers with error handling
try:
    # Load models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    model = AutoModel.from_pretrained('bert-base-uncased', local_files_only=True)
    
    # Move model to device if using MPS
    if device.type == 'mps':
        model = model.to(device)
        logger.info("Model successfully moved to MPS device")
    
    # Configure summarizer to run on CPU
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # Force CPU for summarization
    )
    
    sia = SentimentIntensityAnalyzer()
    logger.info(f"Models successfully loaded on {device}")
except Exception as e:
    logger.error(f"Error initializing transformers: {e}")
    raise

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

class AIResumeGenerator:
    def __init__(self):
        logger.info("Initializing AIResumeGenerator")
        self.ai_skills = self._expand_keywords_with_synonyms(AI_SKILLS)
        self.job_titles = JOB_TITLES
        self.processed_data = defaultdict(list)
        self.temporal_data = defaultdict(dict)
        self.device = device
        logger.debug("AIResumeGenerator initialized successfully")

    def _truncate_text(self, text):
        """Truncate text to fit within BERT token limits with better error handling"""
        try:
            # First try encoding with truncation
            tokens = tokenizer.encode(
                text,
                truncation=True,
                max_length=MAX_TOKENS-10,  # Leave room for special tokens
                return_tensors="pt"
            )
            
            # Decode back to text
            truncated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            
            # Verify the truncated text isn't empty
            if not truncated_text.strip():
                logger.warning("Truncation resulted in empty text, using fallback method")
                return text[:MAX_TEXT_LENGTH]
                
            return truncated_text
            
        except Exception as e:
            logger.warning(f"Error in smart truncation: {e}, using fallback method")
            return text[:MAX_TEXT_LENGTH]

    def _expand_keywords_with_synonyms(self, keywords):
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

    def process_conversations(self, input_file):
        """Process JSON conversations data with enhanced error handling and progress tracking"""
        logger.info(f"Starting to process conversations from {input_file}")
        try:
            # Load and validate JSON data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                raise ValueError("Empty data loaded from JSON file")
                
            total_conversations = len(data) if isinstance(data, list) else 1
            logger.info(f"Successfully loaded JSON data with {total_conversations} conversations")
            
            if isinstance(data, dict):
                logger.debug("Processing single conversation")
                self._process_single_conversation(data)
            elif isinstance(data, list):
                logger.info(f"Processing {total_conversations} conversations...")
                
                # Create smaller batches for processing
                batch_size = BATCH_SIZE
                conversation_batches = [
                    data[i:i + batch_size] 
                    for i in range(0, len(data), batch_size)
                ]
                
                # Process batches with progress bar
                with tqdm(total=len(conversation_batches), 
                         desc="Processing conversation batches") as pbar:
                    for i, batch in enumerate(conversation_batches):
                        try:
                            result = self._process_conversation_batch(batch)
                            self._merge_results([result])
                            pbar.update(1)
                            
                            # Clear device memory periodically
                            if i % 10 == 0 and self.device.type == 'mps':
                                torch.mps.empty_cache()
                                
                        except Exception as batch_error:
                            logger.error(f"Error processing batch {i}: {batch_error}")
                            continue
                            
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error processing conversations: {str(e)}")
            raise

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
                self._merge_results([(local_processed_data, local_temporal_data)])
                
        except Exception as e:
            logger.error(f"Error in single conversation processing: {str(e)}")

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

    def _process_conversation_batch(self, conversations):
        """Process a batch of conversations with optimized memory management"""
        local_processed_data = defaultdict(list)
        local_temporal_data = defaultdict(dict)
        
        batch_texts = []
        batch_dates = []
        batch_metadata = []
        
        for conv in conversations:
            try:
                mapping = conv.get('mapping', {})
                if not mapping and isinstance(conv, dict):
                    mapping = {'root': conv}
                    
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
                        logger.warning(f"Error processing node in batch: {node_error}")
                        continue
                        
            except Exception as conv_error:
                logger.warning(f"Error processing conversation in batch: {conv_error}")
                continue

        if batch_texts:
            try:
                self._analyze_content_batch(batch_texts, batch_dates, batch_metadata,
                                         local_processed_data, local_temporal_data)
            except Exception as e:
                logger.error(f"Error in batch analysis: {str(e)}")
                
        return local_processed_data, local_temporal_data

    def _analyze_content_batch(self, texts, dates, metadata, local_processed_data, local_temporal_data):
        """Analyze a batch of content with optimized device usage"""
        try:
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            
            for idx, (text, date, meta) in enumerate(zip(texts, dates, metadata)):
                try:
                    # Tokenize with proper handling of device placement
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_TOKENS,
                        padding=True
                    )
                    
                    # Move to appropriate device
                    if self.device.type == 'mps':
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate embeddings
                    with torch.no_grad():
                        try:
                            outputs = model(**inputs)
                            embedding = outputs.last_hidden_state.mean(dim=1)
                            
                            # Move embedding to CPU immediately
                            if self.device.type == 'mps':
                                embedding = embedding.cpu()
                                torch.mps.empty_cache()
                        except RuntimeError as e:
                            logger.warning(f"Runtime error in embedding generation: {e}")
                            continue
                    
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

                    # Extract and process projects
                    projects = self._extract_projects(text)
                    if projects:
                        for project in projects:
                            project['embedding'] = embedding[0].numpy()
                            project['date'] = date
                            project['metadata'] = meta
                            local_processed_data['projects'].append(project)

                    # Sentiment analysis
                    try:
                        sentiment = sia.polarity_scores(text)
                        if sentiment['compound'] > 0.2:
                            achievement = {
                                'content': summary,
                                'full_content': text,
                                'sentiment': sentiment['compound'],
                                'date': date,
                                'metadata': meta,
                                'embedding': embedding[0].numpy()
                            }
                            local_processed_data['achievements'].append(achievement)
                    except Exception as sent_error:
                        logger.warning(f"Error in sentiment analysis: {sent_error}")

                    # Extract entities
                    entities = self._perform_ner(text)
                    if entities:
                        local_processed_data['entities'].extend(entities)

                except Exception as text_error:
                    logger.error(f"Error processing text {idx}: {text_error}")
                    continue

        except Exception as e:
            logger.error(f"Critical error in batch processing: {str(e)}")
            raise
        finally:
            if self.device.type == 'mps':
                torch.mps.empty_cache()

    def _extract_skills(self, text):
        """Extract skills with context awareness and validation"""
        try:
            skills = set()
            text_lower = text.lower()
            doc = nlp(text_lower)
            
            for skill in self.ai_skills:
                skill_lower = skill.lower()
                if skill_lower in text_lower:
                    # Validate skill mention with context
                    skill_doc = nlp(skill_lower)
                    for chunk in doc.noun_chunks:
                        if skill_lower in chunk.text:
                            skills.add(skill)
                            break
                            
            return list(skills)
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return []

    def _extract_projects(self, text):
        """Extract project descriptions with enhanced context analysis"""
        try:
            doc = nlp(text)
            projects = []
            
            project_keywords = ['project', 'developed', 'created', 'built', 'implemented', 
                              'designed', 'architected', 'engineered', 'deployed']
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if any(keyword in sent_text for keyword in project_keywords):
                    context = self._get_sentence_context(doc, sent)
                    
                    # Extract technical details and technologies
                    tech_entities = [ent.text for ent in sent.ents 
                                   if ent.label_ in ['ORG', 'PRODUCT', 'GPE']]
                    
                    projects.append({
                        'description': sent.text.strip(),
                        'context': context,
                        'technologies': tech_entities,
                        'entities': [(ent.text, ent.label_) for ent in sent.ents]
                    })
                    
            return projects
        except Exception as e:
            logger.error(f"Error extracting projects: {str(e)}")
            return []

    def _perform_ner(self, text):
        """Perform Named Entity Recognition with filtering"""
        try:
            doc = nlp(text[:MAX_TEXT_LENGTH])  # Limit text length for NER
            
            # Filter relevant entity types
            relevant_types = {'ORG', 'PRODUCT', 'GPE', 'PERSON', 'DATE', 'WORK_OF_ART'}
            
            return [(ent.text, ent.label_) for ent in doc.ents 
                    if ent.label_ in relevant_types]
        except Exception as e:
            logger.error(f"Error in NER: {str(e)}")
            return []

    def _get_sentence_context(self, doc, sent):
        """Get surrounding context for a sentence with error handling"""
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
                # Merge processed data with duplicate checking
                for key, values in local_processed_data.items():
                    if key == 'projects':
                        # Detect and remove duplicate projects
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
                            
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            raise

def main():
    try:
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

        # Initialize and run the resume generator
        generator = AIResumeGenerator()
        
        logger.info("Processing conversations...")
        generator.process_conversations(input_file)
        
        logger.info("Generating resume components...")
        generator.generate_resume(output_dir)
        
        logger.info(f"Resume generation complete. Check '{output_dir}' for all components.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")
        logger.info("Check ai_resume_generator.log for details.")
    finally:
        # Ensure proper cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    # Download required NLTK data if not already present
    try:
        nltk.download([
            'punkt',
            'stopwords',
            'wordnet',
            'vader_lexicon'
        ], quiet=True)
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        
    # Initialize spaCy if not already loaded
    try:
        if not spacy.util.is_package('en_core_web_sm'):
            spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        
    freeze_support()
    main()
