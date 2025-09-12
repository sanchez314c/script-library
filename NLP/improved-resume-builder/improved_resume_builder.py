#!/usr/bin/env python3
"""
Improved ChatGPT/Claude Chat Resume Builder
-------------------------------------------
This script analyzes your chat conversation history with AI assistants like ChatGPT or Claude,
extracts skills, achievements, and career-related information, and generates a comprehensive
resume and analytics dashboard.

The script has been optimized for reliability and performance, with improved error handling
and device compatibility (supports Apple Silicon MPS, CUDA, and CPU).
"""

import logging
import os
import glob
import json
import spacy
import tkinter as tk
from tkinter import filedialog
from collections import Counter, defaultdict
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import re
from bertopic import BERTopic
import networkx as nx
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from io import BytesIO
import numpy as np
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_resume_generator.log"),
        logging.StreamHandler()
    ]
)

class DeviceManager:
    """Manages device selection (MPS, CUDA, or CPU) for model inference."""
    
    def __init__(self):
        self.device = self._setup_device()
        logging.info(f"Using device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            logging.info("MPS (Apple Silicon) acceleration available")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logging.info("CUDA acceleration available")
            return torch.device("cuda")
        logging.info("Using CPU for processing")
        return torch.device("cpu")

class ModelManager:
    """Loads and manages NLP models with appropriate device settings."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.nlp = None
        self.sentence_transformer = None
        self.sentiment_analyzer = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all required NLP models with appropriate error handling."""
        try:
            # Load SpaCy model
            self._load_spacy()
            
            # Load sentence transformer model
            self._load_sentence_transformer()
            
            # Load sentiment analyzer
            self._load_sentiment_analyzer()
            
            logging.info("All models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def _load_spacy(self):
        """Load SpaCy model with fallback to simpler model if needed."""
        try:
            logging.info("Loading SpaCy model: en_core_web_trf")
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            logging.warning("Failed to load en_core_web_trf, attempting to download it...")
            try:
                os.system("python -m spacy download en_core_web_trf")
                self.nlp = spacy.load("en_core_web_trf")
            except Exception as e:
                logging.warning(f"Failed to download en_core_web_trf: {e}. Falling back to en_core_web_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logging.warning("Downloading en_core_web_sm...")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model with device optimization."""
        try:
            logging.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            if self.device.type in ["mps", "cuda"]:
                self.sentence_transformer = self.sentence_transformer.to(self.device)
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer: {e}")
            raise
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analyzer with appropriate device settings."""
        try:
            logging.info("Loading sentiment analysis model")
            device_id = -1 if self.device.type == "cpu" else 0
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device_id
            )
        except Exception as e:
            logging.error(f"Error loading sentiment analyzer: {e}")
            raise

class ChatDataProcessor:
    """Process and analyze chat data to extract relevant information."""
    
    def __init__(self, device_manager, model_manager):
        self.device = device_manager.device
        self.models = model_manager
        
        # Regular expressions for identifying skills
        self.skills_pattern = re.compile(
            r'\b(python|javascript|typescript|react|vue|angular|node\.js|sql|mongodb|aws|'
            r'azure|docker|kubernetes|ci/cd|git|agile|scrum|machine learning|deep learning|'
            r'nlp|computer vision|data science|api|rest|graphql|testing|devops|security|'
            r'tensorflow|pytorch|keras|pandas|numpy|scipy|scikit-learn|flask|django|cloud|'
            r'infrastructure|frontend|backend|fullstack|mobile|ios|android|swift|kotlin|'
            r'java|c\+\+|c#|go|rust|php|ruby|scala|hadoop|spark|tableau|powerbi|excel|'
            r'statistics|mathematics|algorithms|blockchain|crypto|leadership|management)\b',
            re.IGNORECASE
        )
        
        # Embedding cache to avoid recomputing the same embeddings
        self.embedding_cache = {}
    
    def load_and_validate_data(self, folder_path):
        """Load JSON files from a directory and validate their format."""
        all_files = glob.glob(os.path.join(folder_path, "*.json"))
        if not all_files:
            logging.error(f"No JSON files found in {folder_path}")
            return []
        
        data = []
        for file in all_files:
            try:
                logging.info(f"Loading file: {file}")
                with open(file, "r", encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(file_data, list):
                    # Check if this is likely a ChatGPT/Claude conversation format
                    if len(file_data) > 0 and isinstance(file_data[0], dict):
                        # Check for various possible structures
                        if all(key in file_data[0] for key in ['timestamp', 'user_input', 'ai_response']):
                            # Standard format we expect
                            valid_data = self._validate_standard_format(file_data)
                            data.extend(valid_data)
                        elif 'messages' in file_data[0]:
                            # ChatGPT JSON export format
                            logging.info("Detected ChatGPT export format")
                            converted_data = self._convert_chatgpt_format(file_data)
                            data.extend(converted_data)
                        elif 'role' in file_data[0] and 'content' in file_data[0]:
                            # Message format with role and content
                            logging.info("Detected role-content format")
                            converted_data = self._convert_role_content_format(file_data)
                            data.extend(converted_data)
                        else:
                            # Try to adapt unknown format
                            logging.warning(f"Unknown list format in {file}, attempting to adapt")
                            converted_data = self._adapt_unknown_format(file_data)
                            if converted_data:
                                data.extend(converted_data)
                # Handle dictionary format (some ChatGPT exports are like this)
                elif isinstance(file_data, dict):
                    logging.info("Detected dictionary format")
                    converted_data = self._convert_dict_format(file_data)
                    if converted_data:
                        data.extend(converted_data)
                else:
                    logging.warning(f"Unsupported JSON structure in {file}, skipping")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in file {file}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error reading file {file}: {e}")
        
        logging.info(f"Loaded {len(data)} valid conversation entries")
        return data
    
    def _validate_standard_format(self, data):
        """Validate and clean data in our standard format."""
        valid_data = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
                
            # Ensure required fields exist
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'user_input' not in entry:
                entry['user_input'] = 'No user input provided'
            if 'ai_response' not in entry:
                entry['ai_response'] = 'No AI response available'
            
            valid_data.append(entry)
        
        return valid_data
    
    def _convert_chatgpt_format(self, data):
        """Convert ChatGPT JSON export format to our standard format."""
        converted_data = []
        
        for conversation in data:
            if 'messages' not in conversation:
                continue
            
            messages = conversation.get('messages', [])
            timestamp = conversation.get('create_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            for i in range(1, len(messages), 2):
                user_msg = messages[i-1] if i-1 < len(messages) else {'content': ''}
                ai_msg = messages[i] if i < len(messages) else {'content': ''}
                
                user_content = user_msg.get('content', '')
                ai_content = ai_msg.get('content', '')
                
                converted_data.append({
                    'timestamp': timestamp,
                    'user_input': user_content,
                    'ai_response': ai_content
                })
        
        return converted_data
    
    def _convert_role_content_format(self, data):
        """Convert role-content format to our standard format."""
        converted_data = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        user_content = ''
        for i, message in enumerate(data):
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                user_content = content
            elif role in ['assistant', 'system'] and user_content:
                # Only add an entry when we have both user and assistant content
                converted_data.append({
                    'timestamp': message.get('timestamp', timestamp),
                    'user_input': user_content,
                    'ai_response': content
                })
                user_content = ''  # Reset for next pair
        
        return converted_data
    
    def _convert_dict_format(self, data):
        """Convert dictionary format to our standard format."""
        converted_data = []
        
        # Handle ChatGPT-style dictionary with conversations
        if 'conversations' in data:
            conversations = data.get('conversations', [])
            for conv in conversations:
                messages = conv.get('messages', [])
                timestamp = conv.get('create_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                for i in range(1, len(messages), 2):
                    if i >= len(messages):
                        break
                    
                    user_msg = messages[i-1]
                    ai_msg = messages[i]
                    
                    converted_data.append({
                        'timestamp': timestamp,
                        'user_input': user_msg.get('content', ''),
                        'ai_response': ai_msg.get('content', '')
                    })
        # Try other common dictionary formats
        elif 'messages' in data:
            messages = data.get('messages', [])
            timestamp = data.get('create_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            for i in range(1, len(messages), 2):
                if i >= len(messages):
                    break
                
                user_msg = messages[i-1]
                ai_msg = messages[i]
                
                converted_data.append({
                    'timestamp': timestamp,
                    'user_input': user_msg.get('content', ''),
                    'ai_response': ai_msg.get('content', '')
                })
        
        return converted_data
    
    def _adapt_unknown_format(self, data):
        """Try to adapt unknown list format to our standard format."""
        # This is a fallback for unknown formats
        converted_data = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            # Look for fields that might contain user/AI content
            user_input = ''
            ai_response = ''
            
            # Try different field names that might contain user input
            for field in ['user', 'human', 'question', 'input', 'query', 'user_input', 'user_message']:
                if field in item and item[field]:
                    user_input = item[field]
                    break
            
            # Try different field names that might contain AI response
            for field in ['ai', 'assistant', 'bot', 'response', 'answer', 'ai_response', 'output', 'reply']:
                if field in item and item[field]:
                    ai_response = item[field]
                    break
            
            # Check for timestamp fields
            item_timestamp = timestamp
            for field in ['timestamp', 'time', 'date', 'created_at', 'updated_at']:
                if field in item and item[field]:
                    item_timestamp = item[field]
                    break
            
            if user_input or ai_response:
                converted_data.append({
                    'timestamp': item_timestamp,
                    'user_input': user_input,
                    'ai_response': ai_response
                })
        
        return converted_data
    
    def preprocess_text(self, text):
        """Preprocess text using SpaCy for lemmatization and filtering."""
        try:
            doc = self.models.nlp(text)
            return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])
        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            return ""
    
    def preprocess_all_texts(self, data):
        """Preprocess all texts in parallel."""
        all_texts = []
        for item in data:
            all_texts.append(item['user_input'])
            all_texts.append(item['ai_response'])
        
        if not all_texts:
            logging.error("No texts to process")
            return []
        
        max_workers = min(os.cpu_count() or 1, len(all_texts))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.preprocess_text, all_texts))
        
        return [r for r in results if r]  # Filter out empty results
    
    def process_text(self, doc):
        """Process text document to extract tokens, entities, dependencies, and embeddings."""
        try:
            tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
            
            text = " ".join(tokens)
            if not text.strip():
                return [], [], [], 0.0
                
            embedding = self.get_embedding(text)
            embedding_value = embedding.mean().item() if embedding is not None else 0.0
            
            return tokens, entities, dependencies, embedding_value
        except Exception as e:
            logging.error(f"Error in process_text: {e}")
            return [], [], [], 0.0
    
    def get_embedding(self, text):
        """Get embedding vector for text with caching."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.models.sentence_transformer.encode(text, convert_to_tensor=True)
            if self.device.type in ["mps", "cuda"]:
                embedding = embedding.to(self.device)
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    def extract_entities_and_tokens(self, preprocessed_data):
        """Extract entities and tokens from preprocessed text in parallel."""
        if not preprocessed_data:
            logging.error("No preprocessed data to analyze")
            return [], [], [], []
        
        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
                results = list(executor.map(self.process_text, self.models.nlp.pipe(preprocessed_data, batch_size=100)))
            
            tokens, entities, dependencies, embedding_values = zip(*results)
            
            logging.info(f"Extracted tokens, entities, and dependencies from {len(preprocessed_data)} texts")
            return tokens, entities, dependencies, embedding_values
        except Exception as e:
            logging.error(f"Error in parallel entity extraction: {e}")
            return [], [], [], []
    
    def extract_skills(self, data):
        """Extract skills from conversation data."""
        skills_counter = Counter()
        
        # First pass: use regex pattern to extract common skills
        for item in data:
            user_text = item.get('user_input', '')
            ai_text = item.get('ai_response', '')
            
            for text in [user_text, ai_text]:
                found_skills = self.skills_pattern.findall(text.lower())
                skills_counter.update(found_skills)
        
        # Second pass: use entity recognition to find additional skills
        for item in data:
            user_text = item.get('user_input', '')
            ai_text = item.get('ai_response', '')
            
            for text in [user_text, ai_text]:
                doc = self.models.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'LANGUAGE']:
                        skills_counter[ent.text.lower()] += 1
        
        logging.info(f"Extracted {len(skills_counter)} unique skills")
        return dict(skills_counter)
    
    def extract_achievements(self, data):
        """Extract achievements from conversation data."""
        achievements = []
        
        # Patterns that might indicate achievements
        achievement_patterns = [
            r'accomplish(?:ed|ing|ment)?',
            r'achiev(?:ed|ing|ement)?',
            r'implement(?:ed|ing)?',
            r'develop(?:ed|ing)?',
            r'creat(?:ed|ing)?',
            r'build(?:ed|ing)?',
            r'design(?:ed|ing)?',
            r'lead(?:ing)?',
            r'manag(?:ed|ing)?',
            r'optimi(?:zed|zing)',
            r'improv(?:ed|ing)',
            r'reduc(?:ed|ing)',
            r'increas(?:ed|ing)',
            r'succeed(?:ed|ing)?',
            r'launch(?:ed|ing)?'
        ]
        
        pattern = re.compile('|'.join(achievement_patterns), re.IGNORECASE)
        
        for item in data:
            for field in ['user_input', 'ai_response']:
                content = item.get(field, '')
                if not content:
                    continue
                
                if pattern.search(content):
                    # Check for positive sentiment
                    try:
                        sentiment = self.models.sentiment_analyzer(content[:512])[0]  # Limit length for performance
                        if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.6:
                            date_str = item.get('timestamp', '')
                            try:
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                date_val = date_obj.isoformat()
                            except:
                                date_val = date_str
                            
                            achievements.append({
                                'description': content,
                                'date': date_val,
                                'sentiment_score': sentiment['score']
                            })
                    except Exception as e:
                        logging.warning(f"Error analyzing sentiment: {e}")
        
        logging.info(f"Extracted {len(achievements)} achievements")
        return sorted(achievements, key=lambda x: x['sentiment_score'], reverse=True)
    
    def extract_timeline(self, data):
        """Extract timeline events from conversation data."""
        timeline = []
        
        for item in data:
            timestamp = item.get('timestamp', '')
            if not timestamp:
                continue
            
            try:
                date_obj = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                formatted_date = date_obj.strftime('%Y-%m-%d')
            except:
                formatted_date = timestamp
            
            user_input = item.get('user_input', '')[:50]
            if user_input:
                event = f"Conversation about: {user_input}..."
                timeline.append((formatted_date, event))
        
        timeline.sort(key=lambda x: x[0])
        logging.info(f"Extracted {len(timeline)} timeline events")
        return timeline
    
    def get_top_skills(self, tokens):
        """Extract and rank skills from tokens."""
        if not tokens:
            logging.error("No tokens provided for skill extraction")
            return []
        
        try:
            all_tokens = [token for sublist in tokens for token in sublist]
            skill_counts = Counter(all_tokens)
            
            # Common stop words and non-skill words to filter out
            stop_words = set([
                "the", "a", "an", "in", "to", "of", "and", "for", "is", "are", "be", "with", 
                "that", "this", "have", "has", "had", "do", "does", "did", "can", "could", "will", 
                "would", "should", "may", "might", "must", "not", "it", "i", "you", "he", "she", 
                "they", "we", "us", "them", "our", "your", "their", "my", "me", "myself", "yourself", 
                "himself", "herself", "itself", "themself", "ourselves", "yourselves", "themselves", 
                "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", 
                "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", 
                "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
                "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", 
                "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", 
                "shan", "shouldn", "wasn", "weren", "won", "wouldn", "go", "going", "went", "gone"
            ])
            
            filtered_skills = [(skill, count) for skill, count in skill_counts.items() 
                             if skill not in stop_words and len(skill) > 2]
            return sorted(filtered_skills, key=lambda item: item[1], reverse=True)
        except Exception as e:
            logging.error(f"Error extracting skills: {e}")
            return []
    
    def match_skills_to_jobs(self, skills, job_descriptions):
        """Match extracted skills to job descriptions using semantic similarity."""
        if not skills or not job_descriptions:
            logging.error("Missing skills or job descriptions")
            return []
        
        try:
            skill_text = ' '.join([skill[0] for skill in skills[:30]])  # Use top 30 skills
            
            if not skill_text.strip():
                logging.error("Empty skill text generated")
                return []
            
            skill_embedding = self.get_embedding(skill_text)
            job_embeddings = [self.get_embedding(job) for job in job_descriptions]
            
            if skill_embedding is None or not job_embeddings:
                logging.error("Failed to generate embeddings")
                return []
            
            # Calculate similarity between skill embedding and job embeddings
            similarity_scores = []
            for i, job_embedding in enumerate(job_embeddings):
                if job_embedding is not None:
                    similarity = util.pytorch_cos_sim(skill_embedding, job_embedding)
                    similarity_scores.append((job_descriptions[i], similarity.item()))
            
            # Sort by similarity score
            job_matches = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            return job_matches
        except Exception as e:
            logging.error(f"Error matching skills to jobs: {e}")
            return []
    
    def cluster_and_visualize_skills(self, skills):
        """Cluster skills and generate a visualization."""
        if not skills:
            logging.error("No skills provided for visualization")
            return
        
        try:
            plt.clf()  # Clear any existing plots
            G = nx.Graph()
            
            # Add nodes with skill counts
            for skill, count in skills:
                G.add_node(skill, size=count)
            
            # Add edges between similar skills
            for i, (skill1, _) in enumerate(skills):
                for skill2, _ in skills[i+1:i+50]:  # Check only the next 50 skills for efficiency
                    if fuzz.ratio(skill1, skill2) > 70:
                        G.add_edge(skill1, skill2)
            
            if not G.nodes():
                logging.error("No nodes in graph")
                return
                
            pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility
            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, with_labels=True, 
                   node_size=[G.nodes[node]['size']*5 for node in G.nodes()],
                   node_color='lightblue',
                   font_size=8,
                   edge_color='gray',
                   width=0.5)
            plt.title("Skill Cluster Visualization")
            plt.savefig("skill_cluster.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Skill cluster visualization saved as skill_cluster.png")
        except Exception as e:
            logging.error(f"Error in skill visualization: {e}")
    
    def perform_topic_modeling(self, preprocessed_data):
        """Perform topic modeling on preprocessed text data."""
        if not preprocessed_data:
            logging.error("No data provided for topic modeling")
            return pd.DataFrame(), []
        
        try:
            # Only use preprocessing if there are enough documents
            if len(preprocessed_data) >= 10:
                logging.info("Performing topic modeling...")
                topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
                topics, probs = topic_model.fit_transform(preprocessed_data)
                return topic_model.get_topic_info(), topics
            else:
                logging.warning("Not enough data for meaningful topic modeling")
                return pd.DataFrame(columns=['Topic', 'Count', 'Name']), []
        except Exception as e:
            logging.error(f"Error in topic modeling: {e}")
            return pd.DataFrame(columns=['Topic', 'Count', 'Name']), []
    
    def analyze_skill_evolution(self, data, tokens_by_message):
        """Analyze how skills evolve over time in the conversation."""
        if not data or not tokens_by_message:
            logging.error("Missing data for skill evolution analysis")
            return pd.DataFrame()
        
        try:
            skill_evolution = defaultdict(lambda: defaultdict(int))
            
            for item, tokens in zip(data, tokens_by_message):
                # Try different timestamp formats
                try:
                    timestamp = item.get('timestamp', '')
                    date_obj = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    date_key = date_obj.strftime('%Y-%m')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
                        date_key = date_obj.strftime('%Y-%m')
                    except:
                        # Default to the first date entry if timestamp is invalid
                        date_key = "Unknown"
                
                for token in tokens:
                    skill_evolution[date_key][token] += 1
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(skill_evolution).T.fillna(0)
            
            # Only keep columns that have more than one non-zero value
            df = df.loc[:, (df != 0).sum() > 1]
            
            if df.empty:
                logging.warning("No skill evolution data found")
                return pd.DataFrame()
            
            # Plot evolution
            plt.figure(figsize=(15, 10))
            df.plot(figsize=(15, 10))
            plt.title("Skill Evolution Over Time")
            plt.xlabel("Date")
            plt.ylabel("Frequency")
            plt.savefig("skill_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Skill evolution visualization saved as skill_evolution.png")
            return df
        except Exception as e:
            logging.error(f"Error in skill evolution analysis: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, data):
        """Analyze sentiment trends in conversation data."""
        if not data:
            logging.error("No data provided for sentiment analysis")
            return []
        
        sentiments = []
        for item in data:
            try:
                # Analyze both user and AI text
                for field in ['user_input', 'ai_response']:
                    text = item.get(field, '')
                    if text:
                        # Truncate to 512 chars to avoid timeout
                        sentiment = self.models.sentiment_analyzer(text[:512])[0]
                        sentiments.append((
                            item.get('timestamp', 'Unknown'),
                            field,  # Add field type to distinguish user vs AI
                            sentiment['label'],
                            sentiment['score']
                        ))
            except Exception as e:
                logging.warning(f"Error analyzing sentiment: {e}")
                continue
        
        logging.info(f"Analyzed sentiment for {len(sentiments)} messages")
        return sentiments

class ResumeGenerator:
    """Generates PDF resume and dashboard based on extracted information."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the resume."""
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
    
    def generate_resume(self, name, job_matches, skills, timeline, topics, achievements, sentiment_data):
        """Generate a PDF resume with the extracted information."""
        output_path = f"{name.replace(' ', '_')}_AI_Resume.pdf"
        
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Add header section
            self._add_header(story, name)
            
            # Add summary section based on top skills
            self._add_summary(story, skills)
            
            # Add skills section
            self._add_skills_section(story, skills)
            
            # Add job matches section
            if job_matches:
                self._add_job_matches_section(story, job_matches)
            
            # Add achievements section
            if achievements:
                self._add_achievements_section(story, achievements)
            
            # Add timeline section
            if timeline:
                self._add_timeline_section(story, timeline)
            
            # Add topics section if topics are available
            if not topics.empty:
                self._add_topics_section(story, topics)
            
            # Add visualizations
            if os.path.exists("skill_cluster.png"):
                self._add_visualization(story, "skill_cluster.png", "Skill Relationships")
            
            if os.path.exists("skill_evolution.png"):
                self._add_visualization(story, "skill_evolution.png", "Skill Development Over Time")
            
            doc.build(story)
            logging.info(f"Resume generated successfully: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error generating resume: {e}")
            return None
    
    def _add_header(self, story, name):
        """Add header section to the resume."""
        story.append(Paragraph(f"{name}'s Resume", self.styles['Title']))
        story.append(Paragraph('Generated from AI Assistant Conversations', self.styles['SubHeader']))
        story.append(Spacer(1, 20))
    
    def _add_summary(self, story, skills):
        """Add professional summary section based on top skills."""
        if skills:
            top_skills = sorted(skills, key=lambda x: x[1], reverse=True)[:5]
            skill_text = ', '.join(skill for skill, _ in top_skills)
        else:
            skill_text = 'various technologies'
        
        summary = (
            f"Professional with expertise in {skill_text}. "
            "Demonstrated success in developing solutions and tackling complex problems "
            "through AI-assisted conversation history."
        )
        
        story.append(Paragraph('Professional Summary', self.styles['SectionHeader']))
        story.append(Paragraph(summary, self.styles['Normal']))
        story.append(Spacer(1, 20))
    
    def _add_skills_section(self, story, skills):
        """Add skills section to the resume."""
        story.append(Paragraph('Technical Skills', self.styles['SectionHeader']))
        
        if not skills:
            story.append(Paragraph("No technical skills identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Take top 30 skills and organize them by category
        top_skills = skills[:30]
        
        # Create manual skill categories
        categories = {
            'Programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'go', 'ruby', 'swift', 'kotlin', 'scala', 'rust'],
            'Data Science': ['machine learning', 'deep learning', 'nlp', 'data', 'statistics', 'analytics', 'model', 'algorithm', 'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy'],
            'Web & Mobile': ['web', 'react', 'angular', 'vue', 'frontend', 'backend', 'fullstack', 'mobile', 'android', 'ios', 'html', 'css', 'node', 'api', 'rest', 'graphql'],
            'DevOps & Cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'ci/cd', 'git', 'devops', 'deployment', 'infrastructure']
        }
        
        categorized_skills = {category: [] for category in categories}
        other_skills = []
        
        for skill, count in top_skills:
            categorized = False
            for category, keywords in categories.items():
                if any(keyword in skill.lower() for keyword in keywords):
                    categorized_skills[category].append(f"{skill} ({count})")
                    categorized = True
                    break
            
            if not categorized:
                other_skills.append(f"{skill} ({count})")
        
        # Add "Other" category if needed
        if other_skills:
            categorized_skills['Other'] = other_skills
        
        # Generate table data
        table_data = [list(categorized_skills.keys())]
        
        # Find maximum length of any skill list
        max_rows = max(len(skills) for skills in categorized_skills.values())
        
        # Create table rows
        for i in range(max_rows):
            row = []
            for category in categorized_skills:
                if i < len(categorized_skills[category]):
                    row.append(categorized_skills[category][i])
                else:
                    row.append('')
            table_data.append(row)
        
        # Create and style the table
        col_width = 120
        table = Table(table_data, colWidths=[col_width] * len(categorized_skills))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    def _add_job_matches_section(self, story, job_matches):
        """Add job matches section to the resume."""
        story.append(Paragraph('Top Job Matches', self.styles['SectionHeader']))
        
        if not job_matches:
            story.append(Paragraph("No job matches identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Create table with job matches and similarity scores
        job_data = [['Job Title', 'Match Score']]
        for job, score in job_matches[:5]:
            job_data.append([job, f"{score:.2f}"])
        
        table = Table(job_data, colWidths=[400, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    def _add_achievements_section(self, story, achievements):
        """Add achievements section to the resume."""
        story.append(Paragraph('Key Achievements', self.styles['SectionHeader']))
        
        if not achievements:
            story.append(Paragraph("No notable achievements found.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        for achievement in achievements[:5]:  # Show top 5 achievements
            description = achievement['description']
            # Trim description to reasonable length
            if len(description) > 200:
                description = description[:197] + '...'
            
            date_str = achievement.get('date', '')
            date_formatted = date_str
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str)
                    date_formatted = date_obj.strftime('%b %Y')
                except:
                    pass
            
            entry = f"• {date_formatted}: {description}"
            story.append(Paragraph(entry, self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 10))
    
    def _add_timeline_section(self, story, timeline):
        """Add timeline section to the resume."""
        story.append(Paragraph('Conversation Timeline', self.styles['SectionHeader']))
        
        if not timeline:
            story.append(Paragraph("No timeline events identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Create table with timeline events
        timeline_data = [['Date', 'Event']]
        for date, event in timeline[:10]:  # Show top 10 timeline events
            timeline_data.append([date, event])
        
        table = Table(timeline_data, colWidths=[100, 400])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    def _add_topics_section(self, story, topics):
        """Add topics section to the resume."""
        story.append(Paragraph('Key Topics', self.styles['SectionHeader']))
        
        if topics.empty:
            story.append(Paragraph("No topics identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Filter for meaningful topics (not noise)
        filtered_topics = topics[topics['Topic'] != -1]
        if filtered_topics.empty:
            story.append(Paragraph("No meaningful topics identified.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Create table with topics
        topic_data = [['Topic', 'Description', 'Count']]
        for _, row in filtered_topics.head(10).iterrows():
            topic_data.append([f"Topic {row['Topic']}", row['Name'], str(row['Count'])])
        
        table = Table(topic_data, colWidths=[80, 320, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
    
    def _add_visualization(self, story, image_path, title):
        """Add visualization image to the resume."""
        if not os.path.exists(image_path):
            return
        
        story.append(Paragraph(title, self.styles['SectionHeader']))
        story.append(Image(image_path, width=480, height=320))
        story.append(Spacer(1, 20))

class DashboardGenerator:
    """Generates an interactive dashboard with extracted information."""
    
    def __init__(self):
        self.app = None
    
    def create_dashboard(self, skills, job_matches, skill_evolution_df, topic_info, achievements, sentiments):
        """Create an interactive Dash application with visualizations."""
        try:
            self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
            
            self.app.layout = html.Div([
                html.H1("AI Career Analysis Dashboard"),
                dcc.Tabs([
                    dcc.Tab(label='Skills', children=self._create_skills_tab(skills)),
                    dcc.Tab(label='Job Matches', children=self._create_job_matches_tab(job_matches)),
                    dcc.Tab(label='Skill Evolution', children=self._create_skill_evolution_tab(skill_evolution_df)),
                    dcc.Tab(label='Topics', children=self._create_topics_tab(topic_info)),
                    dcc.Tab(label='Achievements', children=self._create_achievements_tab(achievements)),
                    dcc.Tab(label='Sentiment', children=self._create_sentiment_tab(sentiments))
                ])
            ])
            
            logging.info("Dashboard created successfully")
            return self.app
        except Exception as e:
            logging.error(f"Error creating dashboard: {e}")
            return None
    
    def _create_skills_tab(self, skills):
        """Create skills visualization tab."""
        if not skills:
            return [html.Div("No skills data available")]
        
        skill_data = skills[:20]  # Use top 20 skills
        
        fig = px.bar(
            skill_data, 
            x=[s[0] for s in skill_data], 
            y=[s[1] for s in skill_data],
            labels={'x': 'Skill', 'y': 'Frequency'},
            title="Top 20 Skills"
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        return [dcc.Graph(id='skill-graph', figure=fig)]
    
    def _create_job_matches_tab(self, job_matches):
        """Create job matches visualization tab."""
        if not job_matches:
            return [html.Div("No job match data available")]
        
        fig = px.bar(
            job_matches, 
            x=[j[0] for j in job_matches], 
            y=[j[1] for j in job_matches],
            labels={'x': 'Job', 'y': 'Match Score'},
            title="Job Match Scores"
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        return [dcc.Graph(id='job-match-graph', figure=fig)]
    
    def _create_skill_evolution_tab(self, skill_evolution_df):
        """Create skill evolution visualization tab."""
        if skill_evolution_df is None or skill_evolution_df.empty:
            return [html.Div("No skill evolution data available")]
        
        fig = px.line(
            skill_evolution_df,
            labels={'index': 'Date', 'value': 'Frequency'},
            title="Skill Evolution Over Time"
        )
        
        return [dcc.Graph(id='skill-evolution-graph', figure=fig)]
    
    def _create_topics_tab(self, topic_info):
        """Create topics visualization tab."""
        if topic_info is None or topic_info.empty:
            return [html.Div("No topic data available")]
        
        # Filter out -1 topic (noise)
        filtered_topics = topic_info[topic_info['Topic'] != -1]
        
        if filtered_topics.empty:
            return [html.Div("No meaningful topics identified")]
        
        fig = px.scatter(
            filtered_topics,
            x='Count',
            y='Name',
            size='Count',
            hover_data=['Topic'],
            title="Topic Distribution"
        )
        
        return [dcc.Graph(id='topic-graph', figure=fig)]
    
    def _create_achievements_tab(self, achievements):
        """Create achievements visualization tab."""
        if not achievements:
            return [html.Div("No achievements data available")]
        
        # Create a table of achievements
        achievement_rows = []
        for i, achievement in enumerate(achievements[:10]):  # Top 10 achievements
            description = achievement['description']
            if len(description) > 100:
                description = description[:97] + '...'
            
            achievement_rows.append(
                html.Tr([
                    html.Td(i + 1),
                    html.Td(achievement.get('date', '')),
                    html.Td(description),
                    html.Td(f"{achievement.get('sentiment_score', 0):.2f}")
                ])
            )
        
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("#"), 
                    html.Th("Date"), 
                    html.Th("Description"), 
                    html.Th("Sentiment")
                ])
            ),
            html.Tbody(achievement_rows)
        ], className='table')
        
        return [
            html.H3("Top Achievements"),
            table
        ]
    
    def _create_sentiment_tab(self, sentiments):
        """Create sentiment visualization tab."""
        if not sentiments:
            return [html.Div("No sentiment data available")]
        
        # Convert to DataFrame for easier plotting
        sentiment_df = pd.DataFrame(sentiments, columns=['timestamp', 'type', 'label', 'score'])
        
        # Create a scatter plot of sentiment over time
        fig = px.scatter(
            sentiment_df, 
            x='timestamp', 
            y='score', 
            color='label',
            symbol='type',
            labels={'timestamp': 'Time', 'score': 'Sentiment Score', 'label': 'Sentiment', 'type': 'Message Type'},
            title="Conversation Sentiment Over Time"
        )
        
        return [dcc.Graph(id='sentiment-graph', figure=fig)]
    
    def run_dashboard(self):
        """Run the dashboard application."""
        if self.app:
            try:
                self.app.run_server(debug=True)
                return True
            except Exception as e:
                logging.error(f"Error running dashboard: {e}")
                return False
        else:
            logging.error("Dashboard not created")
            return False

def select_folder():
    """Open a folder selection dialog."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing JSON Files",
            initialdir=os.path.expanduser("~/Downloads")  # Start in Downloads folder
        )
        return folder_path
    except Exception as e:
        logging.error(f"Error in folder selection: {e}")
        return None

def main():
    try:
        print("📊 AI Resume Generator 📊")
        print("This tool analyzes your AI assistant conversations and generates a resume.")
        print("It works with various JSON formats from ChatGPT, Claude, and similar assistants.")
        
        # Initialize device and model managers
        device_manager = DeviceManager()
        model_manager = ModelManager(device_manager.device)
        
        # Initialize data processor
        processor = ChatDataProcessor(device_manager, model_manager)
        
        # Get folder path containing JSON conversation data
        folder_path = select_folder()
        if not folder_path:
            print("❌ No folder selected. Exiting.")
            return
        
        print(f"📂 Loading data from: {folder_path}")
        
        # Load and process data
        data = processor.load_and_validate_data(folder_path)
        
        if not data:
            print("❌ No valid conversation data found. Please check your JSON files.")
            return
        
        print(f"✅ Loaded {len(data)} conversation entries")
        
        # Preprocess all text data
        print("🔄 Preprocessing text data...")
        preprocessed_data = processor.preprocess_all_texts(data)
        
        if not preprocessed_data:
            print("❌ Failed to preprocess texts. Exiting.")
            return
        
        # Extract entities and tokens
        print("🔍 Extracting entities and tokens...")
        tokens, entities, dependencies, _ = processor.extract_entities_and_tokens(preprocessed_data)
        
        if not tokens:
            print("❌ No tokens extracted. Exiting.")
            return
        
        # Get top skills
        print("🏆 Identifying top skills...")
        skills = processor.get_top_skills(tokens)
        print(f"✅ Found {len(skills)} skills")
        
        # Extract additional skills using regex and NER
        print("🧩 Extracting additional skills...")
        extracted_skills = processor.extract_skills(data)
        print(f"✅ Found {len(extracted_skills)} skills using NER")
        
        # Extract achievements
        print("🌟 Identifying achievements...")
        achievements = processor.extract_achievements(data)
        print(f"✅ Found {len(achievements)} achievements")
        
        # Extract timeline
        print("📅 Creating timeline...")
        timeline = processor.extract_timeline(data)
        print(f"✅ Created timeline with {len(timeline)} events")
        
        # Define job descriptions for matching
        job_descriptions = [
            "AI Engineer with experience in machine learning and deep learning",
            "Data Scientist proficient in statistical analysis and predictive modeling",
            "NLP Specialist with expertise in text processing and language models",
            "Computer Vision Engineer skilled in image processing and object detection",
            "Software Engineer with experience in distributed systems",
            "Frontend Developer specializing in React and modern JavaScript",
            "Backend Developer with expertise in API design and databases",
            "DevOps Engineer with experience in CI/CD and cloud services",
            "Product Manager with a technical background",
            "Technical Writer specializing in documentation and tutorials"
        ]
        
        # Match skills to jobs
        print("🔄 Matching skills to potential jobs...")
        job_matches = processor.match_skills_to_jobs(skills, job_descriptions)
        print(f"✅ Analyzed compatibility with {len(job_matches)} job types")
        
        # Generate skill visualization
        print("📊 Creating skill visualizations...")
        processor.cluster_and_visualize_skills(skills[:50])  # Use top 50 skills for visualization
        
        # Perform topic modeling
        print("📚 Performing topic modeling...")
        topic_info, document_topics = processor.perform_topic_modeling(preprocessed_data)
        
        # Analyze skill evolution
        print("📈 Analyzing skill evolution...")
        skill_evolution_df = processor.analyze_skill_evolution(data, tokens)
        
        # Analyze sentiment
        print("😊 Analyzing sentiment...")
        conversation_sentiments = processor.analyze_sentiment(data)
        print(f"✅ Analyzed sentiment for {len(conversation_sentiments)} messages")
        
        # Get user name for the resume
        name = input("👤 Enter your name for the resume: ").strip()
        if not name:
            name = "User"
        
        # Generate enhanced resume
        print("📝 Generating resume...")
        resume_gen = ResumeGenerator()
        resume_path = resume_gen.generate_resume(
            name, job_matches, skills, timeline, topic_info, 
            achievements, conversation_sentiments
        )
        
        if resume_path:
            print(f"✅ Resume generated successfully: {resume_path}")
        else:
            print("❌ Failed to generate resume")
        
        # Create and run dashboard
        print("🚀 Creating interactive dashboard...")
        dashboard_gen = DashboardGenerator()
        app = dashboard_gen.create_dashboard(
            skills, job_matches, skill_evolution_df, 
            topic_info, achievements, conversation_sentiments
        )
        
        if app:
            print("✅ Dashboard created successfully")
            print("💻 Starting dashboard server (press Ctrl+C to exit)...")
            dashboard_gen.run_dashboard()
        else:
            print("❌ Failed to create dashboard")
    
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()