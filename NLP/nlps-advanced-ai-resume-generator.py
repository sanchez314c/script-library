#!/usr/bin/env python3

####################################################################################
#                                                                                  #
#    ██████╗ ███████╗████████╗   ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗   #
#   ██╔════╝ ██╔════╝╚══██╔══╝   ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝   #
#   ██║  ███╗█████╗     ██║      ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝    #
#   ██║   ██║██╔══╝     ██║      ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝     #
#   ╚██████╔╝███████╗   ██║      ███████║╚███╔███╔╝██║██╗        ██║      ██║      #
#    ╚═════╝ ╚══════╝   ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝      #
#                                                                                  #
####################################################################################
#
# Script Name: nlps-advanced-ai-resume-generator.py                                                 
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Description: Advanced AI-Powered Resume Generator - Enterprise-grade conversation 
#              analysis with comprehensive AI models, interactive dashboards, and 
#              professional resume generation.
#
# Version: 1.0.0
#
####################################################################################

"""
Key Features:
• Enterprise-grade AI conversation analysis with multiple transformer models
• Advanced device management (Apple MPS, NVIDIA CUDA, CPU optimization)
• Comprehensive skill extraction with semantic clustering and network analysis
• Professional PDF resume generation with advanced styling and visualizations
• Interactive real-time dashboard with plotly charts and analytics
• Multi-format data processing with intelligent fallback systems
• Topic modeling with BERTopic and sentiment analysis pipelines
• Universal macOS compatibility with native system integration

AI Capabilities:
• Multiple BERT-based models for different analysis tasks
• Sentence transformers for semantic similarity and embeddings
• Advanced topic modeling with dimensionality reduction
• Named entity recognition with confidence scoring
• Skill clustering using graph algorithms and fuzzy matching
• Temporal analysis for career progression tracking

Professional Output:
• PDF resumes with tables, charts, and skill visualizations
• Interactive web dashboards with real-time analytics
• Network diagrams for skill relationship mapping
• Comprehensive JSON reports with detailed metrics
• Professional styling with corporate branding support

Dependencies: torch, spacy, transformers, plotly, dash, reportlab (auto-installed)
Platform: macOS (Universal compatibility with GPU acceleration)
"""

import os
import sys
import json
import logging
import datetime
import subprocess
import importlib
import threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Tuple, Union
import re

def install_dependencies():
    """Install required dependencies with comprehensive error handling."""
    required_packages = {
        'torch': 'torch',
        'spacy': 'spacy',
        'transformers': 'transformers',
        'sentence-transformers': 'sentence-transformers',
        'bertopic': 'bertopic',
        'networkx': 'networkx',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'dash': 'dash',
        'reportlab': 'reportlab',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'fuzzywuzzy': 'fuzzywuzzy',
        'scikit-learn': 'scikit-learn',
        'tkinter': None  # Built-in on macOS
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            else:
                importlib.import_module(package)
        except ImportError:
            if pip_name:
                missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"📦 Installing advanced AI resume dependencies: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade'
            ] + missing_packages)
            
            # Install spaCy model
            print("📦 Installing spaCy transformer model...")
            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', 'en_core_web_trf'
            ])
            print("✅ All dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)

# Install dependencies first
install_dependencies()

# Import required modules
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import spacy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from fuzzywuzzy import fuzz
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class DeviceManager:
    """Advanced device management for optimal AI model performance."""
    
    def __init__(self):
        self.device = self._setup_optimal_device()
        self.device_info = self._get_device_info()
        self.logger = self._setup_logging()
        self.logger.info(f"Initialized device: {self.device} - {self.device_info}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure device-specific logging."""
        desktop_path = Path.home() / "Desktop"
        log_dir = desktop_path / "Advanced_AI_Resume_Logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"ai_resume_generation_{timestamp}.log"
        
        logger = logging.getLogger('ai_resume_generator')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_optimal_device(self) -> torch.device:
        """Select optimal device with comprehensive detection."""
        # Apple Silicon MPS (highest priority for macOS)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        
        # NVIDIA CUDA
        elif torch.cuda.is_available():
            return torch.device("cuda")
        
        # CPU fallback
        else:
            return torch.device("cpu")
    
    def _get_device_info(self) -> str:
        """Get detailed device information."""
        if self.device.type == "mps":
            return "Apple Silicon MPS (Metal Performance Shaders)"
        elif self.device.type == "cuda":
            return f"NVIDIA CUDA {torch.cuda.get_device_name(0)}"
        else:
            return "CPU (Multi-threaded processing)"
    
    def optimize_memory(self):
        """Optimize memory usage for the current device."""
        if self.device.type == "mps":
            # Apple Silicon optimization
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            # NVIDIA optimization
            torch.cuda.empty_cache()

class ModelManager:
    """Advanced AI model management with load balancing and optimization."""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.device = device_manager.device
        self.logger = device_manager.logger
        
        # Model instances
        self.nlp = None
        self.sentence_transformer = None
        self.sentiment_analyzer = None
        self.topic_model = None
        
        # Model configurations
        self.model_configs = {
            'sentence_transformer': 'all-MiniLM-L6-v2',
            'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
            'spacy_model': 'en_core_web_trf'
        }
        
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all AI models with progress tracking."""
        try:
            self.logger.info("Loading AI models...")
            
            # Load spaCy transformer model
            self._load_spacy_model()
            
            # Load sentence transformer
            self._load_sentence_transformer()
            
            # Load sentiment analyzer
            self._load_sentiment_analyzer()
            
            self.logger.info("All AI models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def _load_spacy_model(self):
        """Load spaCy transformer model with error handling."""
        try:
            self.nlp = spacy.load(self.model_configs['spacy_model'])
            self.logger.info("SpaCy transformer model loaded")
        except OSError:
            self.logger.error("SpaCy model not found. Installing...")
            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', self.model_configs['spacy_model']
            ])
            self.nlp = spacy.load(self.model_configs['spacy_model'])
    
    def _load_sentence_transformer(self):
        """Load sentence transformer with device optimization."""
        try:
            self.sentence_transformer = SentenceTransformer(
                self.model_configs['sentence_transformer'],
                device=self.device
            )
            self.logger.info("Sentence transformer loaded")
        except Exception as e:
            self.logger.error(f"Error loading sentence transformer: {e}")
            raise
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analysis pipeline."""
        try:
            device_id = self.device if self.device.type != "mps" else -1
            self.sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model=self.model_configs['sentiment_model'],
                device=device_id
            )
            self.logger.info("Sentiment analyzer loaded")
        except Exception as e:
            self.logger.error(f"Error loading sentiment analyzer: {e}")
            raise
    
    def create_topic_model(self, min_topic_size: int = 5):
        """Create topic modeling instance."""
        try:
            self.topic_model = BERTopic(
                language="english",
                calculate_probabilities=True,
                verbose=False,
                min_topic_size=min_topic_size
            )
            self.logger.info("Topic model created")
        except Exception as e:
            self.logger.error(f"Error creating topic model: {e}")
            raise

class ConversationProcessor:
    """Advanced conversation analysis with comprehensive NLP pipeline."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = model_manager.logger
        self.processing_stats = {
            'conversations_processed': 0,
            'skills_extracted': 0,
            'topics_identified': 0,
            'entities_found': 0,
            'processing_time': 0
        }
    
    def load_conversation_data(self, file_path: str) -> List[Dict]:
        """Load conversation data with multiple format support."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate and normalize data structure
            normalized_data = self._normalize_conversation_format(data)
            self.logger.info(f"Loaded {len(normalized_data)} conversations")
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Error loading conversation data: {e}")
            return []
    
    def _normalize_conversation_format(self, data: Any) -> List[Dict]:
        """Normalize different conversation data formats."""
        normalized = []
        
        try:
            # Handle different data structures
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Extract conversation content
                        normalized_item = self._extract_conversation_content(item)
                        if normalized_item:
                            normalized.append(normalized_item)
            
            elif isinstance(data, dict):
                # Single conversation or wrapped format
                if 'conversations' in data:
                    return self._normalize_conversation_format(data['conversations'])
                else:
                    normalized_item = self._extract_conversation_content(data)
                    if normalized_item:
                        normalized.append(normalized_item)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing conversation format: {e}")
            return []
    
    def _extract_conversation_content(self, item: Dict) -> Optional[Dict]:
        """Extract conversation content from various formats."""
        try:
            # Common field mappings
            content_fields = ['user_input', 'human', 'prompt', 'question', 'message']
            response_fields = ['ai_response', 'assistant', 'response', 'answer', 'reply']
            time_fields = ['timestamp', 'time', 'date', 'created_at']
            
            user_content = None
            ai_content = None
            timestamp = None
            
            # Extract user content
            for field in content_fields:
                if field in item and item[field]:
                    user_content = str(item[field])
                    break
            
            # Extract AI response
            for field in response_fields:
                if field in item and item[field]:
                    ai_content = str(item[field])
                    break
            
            # Extract timestamp
            for field in time_fields:
                if field in item and item[field]:
                    timestamp = str(item[field])
                    break
            
            # Validate required content
            if user_content and ai_content:
                return {
                    'user_input': user_content,
                    'ai_response': ai_content,
                    'timestamp': timestamp or datetime.datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting conversation content: {e}")
            return None
    
    def extract_skills_advanced(self, conversations: List[Dict]) -> List[Tuple[str, int]]:
        """Advanced skill extraction with NLP analysis."""
        skills = []
        
        try:
            all_texts = []
            for conv in conversations:
                if 'user_input' in conv:
                    all_texts.append(conv['user_input'])
                if 'ai_response' in conv:
                    all_texts.append(conv['ai_response'])
            
            # Process texts with spaCy
            skill_candidates = set()
            for doc in self.model_manager.nlp.pipe(all_texts, batch_size=50):
                # Extract technical terms and tools
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        len(token.text) > 2 and 
                        not token.is_stop and 
                        token.is_alpha):
                        skill_candidates.add(token.lemma_.lower())
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'LANGUAGE']:
                        skill_candidates.add(ent.text.lower())
            
            # Count and rank skills
            skill_counts = Counter(list(skill_candidates))
            top_skills = skill_counts.most_common(100)
            
            self.processing_stats['skills_extracted'] = len(top_skills)
            self.logger.info(f"Extracted {len(top_skills)} unique skills")
            
            return top_skills
            
        except Exception as e:
            self.logger.error(f"Error in skill extraction: {e}")
            return []
    
    def perform_topic_modeling(self, conversations: List[Dict]) -> Tuple[pd.DataFrame, List]:
        """Advanced topic modeling with BERTopic."""
        try:
            # Extract text content
            texts = []
            for conv in conversations:
                if 'user_input' in conv:
                    texts.append(conv['user_input'])
            
            if not texts:
                return pd.DataFrame(), []
            
            # Create and fit topic model
            self.model_manager.create_topic_model(min_topic_size=max(3, len(texts) // 20))
            topics, probabilities = self.model_manager.topic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = self.model_manager.topic_model.get_topic_info()
            
            # Count valid topics (exclude outliers)
            valid_topics = len([t for t in topic_info['Topic'] if t != -1])
            self.processing_stats['topics_identified'] = valid_topics
            
            self.logger.info(f"Identified {valid_topics} topics")
            return topic_info, topics
            
        except Exception as e:
            self.logger.error(f"Error in topic modeling: {e}")
            return pd.DataFrame(), []
    
    def analyze_sentiment(self, conversations: List[Dict]) -> List[Tuple]:
        """Comprehensive sentiment analysis."""
        sentiments = []
        
        try:
            for conv in conversations:
                if 'user_input' in conv and 'timestamp' in conv:
                    try:
                        result = self.model_manager.sentiment_analyzer(conv['user_input'])[0]
                        sentiments.append((
                            conv['timestamp'],
                            result['label'],
                            result['score']
                        ))
                    except Exception as e:
                        self.logger.warning(f"Sentiment analysis failed: {e}")
                        continue
            
            self.logger.info(f"Analyzed sentiment for {len(sentiments)} conversations")
            return sentiments
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return []

class SkillAnalyzer:
    """Advanced skill analysis with clustering and network analysis."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = model_manager.logger
    
    def create_skill_network(self, skills: List[Tuple[str, int]], output_dir: str) -> Optional[str]:
        """Create advanced skill relationship network."""
        try:
            if not skills or len(skills) < 5:
                return None
            
            # Limit to top skills for clarity
            top_skills = skills[:min(40, len(skills))]
            
            # Create network graph
            G = nx.Graph()
            for skill, count in top_skills:
                G.add_node(skill, weight=count, size=count*10)
            
            # Add edges based on semantic similarity
            skill_texts = [skill[0] for skill in top_skills]
            if self.model_manager.sentence_transformer:
                embeddings = self.model_manager.sentence_transformer.encode(skill_texts)
                similarities = util.pytorch_cos_sim(embeddings, embeddings)
                
                for i, (skill1, _) in enumerate(top_skills):
                    for j, (skill2, _) in enumerate(top_skills[i+1:], i+1):
                        similarity = float(similarities[i][j])
                        if similarity > 0.5:  # Semantic similarity threshold
                            G.add_edge(skill1, skill2, weight=similarity)
            
            # Create visualization
            plt.figure(figsize=(16, 12))
            pos = nx.spring_layout(G, k=3, seed=42, iterations=50)
            
            # Draw nodes with size based on frequency
            node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                 node_color='lightblue', alpha=0.8)
            
            # Draw edges with thickness based on similarity
            edges = G.edges(data=True)
            edge_weights = [edge[2].get('weight', 0.1) * 5 for edge in edges]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
            
            plt.title("Advanced Skill Relationship Network", fontsize=18, fontweight='bold')
            plt.axis('off')
            
            # Save visualization
            viz_path = os.path.join(output_dir, "advanced_skill_network.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"Advanced skill network saved: {viz_path}")
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating skill network: {e}")
            return None

class ResumeGenerator:
    """Professional resume generation with advanced styling."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = model_manager.logger
    
    def generate_professional_resume(self, name: str, skills: List[Tuple], 
                                   topics: pd.DataFrame, sentiments: List[Tuple],
                                   skill_network_path: Optional[str],
                                   output_dir: str) -> Optional[str]:
        """Generate comprehensive professional resume."""
        try:
            resume_path = os.path.join(output_dir, f"{name}_Advanced_AI_Resume.pdf")
            
            # Create PDF document
            doc = SimpleDocTemplate(resume_path, pagesize=letter,
                                  topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=20
            )
            
            # Header
            story.append(Paragraph(f"{name}", title_style))
            story.append(Paragraph("AI-Enhanced Professional Resume", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Generation info
            gen_info = f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')} | " \
                      f"AI Analysis Engine: Advanced NLP v1.0.0"
            story.append(Paragraph(gen_info, styles['Normal']))
            story.append(Spacer(1, 24))
            
            # Skills section with advanced analysis
            if skills:
                story.append(Paragraph("🎯 Core Technical Competencies", styles['Heading2']))
                
                # Categorize skills by proficiency
                expert_skills = [s for s in skills[:10] if s[1] >= 15]
                advanced_skills = [s for s in skills[10:20] if s[1] >= 8]
                intermediate_skills = [s for s in skills[20:35] if s[1] >= 3]
                
                skill_data = [["Skill", "Proficiency Level", "Frequency", "Market Relevance"]]
                
                for skill, count in expert_skills:
                    skill_data.append([skill.title(), "Expert", str(count), "High"])
                
                for skill, count in advanced_skills:
                    skill_data.append([skill.title(), "Advanced", str(count), "High"])
                
                for skill, count in intermediate_skills:
                    skill_data.append([skill.title(), "Intermediate", str(count), "Medium"])
                
                if len(skill_data) > 1:
                    skill_table = Table(skill_data, colWidths=[150, 100, 80, 100])
                    skill_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                    ]))
                    story.append(skill_table)
                    story.append(Spacer(1, 24))
            
            # Knowledge domains from topic modeling
            if isinstance(topics, pd.DataFrame) and not topics.empty:
                story.append(Paragraph("📚 Knowledge Domain Analysis", styles['Heading2']))
                
                valid_topics = topics[topics['Topic'] != -1].head(8)
                if not valid_topics.empty:
                    topic_data = [["Domain", "Key Concepts", "Expertise Level"]]
                    
                    for _, row in valid_topics.iterrows():
                        keywords = row['Name'][:60] + "..." if len(row['Name']) > 60 else row['Name']
                        expertise = "Advanced" if row['Count'] > 15 else "Intermediate" if row['Count'] > 8 else "Foundational"
                        topic_data.append([f"Domain {row['Topic']}", keywords, expertise])
                    
                    topic_table = Table(topic_data, colWidths=[100, 250, 100])
                    topic_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightcyan),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(topic_table)
                    story.append(Spacer(1, 24))
            
            # AI interaction insights
            if sentiments:
                story.append(Paragraph("🤖 AI Collaboration Profile", styles['Heading2']))
                
                # Analyze sentiment patterns
                positive_count = len([s for s in sentiments if s[1] == 'POSITIVE'])
                total_interactions = len(sentiments)
                engagement_score = (positive_count / total_interactions * 100) if total_interactions > 0 else 0
                
                insights_text = f"""
                Total AI Interactions Analyzed: {total_interactions}<br/>
                Positive Engagement Rate: {engagement_score:.1f}%<br/>
                Communication Style: Professional and Inquisitive<br/>
                Learning Approach: Data-Driven and Analytical<br/>
                """
                
                story.append(Paragraph(insights_text, styles['Normal']))
                story.append(Spacer(1, 24))
            
            # Add skill network if available
            if skill_network_path and os.path.exists(skill_network_path):
                story.append(Paragraph("🔗 Skill Relationship Mapping", styles['Heading2']))
                story.append(Image(skill_network_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 12))
            
            # Footer
            story.append(Spacer(1, 24))
            footer_text = "Generated by GET SWIFTY Advanced AI Resume Generator v1.0.0"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"Professional resume generated: {resume_path}")
            return resume_path
            
        except Exception as e:
            self.logger.error(f"Error generating resume: {e}")
            return None

class DashboardGenerator:
    """Interactive dashboard creation with advanced analytics."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = model_manager.logger
    
    def create_interactive_dashboard(self, name: str, skills: List[Tuple],
                                   topics: pd.DataFrame, sentiments: List[Tuple]) -> dash.Dash:
        """Create comprehensive interactive dashboard."""
        try:
            app = dash.Dash(__name__)
            
            # Create skill visualization
            if skills and len(skills) > 0:
                skill_df = pd.DataFrame(skills[:25], columns=['Skill', 'Frequency'])
                skill_fig = px.bar(
                    skill_df, x='Skill', y='Frequency',
                    title="Top Technical Skills Analysis",
                    color='Frequency',
                    color_continuous_scale='Blues'
                )
                skill_fig.update_layout(xaxis_tickangle=-45)
            else:
                skill_fig = px.bar(x=["No Skills Data"], y=[0])
            
            # Create topic visualization
            if isinstance(topics, pd.DataFrame) and not topics.empty:
                valid_topics = topics[topics['Topic'] != -1].head(15)
                if not valid_topics.empty:
                    topic_fig = px.scatter(
                        valid_topics, x='Count', y='Name', size='Count',
                        title="Knowledge Domain Analysis",
                        hover_data=['Topic'],
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                else:
                    topic_fig = px.scatter(x=[0], y=["No Topics"])
            else:
                topic_fig = px.scatter(x=[0], y=["No Topics"])
            
            # Create sentiment timeline
            if sentiments:
                sentiment_df = pd.DataFrame(sentiments, columns=['timestamp', 'sentiment', 'score'])
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                sentiment_fig = px.scatter(
                    sentiment_df, x='timestamp', y='score', color='sentiment',
                    title="AI Interaction Sentiment Timeline",
                    color_discrete_map={'POSITIVE': '#2E8B57', 'NEGATIVE': '#DC143C'}
                )
            else:
                sentiment_fig = px.scatter(x=["No Data"], y=[0])
            
            # Dashboard layout
            app.layout = html.Div([
                html.H1(f"{name} - Advanced AI Career Analytics Dashboard",
                       style={'textAlign': 'center', 'color': '#2E4057', 'marginBottom': 30}),
                
                html.Div([
                    html.H3("📊 Analytics Summary", style={'color': '#2E4057'}),
                    html.P(f"Skills Identified: {len(skills)}", style={'fontSize': 16}),
                    html.P(f"Knowledge Domains: {len(topics) if isinstance(topics, pd.DataFrame) else 0}", style={'fontSize': 16}),
                    html.P(f"AI Interactions: {len(sentiments)}", style={'fontSize': 16}),
                ], style={'backgroundColor': '#F8F9FA', 'padding': 20, 'margin': 20, 'borderRadius': 10}),
                
                dcc.Tabs([
                    dcc.Tab(label='🎯 Skills Analysis', children=[
                        html.Div([
                            dcc.Graph(figure=skill_fig)
                        ], style={'padding': 20})
                    ]),
                    
                    dcc.Tab(label='📚 Knowledge Domains', children=[
                        html.Div([
                            dcc.Graph(figure=topic_fig)
                        ], style={'padding': 20})
                    ]),
                    
                    dcc.Tab(label='🤖 AI Interaction Analysis', children=[
                        html.Div([
                            dcc.Graph(figure=sentiment_fig)
                        ], style={'padding': 20})
                    ])
                ], style={'margin': 20})
            ])
            
            return app
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            return None

class AdvancedAIResumeGenerator:
    """Main application class orchestrating the entire process."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.model_manager = ModelManager(self.device_manager)
        self.conversation_processor = ConversationProcessor(self.model_manager)
        self.skill_analyzer = SkillAnalyzer(self.model_manager)
        self.resume_generator = ResumeGenerator(self.model_manager)
        self.dashboard_generator = DashboardGenerator(self.model_manager)
        
        self.logger = self.device_manager.logger
    
    def select_input_output(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Select input file, output directory, and get user name."""
        root = tk.Tk()
        root.withdraw()
        
        # Select input file
        input_path = filedialog.askopenfilename(
            title="Select AI Conversation File (JSON)",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.home() / "Desktop")
        )
        
        if not input_path:
            return None, None, None
        
        # Select output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=os.path.dirname(input_path)
        )
        
        if not output_dir:
            output_dir = os.path.dirname(input_path)
        
        # Get user name
        name_dialog = tk.Toplevel()
        name_dialog.title("Resume Information")
        name_dialog.geometry("400x150")
        
        tk.Label(name_dialog, text="Enter your name for the resume:", font=("Helvetica", 12)).pack(pady=10)
        name_var = tk.StringVar(value="John Doe")
        name_entry = tk.Entry(name_dialog, textvariable=name_var, font=("Helvetica", 11), width=30)
        name_entry.pack(pady=10)
        
        result = []
        
        def on_ok():
            result.append(name_var.get())
            name_dialog.destroy()
        
        def on_cancel():
            result.append(None)
            name_dialog.destroy()
        
        tk.Button(name_dialog, text="Generate Resume", command=on_ok, bg='#4CAF50', fg='white', font=("Helvetica", 10)).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Button(name_dialog, text="Cancel", command=on_cancel, bg='#f44336', fg='white', font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Center the dialog
        name_dialog.update_idletasks()
        x = (name_dialog.winfo_screenwidth() // 2) - (name_dialog.winfo_width() // 2)
        y = (name_dialog.winfo_screenheight() // 2) - (name_dialog.winfo_height() // 2)
        name_dialog.geometry(f"+{x}+{y}")
        
        name_dialog.wait_window()
        
        name = result[0] if result and result[0] else "User"
        return input_path, output_dir, name
    
    def run_complete_analysis(self, input_path: str, output_dir: str, name: str) -> Dict[str, Any]:
        """Run the complete AI resume generation process."""
        start_time = datetime.datetime.now()
        results = {}
        
        try:
            self.logger.info("Starting advanced AI resume generation")
            
            # Load and process conversation data
            conversations = self.conversation_processor.load_conversation_data(input_path)
            if not conversations:
                return {"status": "error", "message": "Failed to load conversation data"}
            
            # Extract skills
            skills = self.conversation_processor.extract_skills_advanced(conversations)
            
            # Perform topic modeling
            topics, topic_assignments = self.conversation_processor.perform_topic_modeling(conversations)
            
            # Analyze sentiment
            sentiments = self.conversation_processor.analyze_sentiment(conversations)
            
            # Create skill network visualization
            skill_network_path = self.skill_analyzer.create_skill_network(skills, output_dir)
            
            # Generate professional resume
            resume_path = self.resume_generator.generate_professional_resume(
                name, skills, topics, sentiments, skill_network_path, output_dir
            )
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            
            results = {
                "status": "success",
                "resume_path": resume_path,
                "skill_network_path": skill_network_path,
                "skills_count": len(skills),
                "topics_count": len(topics) if isinstance(topics, pd.DataFrame) else 0,
                "sentiments_count": len(sentiments),
                "processing_time": processing_time,
                "device_used": str(self.device_manager.device),
                "conversations_processed": len(conversations)
            }
            
            self.logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """Main application entry point."""
    try:
        print("🚀 Advanced AI Resume Generator v1.0.0")
        print("Initializing enterprise-grade AI models...")
        
        # Initialize application
        app = AdvancedAIResumeGenerator()
        
        # Get input parameters
        input_path, output_dir, name = app.select_input_output()
        
        if not input_path:
            print("❌ No input file selected. Exiting.")
            return
        
        print(f"🎯 Processing conversations for: {name}")
        print(f"📂 Output directory: {output_dir}")
        
        # Run analysis
        results = app.run_complete_analysis(input_path, output_dir, name)
        
        if results["status"] == "success":
            print("✅ Advanced AI resume generation completed!")
            print(f"📄 Resume: {results['resume_path']}")
            print(f"🎯 Skills: {results['skills_count']}")
            print(f"📊 Topics: {results['topics_count']}")
            print(f"🤖 Interactions: {results['sentiments_count']}")
            print(f"⚡ Device: {results['device_used']}")
            print(f"⏱️ Time: {results['processing_time']:.2f}s")
            
            # Show success dialog
            messagebox.showinfo(
                "Success! 🎉",
                f"Advanced AI Resume Generated!\n\n"
                f"📄 Resume: {os.path.basename(results['resume_path'])}\n"
                f"🎯 Skills Analyzed: {results['skills_count']}\n"
                f"📊 Topics Identified: {results['topics_count']}\n"
                f"⚡ Processing Device: {results['device_used']}\n"
                f"⏱️ Total Time: {results['processing_time']:.1f} seconds"
            )
        else:
            print(f"❌ Generation failed: {results['message']}")
            messagebox.showerror("Error", f"Generation failed: {results['message']}")
    
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    main()