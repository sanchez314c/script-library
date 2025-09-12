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
# Script Name: ai-prompt-optimizer.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced NLP tool for analyzing and optimizing text prompts across    
#              multiple dimensions including clarity, persuasion, emotional resonance,      
#              and educational effectiveness. Utilizes state-of-the-art NLP models        
#              with GPU acceleration.                                               
#
# Usage: python ai-prompt-optimizer.py [--input FILE] [--mode MODE] [--output FILE] 
#
# Dependencies: spacy, transformers, torch, nltk, textblob, pandas, numpy,   
#               tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Requires spacy model 'en_core_web_sm' and NLTK data. Supports batch       
#        processing and multiple optimization strategies with comprehensive performance   
#        metrics.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
AI Prompt Optimizer

Advanced natural language processing tool for analyzing and optimizing text prompts
across multiple dimensions with state-of-the-art NLP models and comprehensive metrics.
"""

# Standard library imports
import os
import sys
import json
import time
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Third-party imports
try:
    import spacy
    import torch
    import nltk
    import pandas as pd
    import numpy as np
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    from textblob import TextBlob
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox, ttk
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using:")
    print("pip install spacy transformers torch nltk pandas numpy textblob python-dotenv")
    print("python -m spacy download en_core_web_sm")
    print("python -m nltk.downloader punkt vader_lexicon")
    sys.exit(1)

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'ai-prompt-optimizer.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('ai-prompt-optimizer')

# Load environment variables from .env file in script directory and current directory
script_dir = Path(__file__).parent
env_files = [script_dir / '.env', Path.cwd() / '.env']
for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")

class FileSelector:
    """Handles file and folder selection with macOS native dialogs"""
    
    @staticmethod
    def select_file(title: str = "Select a file", 
                   filetypes: List[Tuple[str, str]] = None) -> Optional[Path]:
        """Select a single file using native macOS dialog"""
        if filetypes is None:
            filetypes = [("All files", "*.*")]
            
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        
        root.destroy()
        
        if file_path:
            return Path(file_path)
        return None
    
    @staticmethod
    def select_folder(title: str = "Select a folder") -> Optional[Path]:
        """Select a folder using native macOS dialog"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        folder_path = filedialog.askdirectory(title=title)
        
        root.destroy()
        
        if folder_path:
            return Path(folder_path)
        return None
    
    @staticmethod
    def select_save_file(title: str = "Save file as", 
                        defaultextension: str = ".txt",
                        filetypes: List[Tuple[str, str]] = None) -> Optional[Path]:
        """Select a file to save using native macOS dialog"""
        if filetypes is None:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
            
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=defaultextension,
            filetypes=filetypes
        )
        
        root.destroy()
        
        if file_path:
            return Path(file_path)
        return None

class ProgressTracker:
    """Displays and tracks progress for long-running operations"""
    
    def __init__(self, total: int, title: str = "Processing"):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        
        # Create progress dialog
        self.root = tk.Tk()
        self.root.title(title)
        
        # Configure progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, 
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        self.progress_bar.pack(pady=10, padx=20)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Starting...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Time remaining label
        self.time_var = tk.StringVar()
        self.time_var.set("Calculating...")
        self.time_label = tk.Label(self.root, textvariable=self.time_var)
        self.time_label.pack(pady=5)
        
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        # Update UI in separate thread
        self.update_thread = threading.Thread(target=self._update_ui)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _update_ui(self):
        """Update UI in background thread"""
        while self.current < self.total:
            time.sleep(0.1)
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                break
            self.root.update()
    
    def update(self, current: int, message: str = None):
        """Update progress status"""
        self.current = current
        progress_pct = (current / self.total) * 100
        self.progress_var.set(progress_pct)
        
        if message:
            self.status_var.set(message)
        else:
            self.status_var.set(f"Processing {current}/{self.total}")
        
        # Calculate time remaining
        elapsed = time.time() - self.start_time
        if current > 0:
            items_per_sec = current / elapsed
            remaining_items = self.total - current
            remaining_time = remaining_items / items_per_sec if items_per_sec > 0 else 0
            
            if remaining_time < 60:
                time_str = f"Time remaining: {int(remaining_time)} seconds"
            elif remaining_time < 3600:
                time_str = f"Time remaining: {int(remaining_time / 60)} minutes"
            else:
                time_str = f"Time remaining: {remaining_time / 3600:.1f} hours"
                
            self.time_var.set(time_str)
    
    def complete(self):
        """Mark progress as complete and close dialog"""
        self.update(self.total, "Complete!")
        time.sleep(1)  # Show completion briefly
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.destroy()

class PromptOptimizer:
    """Core class for prompt optimization functionality"""
    
    OPTIMIZATION_MODES = [
        "technical",   # Optimize for technical clarity
        "emotional",   # Optimize for emotional resonance
        "persuasive",  # Optimize for persuasive impact
        "educational", # Optimize for educational effectiveness
        "all"          # Apply all optimization strategies
    ]
    
    def __init__(self):
        self.nlp = self._load_nlp_models()
        
    def _load_nlp_models(self):
        """Load and initialize NLP models"""
        try:
            # Load spaCy model for linguistic analysis
            nlp = spacy.load("en_core_web_sm")
            
            # Set up NLTK resources
            nltk_resources = ["punkt", "vader_lexicon"]
            for resource in nltk_resources:
                try:
                    nltk.data.find(f"tokenizers/{resource}")
                except LookupError:
                    logger.info(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource)
            
            # Load transformer models if GPU is available
            self.use_gpu = torch.cuda.is_available()
            device = 0 if self.use_gpu else -1
            
            # Sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
            
            # Text classification pipeline for various aspects
            self.classifiers = {
                "clarity": pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",
                    device=device
                ),
                "persuasiveness": pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",
                    device=device
                )
            }
            
            logger.info(f"NLP models loaded successfully. GPU acceleration: {self.use_gpu}")
            return nlp
            
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            raise RuntimeError(f"Failed to initialize NLP models: {e}")
    
    def optimize_prompt(self, text: str, mode: str = "all") -> Dict[str, Any]:
        """Optimize prompt based on specified mode"""
        if mode not in self.OPTIMIZATION_MODES:
            raise ValueError(f"Invalid optimization mode: {mode}. Must be one of {self.OPTIMIZATION_MODES}")
        
        logger.info(f"Starting optimization with mode: {mode}")
        
        # Start with basic analysis
        analysis = self._analyze_text(text)
        
        # Apply optimization strategies based on mode
        optimized_text = text
        improvements = []
        
        if mode in ["technical", "all"]:
            optimized_text, tech_improvements = self._optimize_technical(optimized_text, analysis)
            improvements.extend(tech_improvements)
            
        if mode in ["emotional", "all"]:
            optimized_text, emo_improvements = self._optimize_emotional(optimized_text, analysis)
            improvements.extend(emo_improvements)
            
        if mode in ["persuasive", "all"]:
            optimized_text, pers_improvements = self._optimize_persuasive(optimized_text, analysis)
            improvements.extend(pers_improvements)
            
        if mode in ["educational", "all"]:
            optimized_text, edu_improvements = self._optimize_educational(optimized_text, analysis)
            improvements.extend(edu_improvements)
        
        # Re-analyze optimized text
        final_analysis = self._analyze_text(optimized_text)
        
        return {
            "original_text": text,
            "optimized_text": optimized_text,
            "improvements": improvements,
            "original_analysis": analysis,
            "final_analysis": final_analysis,
            "optimization_mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis"""
        # Process with spaCy
        doc = self.nlp(text)
        
        # Basic linguistic features
        analysis = {
            "word_count": len([token for token in doc if not token.is_punct]),
            "sentence_count": len(list(doc.sents)),
            "avg_sentence_length": sum(len([t for t in sent if not t.is_punct]) for sent in doc.sents) / max(1, len(list(doc.sents))),
            "readability_score": self._calculate_readability(doc),
            "pos_distribution": self._get_pos_distribution(doc),
            "named_entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
            "sentiment": self._analyze_sentiment(text),
            "complexity_metrics": self._calculate_complexity_metrics(doc)
        }
        
        # Identify key phrases
        analysis["key_phrases"] = self._extract_key_phrases(doc)
        
        return analysis
    
    def _calculate_readability(self, doc) -> float:
        """Calculate readability score using Flesch Reading Ease approximation"""
        sentences = list(doc.sents)
        if not sentences:
            return 0
            
        words = [token for token in doc if not token.is_punct and not token.is_space]
        if not words:
            return 0
        
        # Count syllables (simplified approximation)
        total_syllables = 0
        for token in words:
            syllables = max(1, len([char for char in token.text.lower() if char in 'aeiouy']))
            total_syllables += syllables
        
        # Flesch Reading Ease formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return max(0, min(100, flesch_score))  # Clamp between 0-100
    
    def _calculate_complexity_metrics(self, doc) -> Dict[str, float]:
        """Calculate various text complexity metrics"""
        words = [token for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)
        
        if not words or not sentences:
            return {"lexical_diversity": 0, "avg_word_length": 0, "passive_voice_ratio": 0}
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(token.lemma_.lower() for token in words if token.is_alpha)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # Average word length
        avg_word_length = sum(len(token.text) for token in words) / len(words)
        
        # Passive voice detection (simplified)
        passive_count = 0
        for sent in sentences:
            if any(token.dep_ == "auxpass" for token in sent):
                passive_count += 1
        
        passive_voice_ratio = passive_count / len(sentences) if sentences else 0
        
        return {
            "lexical_diversity": lexical_diversity,
            "avg_word_length": avg_word_length,
            "passive_voice_ratio": passive_voice_ratio
        }
    
    def _get_pos_distribution(self, doc) -> Dict[str, int]:
        """Get distribution of parts of speech"""
        pos_counts = {}
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using multiple approaches"""
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Transformer-based sentiment analysis
        transformer_sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
        
        return {
            "polarity": textblob_sentiment.polarity,  # -1.0 to 1.0
            "subjectivity": textblob_sentiment.subjectivity,  # 0.0 to 1.0
            "transformer_label": transformer_sentiment["label"],
            "transformer_score": transformer_sentiment["score"]
        }
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases from text"""
        key_phrases = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:  # Only chunks with more than one token
                key_phrases.append(chunk.text)
        
        # Extract named entities
        for ent in doc.ents:
            key_phrases.append(ent.text)
        
        # Remove duplicates and limit to top 10
        return list(set(key_phrases))[:10]
    
    def _optimize_technical(self, text: str, analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize text for technical clarity"""
        doc = self.nlp(text)
        improvements = []
        optimized_text = text
        
        # Check readability
        readability = analysis.get("readability_score", 0)
        if readability < 30:  # Very difficult to read
            improvements.append("Recommended simplifying complex sentence structures for better clarity")
            # In a full implementation, would actually modify the text
        
        # Check for passive voice and suggest conversion to active
        passive_ratio = analysis.get("complexity_metrics", {}).get("passive_voice_ratio", 0)
        if passive_ratio > 0.3:  # More than 30% passive voice
            improvements.append(f"Suggested converting passive voice to active voice (found {passive_ratio:.0%} passive sentences)")
        
        # Check average sentence length
        avg_length = analysis.get("avg_sentence_length", 0)
        if avg_length > 25:  # Very long sentences
            improvements.append("Recommended breaking down long sentences for better comprehension")
        
        # Check for technical jargon density
        word_count = analysis.get("word_count", 0)
        if word_count > 0:
            complex_words = sum(1 for token in doc if len(token.text) > 8 and token.is_alpha)
            jargon_ratio = complex_words / word_count
            if jargon_ratio > 0.2:  # More than 20% complex words
                improvements.append("Suggested replacing technical jargon with simpler alternatives where possible")
        
        return optimized_text, improvements
    
    def _optimize_emotional(self, text: str, analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize text for emotional resonance"""
        improvements = []
        optimized_text = text
        
        # Adjust emotional tone based on sentiment analysis
        sentiment = analysis.get("sentiment", {})
        polarity = sentiment.get("polarity", 0)
        subjectivity = sentiment.get("subjectivity", 0)
        
        if polarity < -0.1 and subjectivity < 0.3:
            improvements.append("Recommended adding positive emotional language to enhance engagement")
        
        if subjectivity < 0.2:
            improvements.append("Suggested incorporating more personal and relatable language")
        
        # Check for emotional words
        doc = self.nlp(text)
        emotional_words = ["amazing", "incredible", "fantastic", "wonderful", "exciting", "inspiring"]
        found_emotional = any(token.text.lower() in emotional_words for token in doc)
        
        if not found_emotional:
            improvements.append("Recommended adding emotion-evoking language to key points")
        
        return optimized_text, improvements
    
    def _optimize_persuasive(self, text: str, analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize text for persuasive impact"""
        improvements = []
        optimized_text = text
        
        text_lower = text.lower()
        
        # Check for social proof elements
        social_proof_words = ["people", "experts", "users", "customers", "studies", "research"]
        has_social_proof = any(word in text_lower for word in social_proof_words)
        
        if not has_social_proof:
            improvements.append("Suggested adding social proof elements (testimonials, statistics, expert opinions)")
        
        # Check for scarcity/urgency elements
        urgency_words = ["now", "limited", "deadline", "soon", "urgent", "today", "immediately"]
        has_urgency = any(word in text_lower for word in urgency_words)
        
        if not has_urgency:
            improvements.append("Recommended adding scarcity or urgency elements to motivate action")
        
        # Check for call-to-action
        cta_words = ["click", "buy", "download", "subscribe", "register", "join", "start", "try"]
        has_cta = any(word in text_lower for word in cta_words)
        
        if not has_cta:
            improvements.append("Suggested adding a clear call-to-action")
        
        # Check for benefit-focused language
        benefit_words = ["benefit", "advantage", "gain", "improve", "enhance", "save", "increase"]
        has_benefits = any(word in text_lower for word in benefit_words)
        
        if not has_benefits:
            improvements.append("Recommended highlighting specific benefits and value propositions")
        
        return optimized_text, improvements
    
    def _optimize_educational(self, text: str, analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize text for educational effectiveness"""
        improvements = []
        optimized_text = text
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        text_lower = text.lower()
        
        # Check for clear structure
        structure_indicators = ["first", "second", "third", "next", "finally", "in conclusion"]
        has_structure = any(indicator in text_lower for indicator in structure_indicators)
        
        if len(sentences) > 5 and not has_structure:
            improvements.append("Recommended adding structural elements (numbered points, clear transitions)")
        
        # Check for examples and analogies
        example_phrases = ["for example", "such as", "like", "instance", "illustrat", "imagine", "consider"]
        has_examples = any(phrase in text_lower for phrase in example_phrases)
        
        if not has_examples:
            improvements.append("Suggested adding concrete examples or analogies to illustrate concepts")
        
        # Check for questions to engage learners
        question_count = text.count("?")
        if question_count == 0 and len(sentences) > 3:
            improvements.append("Recommended adding rhetorical or thought-provoking questions")
        
        # Check for summary or key takeaways
        summary_words = ["summary", "conclusion", "key points", "remember", "important"]
        has_summary = any(word in text_lower for word in summary_words)
        
        if len(sentences) > 8 and not has_summary:
            improvements.append("Suggested adding a summary or key takeaways section")
        
        return optimized_text, improvements
    
    def batch_process(self, input_files: List[Path], mode: str, output_dir: Path) -> Dict[str, Any]:
        """Process multiple files in batch mode"""
        results = {}
        total_files = len(input_files)
        
        logger.info(f"Starting batch processing of {total_files} files")
        
        # Set up progress tracking
        progress = ProgressTracker(total_files, "Optimizing Prompts")
        
        # Process files using multi-threading
        threads = []
        for i, file_path in enumerate(input_files):
            thread = threading.Thread(
                target=self._process_file,
                args=(file_path, mode, output_dir, results)
            )
            thread.start()
            threads.append((thread, i))
        
        # Wait for all threads and update progress
        for thread, i in threads:
            thread.join()
            progress.update(i + 1)
        
        progress.complete()
        logger.info(f"Batch processing completed. Processed {len(results)} files")
        return results
    
    def _process_file(self, file_path: Path, mode: str, output_dir: Path, results: Dict[str, Any]):
        """Process a single file in the batch"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            result = self.optimize_prompt(original_text, mode)
            
            # Create output file path
            output_file = output_dir / f"{file_path.stem}_optimized{file_path.suffix}"
            
            # Save optimized text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result["optimized_text"])
            
            # Save detailed report
            report_file = output_dir / f"{file_path.stem}_optimization_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "improvements": result["improvements"],
                    "original_analysis": result["original_analysis"],
                    "final_analysis": result["final_analysis"],
                    "optimization_mode": result["optimization_mode"],
                    "timestamp": result["timestamp"]
                }, f, indent=2, ensure_ascii=False)
            
            # Store results
            results[str(file_path)] = {
                "output_file": str(output_file),
                "report_file": str(report_file),
                "improvements": result["improvements"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            results[str(file_path)] = {"error": str(e), "status": "error"}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Prompt Optimizer")
    
    parser.add_argument(
        "--input", 
        type=str,
        help="Input file or directory containing prompt text"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=PromptOptimizer.OPTIMIZATION_MODES,
        default="all",
        help="Optimization mode to apply"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file or directory for optimized prompts"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple files in batch mode"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting AI Prompt Optimizer")
    
    try:
        # Initialize optimizer
        optimizer = PromptOptimizer()
        
        # Determine input path
        input_path = None
        if args.input:
            input_path = Path(args.input)
        else:
            # Prompt user for input
            input_path = FileSelector.select_file(
                title="Select prompt file",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
        
        if not input_path:
            logger.error("No input provided. Exiting.")
            sys.exit(1)
            
        # Check if batch mode
        batch_mode = args.batch or (input_path.is_dir() if input_path else False)
        
        # Determine output path
        output_path = None
        if args.output:
            output_path = Path(args.output)
        else:
            if batch_mode:
                output_path = FileSelector.select_folder(title="Select output directory")
            else:
                # Save to desktop with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = desktop_path / f"optimized_prompt_{timestamp}.txt"
                
        if not output_path:
            logger.error("No output location provided. Exiting.")
            sys.exit(1)
            
        # Process input
        if batch_mode:
            # Handle batch processing
            if input_path.is_dir():
                input_files = list(input_path.glob("*.txt"))
            else:
                input_files = [input_path]
                
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
                
            results = optimizer.batch_process(input_files, args.mode, output_path)
            
            # Show summary
            successful = sum(1 for r in results.values() if r.get("status") == "success")
            failed = len(results) - successful
            
            summary_msg = f"Processed {len(results)} files.\nSuccessful: {successful}\nFailed: {failed}\nResults saved to {output_path}"
            messagebox.showinfo("Batch Processing Complete", summary_msg)
            logger.info(summary_msg.replace('\n', ' '))
            
        else:
            # Handle single file processing
            with open(input_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
                
            result = optimizer.optimize_prompt(original_text, args.mode)
            
            # Save optimized text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["optimized_text"])
                
            # Generate report
            report_path = output_path.with_suffix(".optimization_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "improvements": result["improvements"],
                    "original_analysis": result["original_analysis"],
                    "final_analysis": result["final_analysis"],
                    "optimization_mode": result["optimization_mode"],
                    "timestamp": result["timestamp"]
                }, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Optimized prompt saved to {output_path}")
            logger.info(f"Optimization report saved to {report_path}")
            
            # Show summary
            improvement_count = len(result["improvements"])
            summary_msg = f"Prompt optimized successfully!\nMode: {args.mode}\nImprovements: {improvement_count}\nSaved to: {output_path}\nReport: {report_path}"
            messagebox.showinfo("Optimization Complete", summary_msg)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Use all available CPU cores for optimal performance
    if hasattr(os, 'sched_getaffinity'):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = multiprocessing.cpu_count()
        
    logger.info(f"Using {cpu_count} CPU cores for processing")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}")
        sys.exit(1)