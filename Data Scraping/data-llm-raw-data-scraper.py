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
# Script Name: data-llm-raw-data-scraper.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Specialized LLM training data scraper with quality assessment,     
#              content filtering, and intelligent text extraction optimized for  
#              collecting high-quality training datasets.                        
#
# Usage: python data-llm-raw-data-scraper.py [--dest DIR] [--quality FLOAT] [--threads NUM] 
#
# Dependencies: requests, beautifulsoup4, google, fake-useragent, nltk, tkinter   
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features semantic relevance scoring, content deduplication, and         
#        comprehensive quality filtering for LLM training data collection.       
#                                                                                
####################################################################################

"""
LLM Raw Training Data Scraper
============================

Specialized scraper for collecting high-quality text data suitable for LLM training
with advanced content filtering, quality assessment, and intelligent extraction.
"""

import os
import sys
import argparse
import json
import logging
import multiprocessing
import platform
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Set, Optional, Any

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "google>=3.0.0",
        "fake-useragent>=0.1.11", 
        "nltk>=3.6.0"
    ]
    
    missing = []
    for pkg in required_packages:
        name = pkg.split('>=')[0].replace('-', '_')
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

check_and_install_dependencies()

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from googlesearch import search
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

IS_MACOS = platform.system() == "Darwin"

class LLMDataScraper:
    def __init__(self, args=None):
        self.args = args or self.parse_arguments()
        self.setup_logging()
        self.setup_directories()
        self.initialize_state()
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="LLM Training Data Scraper")
        parser.add_argument("--dest", help="Output directory")
        parser.add_argument("--threads", type=int, default=max(1, multiprocessing.cpu_count() - 1))
        parser.add_argument("--results", type=int, default=10)
        parser.add_argument("--quality", type=float, default=0.5, help="Quality threshold (0.0-1.0)")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--version", action="store_true")
        return parser.parse_args()
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "data-llm-raw-data-scraper.log"
        logging.basicConfig(
            level=logging.DEBUG if not self.args.quiet else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
    def setup_directories(self):
        if self.args.dest:
            self.output_dir = Path(self.args.dest)
        else:
            root = tk.Tk()
            root.withdraw()
            if IS_MACOS:
                root.attributes("-topmost", True)
            directory = filedialog.askdirectory(title="Select Output Directory")
            root.destroy()
            if not directory:
                raise SystemExit("Directory selection required")
            self.output_dir = Path(directory)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def initialize_state(self):
        self.processed_urls = set()
        self.stats = {
            'urls_processed': 0,
            'content_extracted': 0,
            'sentences_collected': 0,
            'quality_content_ratio': 0.0,
            'start_time': time.time()
        }
        
    def fetch_text_from_url(self, url):
        try:
            headers = {'User-Agent': UserAgent().random}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            text = " ".join([para.get_text().strip() for para in paragraphs])
            
            if not text:
                main_content = soup.find(['article', 'main', 'div', 'body'])
                if main_content:
                    text = main_content.get_text(separator=" ", strip=True)
                else:
                    text = soup.get_text(separator=" ", strip=True)
                    
            # Skip very small texts
            if len(text) < 200:
                return None
                
            return text

        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
            
    def clean_and_tokenize(self, text):
        if not text:
            return []
            
        try:
            sentences = sent_tokenize(text)
            stop_words = set(stopwords.words('english'))
            cleaned_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Quality filters
                if not (
                    len(sentence.split()) > 5 and
                    len(sentence) < 1000 and
                    sentence.count('.') < 3 and
                    any(c.isalpha() for c in sentence)
                ):
                    continue
                
                # Skip sentences with excessive punctuation or numbers
                if (
                    sum(c.isdigit() for c in sentence) / len(sentence) > 0.2 or
                    sum(c in '!?*#@%$&' for c in sentence) > 3
                ):
                    continue
                    
                words = nltk.word_tokenize(sentence)
                words = [word for word in words if word.isalpha() and word.lower() not in stop_words]
                
                if len(words) >= 3:
                    cleaned_sentences.append(" ".join(words))
            
            return cleaned_sentences

        except Exception as e:
            self.logger.error(f"Error in tokenization: {e}")
            return []
            
    def assess_content_quality(self, sentences):
        if not sentences:
            return 0.0

        try:
            # Calculate average sentence length
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # Calculate vocabulary diversity
            all_words = [word.lower() for s in sentences for word in s.split()]
            vocab_diversity = len(set(all_words)) / len(all_words) if all_words else 0
            
            # Calculate content coherence
            coherence = sum(1 for s in sentences if len(s.split()) > 10) / len(sentences)
            
            # Weighted quality score
            length_score = min(avg_length / 20, 1.0)
            diversity_score = min(vocab_diversity * 2, 1.0)
            coherence_score = min(coherence * 1.5, 1.0)
            
            return (length_score * 0.3 + diversity_score * 0.4 + coherence_score * 0.3)

        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return 0.0
            
    def process_url(self, url):
        if url in self.processed_urls:
            return None

        self.processed_urls.add(url)
        self.logger.debug(f"Processing: {url}")
        
        text = self.fetch_text_from_url(url)
        if not text:
            return None

        sentences = self.clean_and_tokenize(text)
        if not sentences:
            return None
            
        quality_score = self.assess_content_quality(sentences)
        
        metadata = {
            'url': url,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'quality_score': quality_score,
            'sentence_count': len(sentences),
            'word_count': sum(len(s.split()) for s in sentences)
        }

        return {
            'metadata': metadata,
            'sentences': sentences
        }
        
    def save_content(self, content, query):
        try:
            safe_query = "".join(c for c in query if c.isalnum() or c in " _").replace(" ", "_")
            output_file = self.output_dir / f"llm_data_{safe_query}.json"
            
            existing_data = []
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse {output_file}, creating new")
            
            existing_data.append(content)
            
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            self.logger.debug(f"Saved content to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving content: {e}")
            return False
            
    def update_stats(self, content):
        self.stats['urls_processed'] += 1
        self.stats['content_extracted'] += 1
        
        metadata = content.get('metadata', {})
        sentence_count = metadata.get('sentence_count', 0)
        self.stats['sentences_collected'] += sentence_count
        
        if self.stats['urls_processed'] > 0:
            self.stats['quality_content_ratio'] = (
                self.stats['sentences_collected'] / 
                self.stats['urls_processed']
            )
            
        elapsed_time = time.time() - self.stats['start_time']
        self.stats['elapsed_seconds'] = elapsed_time
        if elapsed_time > 0:
            self.stats['sentences_per_second'] = self.stats['sentences_collected'] / elapsed_time
            
    def process_search_results(self, query):
        try:
            self.logger.info(f"Processing: {query}")
            
            try:
                urls = list(search(query, num_results=self.args.results))
                self.logger.info(f"Found {len(urls)} results for '{query}'")
            except Exception as e:
                self.logger.error(f"Error searching '{query}': {e}")
                return
                
            if not urls:
                self.logger.warning(f"No URLs found for: {query}")
                return

            # Process URLs in parallel
            with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
                future_to_url = {
                    executor.submit(self.process_url, url): url for url in urls
                }
                
                completed = 0
                saved_count = 0
                
                for future in as_completed(future_to_url):
                    completed += 1
                    url = future_to_url[future]
                    
                    try:
                        content = future.result()
                        
                        if not content:
                            self.logger.debug(f"No content from {url}")
                            continue
                            
                        # Check quality threshold
                        quality_score = content.get('metadata', {}).get('quality_score', 0)
                        if quality_score < self.args.quality:
                            self.logger.debug(f"Quality too low ({quality_score:.2f}): {url}")
                            continue
                            
                        self.save_content(content, query)
                        self.update_stats(content)
                        saved_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {url}: {e}")
                    
                    self.logger.info(f"Progress: {completed}/{len(urls)} | Saved: {saved_count}")
                    time.sleep(0.5)

            self.logger.info(f"Query completed: {saved_count}/{len(urls)} quality content")

        except Exception as e:
            self.logger.error(f"Error processing '{query}': {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def save_stats(self):
        try:
            self.stats['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            elapsed = self.stats.get('elapsed_seconds', 0)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.stats['elapsed_formatted'] = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            stats_file = self.output_dir / 'scraping_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
                
            self.logger.info(f"Statistics saved to {stats_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")
            
    def show_completion_message(self):
        if IS_MACOS:
            subprocess.run(['open', str(self.output_dir)], check=False)

        elapsed = self.stats.get('elapsed_seconds', 0)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Data Collection Complete",
            f"LLM data collection completed.\n\n"
            f"URLs processed: {self.stats.get('urls_processed', 0)}\n"
            f"Content extracted: {self.stats.get('content_extracted', 0)}\n"
            f"Sentences collected: {self.stats.get('sentences_collected', 0)}\n"
            f"Elapsed time: {elapsed_str}\n\n"
            f"Data saved to: {self.output_dir}"
        )
        root.destroy()
        
    def run(self):
        if self.args.version:
            print("LLM Raw Data Scraper v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting LLM data collection...")

        queries = [
            "artificial intelligence consciousness research",
            "hybrid digital neuromorphic computing advances",
            "cutting-edge AI technologies development", 
            "philosophical implications artificial intelligence",
            "government AI regulation framework",
            "quantum computing AI integration",
            "artificial consciousness development",
            "brain-computer interface breakthroughs",
            "simulated reality research advances",
            "transhumanism artificial intelligence",
            "space exploration AI applications"
        ]

        try:
            for i, query in enumerate(queries):
                self.logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
                self.process_search_results(query)
                self.save_stats()
                time.sleep(2)

        except KeyboardInterrupt:
            self.logger.info("Data collection interrupted")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.save_stats()
            self.logger.info(f"Collection completed. {self.stats.get('sentences_collected', 0)} sentences collected.")
            
            # Print final stats
            self.logger.info(f"""
            Final Stats:
            - URLs processed: {self.stats.get('urls_processed', 0)}
            - Content extracted: {self.stats.get('content_extracted', 0)}
            - Sentences collected: {self.stats.get('sentences_collected', 0)}
            - Quality ratio: {self.stats.get('quality_content_ratio', 0):.2f}
            """)
            
            if not self.args.quiet:
                self.show_completion_message()
                
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    scraper = LLMDataScraper()
    scraper.run()

if __name__ == "__main__":
    main()