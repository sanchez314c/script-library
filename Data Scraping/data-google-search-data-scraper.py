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
# Script Name: data-google-search-data-scraper.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced data extraction and analysis scraper with natural       
#              language processing, intelligent sentence ranking, and link tree  
#              generation for comprehensive content analysis.                     
#
# Usage: python data-google-search-data-scraper.py [--dest DIR] [--threads NUM] [--results NUM] 
#
# Dependencies: requests, beautifulsoup4, google, fake-useragent, nltk, tkinter   
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features NLP content analysis, sentence extraction, and comprehensive   
#        link tree generation with universal macOS compatibility.                
#                                                                                
####################################################################################

"""
Google Search Data Analysis Scraper
==================================

Advanced web scraper for data extraction and analysis with natural language
processing, content ranking, and comprehensive link tree generation.
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
from typing import List, Set, Dict, Optional, Any
from urllib.parse import urljoin

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

class GoogleDataScraper:
    def __init__(self, args=None):
        self.args = args or self.parse_arguments()
        self.setup_logging()
        self.setup_directories()
        self.initialize_state()
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Advanced Data Analysis Scraper")
        parser.add_argument("--dest", help="Destination directory")
        parser.add_argument("--threads", type=int, default=max(1, multiprocessing.cpu_count() - 1))
        parser.add_argument("--results", type=int, default=10)
        parser.add_argument("--output", default="link_tree_data.json")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--version", action="store_true")
        return parser.parse_args()
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "data-google-search-data-scraper.log"
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
            self.download_dir = Path(self.args.dest)
        else:
            root = tk.Tk()
            root.withdraw()
            if IS_MACOS:
                root.attributes("-topmost", True)
            directory = filedialog.askdirectory(title="Select Output Directory")
            root.destroy()
            if not directory:
                raise SystemExit("Directory selection required")
            self.download_dir = Path(directory)
        
        self.download_dir.mkdir(exist_ok=True, parents=True)
        
    def initialize_state(self):
        self.visited_urls = set()
        self.processed_content = []
        
    def fetch_text_from_url(self, url):
        try:
            headers = {'User-Agent': UserAgent().random}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
                
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            text = " ".join([para.get_text().strip() for para in paragraphs])
            
            if not text:
                text = soup.get_text(separator=" ", strip=True)
                
            return text
                
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
            
    def clean_and_tokenize(self, text):
        if not text or len(text) < 50:
            return []
            
        try:
            sentences = sent_tokenize(text)
            stop_words = set(stopwords.words('english'))
            cleaned_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                if len(sentence) < 20:
                    continue
                    
                words = nltk.word_tokenize(sentence)
                words = [word for word in words if word.isalpha() and word.lower() not in stop_words]
                
                if len(words) >= 3:
                    cleaned_sentences.append(" ".join(words))
            
            return cleaned_sentences
        except Exception as e:
            self.logger.error(f"Error in tokenization: {e}")
            return []
            
    def find_impactful_sentences(self, sentences, top_n=5):
        if not sentences:
            return []
            
        try:
            all_words = " ".join(sentences).split()
            word_freq = Counter(all_words)
            
            # Remove common words
            common_words = {"the", "and", "to", "of", "in", "is", "that", "it", "for", "on", "with"}
            for word in common_words:
                word_freq.pop(word, None)
            
            most_common_words = [word for word, _ in word_freq.most_common(top_n)]
            impactful = [s for s in sentences if any(word in s.split() for word in most_common_words)]
            
            return impactful[:10]
            
        except Exception as e:
            self.logger.error(f"Error finding impactful sentences: {e}")
            return []
            
    def analyze_url_content(self, url):
        self.logger.debug(f"Analyzing: {url}")
        
        text = self.fetch_text_from_url(url)
        if not text:
            return {"url": url, "sentences": [], "error": "Failed to fetch content"}

        cleaned_sentences = self.clean_and_tokenize(text)
        
        if not cleaned_sentences:
            return {"url": url, "sentences": [], "error": "No meaningful content"}
            
        impactful_sentences = self.find_impactful_sentences(cleaned_sentences)
        quality_score = min(100, len(impactful_sentences) * 10)

        return {
            "url": url,
            "sentences": impactful_sentences,
            "total_sentences": len(cleaned_sentences),
            "impactful_count": len(impactful_sentences),
            "quality_score": quality_score,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def generate_link_tree(self, search_queries):
        link_tree = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_queries": len(search_queries),
                "results_per_query": self.args.results,
                "threads_used": self.args.threads
            },
            "queries": {}
        }
        
        for query_idx, query in enumerate(search_queries):
            self.logger.info(f"Processing query ({query_idx+1}/{len(search_queries)}): {query}")
            
            try:
                urls = list(search(query, num_results=self.args.results))
                self.logger.info(f"Found {len(urls)} results for '{query}'")
            except Exception as e:
                self.logger.error(f"Error searching '{query}': {e}")
                urls = []
            
            link_tree["queries"][query] = {
                "urls": urls,
                "analysis": []
            }
            
            if not urls:
                continue
                
            # Process URLs in parallel
            with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
                future_to_url = {executor.submit(self.analyze_url_content, url): url 
                               for url in urls}
                
                completed = 0
                for future in as_completed(future_to_url):
                    completed += 1
                    url = future_to_url[future]
                    
                    try:
                        analysis = future.result()
                        if analysis.get("impactful_count", 0) > 0:
                            link_tree["queries"][query]["analysis"].append(analysis)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {url}: {e}")
                    
                    self.logger.info(f"Progress: {completed}/{len(urls)} URLs for '{query}'")
                    time.sleep(0.5)
            
            # Add summary stats
            link_tree["queries"][query]["stats"] = {
                "total_urls": len(urls),
                "analyzed_urls": len(link_tree["queries"][query]["analysis"]),
                "total_sentences": sum(a.get("total_sentences", 0) for a in link_tree["queries"][query]["analysis"]),
                "impactful_sentences": sum(a.get("impactful_count", 0) for a in link_tree["queries"][query]["analysis"])
            }
            
            time.sleep(2)  # Rate limiting

        # Add overall stats
        link_tree["stats"] = {
            "total_queries": len(search_queries),
            "total_urls": sum(len(q.get("urls", [])) for q in link_tree["queries"].values()),
            "analyzed_urls": sum(q.get("stats", {}).get("analyzed_urls", 0) for q in link_tree["queries"].values()),
            "total_sentences": sum(q.get("stats", {}).get("total_sentences", 0) for q in link_tree["queries"].values()),
            "impactful_sentences": sum(q.get("stats", {}).get("impactful_sentences", 0) for q in link_tree["queries"].values())
        }
        
        return link_tree
        
    def save_link_tree(self, link_tree):
        try:
            output_file = self.download_dir / self.args.output
            with open(output_file, 'w') as f:
                json.dump(link_tree, f, indent=4)
            self.logger.info(f"Link tree saved to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving link tree: {e}")
            return None
            
    def show_completion_message(self, link_tree, output_file):
        if IS_MACOS:
            subprocess.run(['open', str(output_file)], check=False)

        stats = link_tree.get("stats", {})
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Analysis Complete",
            f"Data analysis completed.\n\n"
            f"Queries processed: {stats.get('total_queries', 0)}\n"
            f"URLs analyzed: {stats.get('analyzed_urls', 0)}/{stats.get('total_urls', 0)}\n"
            f"Sentences extracted: {stats.get('total_sentences', 0)}\n"
            f"Impactful sentences: {stats.get('impactful_sentences', 0)}\n\n"
            f"Output: {output_file}"
        )
        root.destroy()
        
    def run(self):
        if self.args.version:
            print("Google Search Data Scraper v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting data analysis process...")

        queries = [
            "LLM research methodology filetype:pdf",
            "GPT architecture analysis filetype:pdf", 
            "Neural network optimization techniques filetype:pdf",
            "AI ethics framework filetype:pdf",
            "Machine learning benchmarks filetype:pdf",
            "Quantum computing AI integration filetype:pdf",
            "Deep learning optimization filetype:pdf",
            "AI security protocols filetype:pdf",
            "Reinforcement learning advances filetype:pdf",
            "Computer vision breakthroughs filetype:pdf"
        ]

        try:
            self.logger.info(f"Starting analysis with {self.args.threads} threads")
            link_tree = self.generate_link_tree(queries)
            
            output_file = self.save_link_tree(link_tree)
            
            if output_file and not self.args.quiet:
                self.show_completion_message(link_tree, output_file)

        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("Analysis process completed")
            
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    scraper = GoogleDataScraper()
    scraper.run()

if __name__ == "__main__":
    main()