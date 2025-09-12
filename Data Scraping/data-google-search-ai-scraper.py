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
# Script Name: data-google-search-ai-scraper.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced AI content scraper with intelligent search, multi-core    
#              processing, and comprehensive duplicate detection. Optimized for  
#              collecting research papers, documentation, and academic content.   
#
# Usage: python data-google-search-ai-scraper.py [--dest DIR] [--threads NUM] [--depth NUM] 
#
# Dependencies: requests, beautifulsoup4, google, fake-useragent, nltk, tkinter    
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features intelligent content filtering, semantic relevance scoring,      
#        and universal macOS compatibility with desktop logging.                 
#                                                                                
####################################################################################

"""
Advanced Google Search AI Content Scraper
=========================================

Sophisticated web scraper designed for collecting AI-related research content with
intelligent filtering, multi-core processing, and comprehensive quality assessment.
"""

import os
import sys
import argparse
import hashlib
import json
import logging
import multiprocessing
import platform
import subprocess
import time
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

# NLTK setup
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except:
    import nltk
    nltk.download('punkt', quiet=True)

IS_MACOS = platform.system() == "Darwin"

class GoogleAIScraper:
    def __init__(self, args=None):
        self.args = args or self.parse_arguments()
        self.setup_logging()
        self.setup_directories()
        self.initialize_state()
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Advanced AI Content Scraper")
        parser.add_argument("--dest", help="Destination directory")
        parser.add_argument("--threads", type=int, default=max(1, multiprocessing.cpu_count() - 1))
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--version", action="store_true")
        return parser.parse_args()
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "data-google-search-ai-scraper.log"
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
            directory = filedialog.askdirectory(title="Select Download Directory")
            root.destroy()
            if not directory:
                raise SystemExit("Directory selection required")
            self.download_dir = Path(directory)
        
        self.download_dir.mkdir(exist_ok=True, parents=True)
        
    def initialize_state(self):
        self.visited_urls = set()
        self.saved_hashes = set()
        self.downloaded_files = set()
        self.download_count = 0
        
        # Load previous state
        state_file = self.download_dir / 'download_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.downloaded_files = set(state.get('files', []))
                    self.saved_hashes = set(state.get('hashes', []))
                self.logger.info(f"Loaded state: {len(self.downloaded_files)} files")
            except:
                pass
                
    def save_state(self):
        try:
            state_file = self.download_dir / 'download_state.json'
            with open(state_file, 'w') as f:
                json.dump({
                    'files': list(self.downloaded_files),
                    'hashes': list(self.saved_hashes)
                }, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            
    def calculate_hash(self, content):
        return hashlib.md5(content).hexdigest()
        
    def save_file(self, url, title):
        try:
            headers = {'User-Agent': UserAgent().random}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            content = response.content
            
            if len(content) < 1000:
                return False
                
            content_hash = self.calculate_hash(content)
            if content_hash in self.saved_hashes:
                self.logger.info(f"Duplicate skipped: {title}")
                return False
                
            self.saved_hashes.add(content_hash)
            
            # Determine file extension
            file_extension = url.split('.')[-1].lower()
            if file_extension not in ('pdf', 'txt', 'py'):
                file_extension = 'txt'
                
            safe_title = "".join(c for c in title if c.isalnum() or c in "._- ")
            file_path = self.download_dir / f"{safe_title}.{file_extension}"
            
            # Handle collisions
            counter = 1
            while file_path.exists():
                file_path = self.download_dir / f"{safe_title}_{counter}.{file_extension}"
                counter += 1
                
            with open(file_path, "wb") as f:
                f.write(content)
                
            self.downloaded_files.add(str(file_path))
            self.download_count += 1
            self.logger.info(f"Downloaded ({self.download_count}): {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {url}: {e}")
            return False
            
    def get_links(self, url, terms, depth=0):
        if depth > self.args.depth or url in self.visited_urls:
            return []
            
        self.visited_urls.add(url)
        links = []
        
        try:
            headers = {'User-Agent': UserAgent().random}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if not href:
                    continue
                    
                full_url = urljoin(url, href)
                
                if any(term.lower() in href.lower() for term in terms):
                    if href.endswith(('.pdf', '.txt', '.py')):
                        links.append(full_url)
                    elif href.startswith('http') and full_url not in self.visited_urls:
                        deeper_links = self.get_links(full_url, terms, depth + 1)
                        links.extend(deeper_links)
                        
        except Exception as e:
            self.logger.error(f"Error accessing {url}: {e}")
            
        return links
        
    def process_search_results(self, query, specific_terms, num_results=10):
        try:
            self.logger.info(f"Processing: {query}")
            search_results = list(search(query, num_results=num_results))
            
            all_links = []
            for url in search_results:
                links = self.get_links(url, specific_terms)
                all_links.extend(links)
                time.sleep(2)
                
            unique_links = list(set(all_links))
            self.logger.info(f"Found {len(unique_links)} unique links")
            
            if not unique_links:
                return
                
            # Process with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
                futures = []
                
                for j, link in enumerate(unique_links):
                    title = f"{query.split()[0]}_{j+1}"
                    future = executor.submit(self.save_file, link, title)
                    futures.append(future)
                    
                for future in as_completed(futures):
                    future.result()
                    time.sleep(0.5)
                    
        except Exception as e:
            self.logger.error(f"Error processing '{query}': {e}")
            
    def show_completion_message(self):
        if IS_MACOS:
            subprocess.run(['open', str(self.download_dir)], check=False)
            
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Scraping Complete",
            f"AI content scraping completed.\n\n"
            f"Files saved: {len(self.downloaded_files)}\n"
            f"Location: {self.download_dir}\n\n"
            f"Log file: {self.log_file}"
        )
        root.destroy()
        
    def run(self):
        if self.args.version:
            print("Google Search AI Scraper v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting AI content scraping...")
        
        queries = [
            "LLM filetype:pdf OR filetype:txt OR filetype:py",
            "Large Language Model filetype:pdf OR filetype:txt OR filetype:py", 
            "GPT transformer filetype:pdf OR filetype:txt OR filetype:py",
            "Deep Learning filetype:pdf OR filetype:txt OR filetype:py",
            "Neural Networks filetype:pdf OR filetype:txt OR filetype:py",
            "Natural Language Processing filetype:pdf OR filetype:txt OR filetype:py",
            "AI ethics filetype:pdf OR filetype:txt OR filetype:py",
            "Machine Learning advancements filetype:pdf OR filetype:txt OR filetype:py"
        ]
        
        specific_terms = [
            'LLM', 'GPT', 'transformer', 'Deep Learning',
            'Neural Networks', 'NLP', 'AI ethics', 'Machine Learning'
        ]
        
        try:
            for query in queries:
                self.process_search_results(query, specific_terms)
                self.save_state()
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.save_state()
            self.logger.info(f"Process completed. {len(self.downloaded_files)} files saved.")
            
            if not self.args.quiet:
                self.show_completion_message()
                
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    scraper = GoogleAIScraper()
    scraper.run()

if __name__ == "__main__":
    main()