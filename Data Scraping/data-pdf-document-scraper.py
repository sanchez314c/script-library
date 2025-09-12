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
# Script Name: data-pdf-document-scraper.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-performance PDF document scraper optimized for academic      
#              and research papers with recursive crawling, intelligent          
#              filtering, and comprehensive metadata preservation.               
#
# Usage: python data-pdf-document-scraper.py [--dest DIR] [--threads NUM] [--depth NUM] 
#
# Dependencies: requests, beautifulsoup4, fake-useragent, tkinter                 
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features recursive website crawling, duplicate detection, and           
#        academic content prioritization with universal macOS compatibility.     
#                                                                                
####################################################################################

"""
PDF Document Research Scraper
============================

High-performance PDF scraper designed for academic and research document collection
with intelligent filtering, recursive crawling, and comprehensive quality assessment.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Set, Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "fake-useragent>=0.1.11"
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
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

IS_MACOS = platform.system() == "Darwin"

class PDFScraper:
    def __init__(self, args=None):
        self.args = args or self.parse_arguments()
        self.setup_logging()
        self.setup_directories()
        self.initialize_state()
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="High-Performance PDF Scraper")
        parser.add_argument("--dest", help="Destination directory")
        parser.add_argument("--threads", type=int, default=max(1, multiprocessing.cpu_count() - 1))
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--start-url", action="append", help="Starting URLs")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--version", action="store_true")
        return parser.parse_args()
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "data-pdf-document-scraper.log"
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
            self.save_dir = Path(self.args.dest)
        else:
            root = tk.Tk()
            root.withdraw()
            if IS_MACOS:
                root.attributes("-topmost", True)
            directory = filedialog.askdirectory(title="Select PDF Save Directory")
            root.destroy()
            if not directory:
                raise SystemExit("Directory selection required")
            self.save_dir = Path(directory)
        
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def initialize_state(self):
        self.visited = set()
        self.downloaded = set()
        self.download_count = 0
        self.visited_count = 0
        
        # Load previous state
        state_file = self.save_dir / 'scraper_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.visited = set(state.get('visited', []))
                    self.downloaded = set(state.get('downloaded', []))
                self.logger.info(f"Loaded state: {len(self.downloaded)} PDFs")
            except:
                pass
                
    def save_state(self):
        try:
            state_file = self.save_dir / 'scraper_state.json'
            with open(state_file, 'w') as f:
                json.dump({
                    'visited': list(self.visited),
                    'downloaded': list(self.downloaded),
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'stats': {
                        'total_visited': len(self.visited),
                        'total_downloaded': len(self.downloaded)
                    }
                }, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            
    def get_page_content(self, url):
        try:
            headers = {
                'User-Agent': UserAgent().random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    return response.content
                except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                    if attempt == 2:
                        raise
                    time.sleep(2)
                    
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
            
    def is_valid_pdf_url(self, url):
        pdf_check = (
            url.lower().endswith('.pdf') and
            url not in self.downloaded
        )
        
        if not pdf_check:
            return False
            
        # Check for spam indicators
        skip_indicators = [
            'advertisement', 'banner', 'track', 'pop-up', 'click', 'ad-', 
            'advert', 'promo', '/ads/', 'analytics', 'pixel'
        ]
        
        if any(indicator in url.lower() for indicator in skip_indicators):
            return False
            
        # Check for suspicious domains
        url_parts = urlparse(url)
        if any(spam in url_parts.netloc.lower() for spam in ['ad.', 'ads.', 'advert.', 'click']):
            return False
            
        return True
        
    def download_pdf(self, pdf_url):
        if pdf_url in self.downloaded:
            return False

        try:
            content = self.get_page_content(pdf_url)
            
            if not content:
                return False
                
            # Minimum size and PDF signature check
            if len(content) < 1000 or not content.startswith(b'%PDF'):
                self.logger.debug(f"Invalid PDF: {pdf_url}")
                return False

            # Extract filename
            url_path = urlparse(pdf_url).path
            file_name = os.path.basename(url_path)
            
            if not file_name.endswith('.pdf'):
                file_name += '.pdf'
            
            # Clean filename
            file_name = "".join(c for c in file_name if c.isalnum() or c in "._- ")
            if len(file_name) > 200:
                file_name = file_name[:195] + '.pdf'

            # Create unique filename
            file_path = self.save_dir / file_name
            counter = 1
            while file_path.exists():
                stem = file_path.stem.split('_')[0]
                file_path = self.save_dir / f"{stem}_{counter}.pdf"
                counter += 1

            # Save file
            with open(file_path, 'wb') as f:
                f.write(content)

            self.downloaded.add(pdf_url)
            self.download_count += 1
            self.logger.info(f"Downloaded ({self.download_count}): {file_name}")
            
            return True

        except Exception as e:
            self.logger.error(f"Error downloading {pdf_url}: {e}")
            return False
            
    def extract_links(self, url, html_content):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                if not href:
                    continue
                    
                full_url = urljoin(url, href)
                
                # Skip unwanted links
                if any(skip in full_url.lower() for skip in ['#', 'javascript:', 'mailto:', 'tel:']):
                    continue
                    
                if not full_url.startswith(('http://', 'https://')):
                    continue
                    
                if full_url == url:
                    continue
                    
                # Prioritize PDF links
                if full_url.lower().endswith('.pdf'):
                    links.insert(0, full_url)
                else:
                    links.append(full_url)
            
            # Prioritize academic content
            academic_indicators = ['paper', 'research', 'publication', 'study', 'science', 'journal', 'proceedings']
            for i, link in enumerate(links[:]):
                if any(indicator in link.lower() for indicator in academic_indicators):
                    if i > 0:
                        links.remove(link)
                        links.insert(0, link)
            
            return links
            
        except Exception as e:
            self.logger.error(f"Error extracting links from {url}: {e}")
            return []
            
    def process_page(self, url, depth=0):
        if depth > self.args.depth or url in self.visited:
            return

        self.visited.add(url)
        self.visited_count += 1
        
        self.logger.debug(f"Processing (depth {depth}): {url}")
        
        content = self.get_page_content(url)
        if not content:
            return

        # Handle direct PDF URL
        if self.is_valid_pdf_url(url):
            self.download_pdf(url)
            return

        # Extract links from HTML
        links = self.extract_links(url, content)
        
        # Process PDF links immediately
        pdf_links = [link for link in links if self.is_valid_pdf_url(link)]
        regular_links = [link for link in links if link not in pdf_links]
        
        # Download PDFs first
        for pdf_link in pdf_links:
            self.download_pdf(pdf_link)
            time.sleep(0.5)
            
        # Save state periodically
        if self.download_count % 10 == 0:
            self.save_state()
            
        # Process regular links for next depth
        if depth < self.args.depth and regular_links:
            with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
                for link in regular_links:
                    if link not in self.visited:
                        executor.submit(self.process_page, link, depth + 1)
                        time.sleep(0.5)
                        
    def show_completion_message(self):
        if IS_MACOS:
            subprocess.run(['open', str(self.save_dir)], check=False)

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "PDF Scraping Complete",
            f"PDF document scraping completed.\n\n"
            f"Pages visited: {len(self.visited)}\n"
            f"PDFs downloaded: {len(self.downloaded)}\n\n"
            f"Files saved to: {self.save_dir}"
        )
        root.destroy()
        
    def run(self):
        if self.args.version:
            print("PDF Document Scraper v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting PDF document scraping...")

        # Default start URLs for academic content
        start_urls = self.args.start_url or [
            "https://arxiv.org/list/cs.AI/recent",
            "https://papers.nips.cc/",
            "https://openai.com/research/",
            "https://research.google/pubs/",
            "https://ai.facebook.com/research/publications/",
            "https://www.microsoft.com/en-us/research/research-area/artificial-intelligence/"
        ]
        
        self.logger.info(f"Using {len(start_urls)} start URLs")

        try:
            # Process start URLs in parallel
            with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
                futures = []
                
                for url in start_urls:
                    self.logger.info(f"Starting with: {url}")
                    futures.append(executor.submit(self.process_page, url))
                
                for i, future in enumerate(futures, 1):
                    try:
                        future.result()
                        self.logger.info(f"Completed start URL {i}/{len(start_urls)}")
                        self.save_state()
                    except Exception as e:
                        self.logger.error(f"Error processing start URL {i}: {e}")

        except KeyboardInterrupt:
            self.logger.info("Scraping interrupted")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.save_state()
            
            self.logger.info(f"""
            Scraping Summary:
            - Pages visited: {len(self.visited)}
            - PDFs downloaded: {len(self.downloaded)}
            - Save location: {self.save_dir}
            """)
            
            if not self.args.quiet:
                self.show_completion_message()
            
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    scraper = PDFScraper()
    scraper.run()

if __name__ == "__main__":
    main()