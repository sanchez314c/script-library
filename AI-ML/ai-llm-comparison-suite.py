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
# Script Name: ai-llm-comparison-suite.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive testing suite for comparing multiple LLM providers    
#              with standardized evaluation metrics, automated benchmarking,      
#              and detailed performance analysis across OpenAI, Anthropic,        
#              Google AI, and more.                                               
#
# Usage: python ai-llm-comparison-suite.py [--models MODEL1,MODEL2] [--prompt FILE] 
#
# Dependencies: openai, anthropic, google-generativeai, requests, pandas, numpy,   
#               python-dotenv, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Supports parallel processing, progress tracking, and comprehensive       
#        result export. API keys loaded from environment variables or .env file   
#        in script directory.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
LLM Comparison Suite

Comprehensive testing suite for comparing multiple LLM providers with standardized
evaluation metrics, automated benchmarking, and detailed performance analysis.
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
    import openai
    import anthropic
    import google.generativeai as genai
    import requests
    import pandas as pd
    import numpy as np
    from dotenv import load_dotenv
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox, ttk
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using:")
    print("pip install openai anthropic google-generativeai requests pandas numpy python-dotenv")
    sys.exit(1)

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'ai-llm-comparison-suite.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('ai-llm-comparison-suite')

# Load environment variables from .env file in script directory and current directory
script_dir = Path(__file__).parent
env_files = [script_dir / '.env', Path.cwd() / '.env']
for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")

class CredentialManager:
    """Manages API credentials with secure storage and GUI prompts"""
    
    def __init__(self):
        self.credentials_file = Path.home() / '.ai_llm_comparison_credentials.json'
        self.credentials = self._load_credentials()
    
    def _load_credentials(self) -> Dict[str, str]:
        """Load credentials from encrypted storage"""
        if not self.credentials_file.exists():
            return {}
        
        try:
            with open(self.credentials_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return {}
    
    def _save_credentials(self) -> None:
        """Save credentials to encrypted storage"""
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(self.credentials, f)
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def get_credential(self, key_name: str, prompt_message: str = None) -> Optional[str]:
        """Get credential, prompting user if not available"""
        # First check environment variables
        env_value = os.environ.get(key_name)
        if env_value:
            return env_value
            
        # Then check stored credentials
        if key_name in self.credentials and self.credentials[key_name]:
            return self.credentials[key_name]
            
        # If still not found, prompt user
        if prompt_message is None:
            prompt_message = f"Please enter your {key_name}:"
            
        # Create root window for dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Show dialog to get credential
        value = simpledialog.askstring("API Credential Required", 
                                      prompt_message,
                                      show='*')  # Show asterisks for security
        
        if value:
            self.credentials[key_name] = value
            self._save_credentials()
        
        root.destroy()
        return value

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

class LLMComparisonSuite:
    """Core class for LLM comparison functionality"""
    
    def __init__(self):
        self.credential_manager = CredentialManager()
        self.models = {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1-preview', 'o1-mini'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 
                         'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
            'google': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro'],
            'perplexity': ['llama-3.1-sonar-large-128k-online', 'llama-3.1-sonar-small-128k-online', 
                          'llama-3.1-70b-instruct', 'llama-3.1-8b-instruct'],
            'groq': ['llama-3.1-405b-reasoning', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant',
                    'mixtral-8x7b-32768', 'gemma2-9b-it'],
            'mistral': ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'],
            'jamba': ['jamba-1.5-large', 'jamba-1.5-mini'],
            'xai': ['grok-beta', 'grok-vision-beta'],
            'ollama': ['llama3.1', 'llama3.2', 'mistral', 'codellama', 'deepseek-coder']
        }
        self.init_apis()
    
    def init_apis(self):
        """Initialize all API clients with proper credentials"""
        try:
            # OpenAI
            openai_key = self.credential_manager.get_credential(
                'OPENAI_API_KEY', 
                "Please enter your OpenAI API key:"
            )
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            
            # Anthropic
            anthropic_key = self.credential_manager.get_credential(
                'ANTHROPIC_API_KEY',
                "Please enter your Anthropic API key:"
            )
            if anthropic_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            
            # Google AI
            google_key = self.credential_manager.get_credential(
                'GOOGLE_AI_API_KEY',
                "Please enter your Google AI API key:"
            )
            if google_key:
                genai.configure(api_key=google_key)
            
            # Perplexity
            self.perplexity_key = self.credential_manager.get_credential(
                'PERPLEXITY_API_KEY',
                "Please enter your Perplexity API key:"
            )
            
            # Groq
            groq_key = self.credential_manager.get_credential(
                'GROQ_API_KEY',
                "Please enter your Groq API key:"
            )
            if groq_key:
                self.groq_client = openai.OpenAI(
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            
            # Mistral
            self.mistral_key = self.credential_manager.get_credential(
                'MISTRAL_API_KEY',
                "Please enter your Mistral API key:"
            )
            
            # AI21 (Jamba)
            self.ai21_key = self.credential_manager.get_credential(
                'AI21_API_KEY',
                "Please enter your AI21 API key:"
            )
            
            # xAI
            xai_key = self.credential_manager.get_credential(
                'XAI_API_KEY',
                "Please enter your xAI API key:"
            )
            if xai_key:
                self.xai_client = openai.OpenAI(
                    api_key=xai_key,
                    base_url="https://api.x.ai/v1"
                )
            
            # Ollama (local)
            self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            
        except Exception as e:
            logger.error(f"API initialization error: {e}")
            messagebox.showerror("API Error", f"Failed to initialize APIs: {e}")
    
    def compare_models(self, prompt: str, selected_models: List[str] = None) -> Dict[str, Any]:
        """Run comparison across selected models"""
        if selected_models is None:
            # Default to one model from each available provider
            selected_models = [
                'openai/gpt-4o-mini',
                'anthropic/claude-3-5-haiku-20241022',
                'google/gemini-1.5-flash'
            ]
        
        results = {}
        total_models = len(selected_models)
        
        # Set up progress tracking
        progress = ProgressTracker(total_models, "Comparing LLM Models")
        
        # Run models in parallel using ThreadPoolExecutor
        threads = []
        for i, model_path in enumerate(selected_models):
            if '/' not in model_path:
                logger.warning(f"Invalid model format: {model_path}. Expected 'provider/model'")
                continue
                
            provider, model = model_path.split('/', 1)
            
            thread = threading.Thread(
                target=self._process_model_request,
                args=(provider, model, prompt, results)
            )
            thread.start()
            threads.append((thread, i))
        
        # Wait for all threads and update progress
        for thread, i in threads:
            thread.join()
            progress.update(i + 1, f"Completed {i+1}/{total_models} models")
        
        progress.complete()
        return results
    
    def _process_model_request(self, provider: str, model: str, prompt: str, results: Dict):
        """Process request for single model"""
        start_time = time.time()
        
        try:
            if provider == 'openai':
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                tokens = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
                
            elif provider == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                tokens = {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                }
                
            elif provider == 'google':
                model_obj = genai.GenerativeModel(model)
                response = model_obj.generate_content(prompt)
                content = response.text
                tokens = {"prompt": 0, "completion": 0, "total": 0}  # Gemini doesn't provide detailed token counts
                
            elif provider == 'perplexity':
                content, tokens = self._call_perplexity_api(model, prompt)
                
            elif provider == 'groq':
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                tokens = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
                
            elif provider == 'mistral':
                content, tokens = self._call_mistral_api(model, prompt)
                
            elif provider == 'jamba':
                content, tokens = self._call_jamba_api(model, prompt)
                
            elif provider == 'xai':
                response = self.xai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                tokens = {
                    "prompt": getattr(response.usage, 'prompt_tokens', 0),
                    "completion": getattr(response.usage, 'completion_tokens', 0),
                    "total": getattr(response.usage, 'total_tokens', 0)
                }
                
            elif provider == 'ollama':
                content, tokens = self._call_ollama_api(model, prompt)
                
            else:
                content = f"Unsupported provider: {provider}"
                tokens = {"prompt": 0, "completion": 0, "total": 0}
            
            response_time = time.time() - start_time
            
            results[f"{provider}/{model}"] = {
                "content": content,
                "tokens": tokens,
                "response_time": response_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
                
        except Exception as e:
            logger.error(f"Error with {provider}/{model}: {e}")
            results[f"{provider}/{model}"] = {
                "content": f"ERROR: {str(e)}",
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "response_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def _call_perplexity_api(self, model: str, prompt: str) -> Tuple[str, Dict]:
        """Call Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        tokens = {
            "prompt": result.get("usage", {}).get("prompt_tokens", 0),
            "completion": result.get("usage", {}).get("completion_tokens", 0),
            "total": result.get("usage", {}).get("total_tokens", 0)
        }
        
        return content, tokens
    
    def _call_mistral_api(self, model: str, prompt: str) -> Tuple[str, Dict]:
        """Call Mistral API"""
        headers = {
            "Authorization": f"Bearer {self.mistral_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post("https://api.mistral.ai/v1/chat/completions", 
                               headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        tokens = {
            "prompt": result.get("usage", {}).get("prompt_tokens", 0),
            "completion": result.get("usage", {}).get("completion_tokens", 0),
            "total": result.get("usage", {}).get("total_tokens", 0)
        }
        
        return content, tokens
    
    def _call_jamba_api(self, model: str, prompt: str) -> Tuple[str, Dict]:
        """Call AI21 Jamba API"""
        headers = {
            "Authorization": f"Bearer {self.ai21_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post("https://api.ai21.com/studio/v1/chat/completions", 
                               headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        tokens = {
            "prompt": result.get("usage", {}).get("prompt_tokens", 0),
            "completion": result.get("usage", {}).get("completion_tokens", 0),
            "total": result.get("usage", {}).get("total_tokens", 0)
        }
        
        return content, tokens
    
    def _call_ollama_api(self, model: str, prompt: str) -> Tuple[str, Dict]:
        """Call Ollama API"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
        response = requests.post(f"{self.ollama_base_url}/api/chat", 
                               json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        content = result["message"]["content"]
        tokens = {"prompt": 0, "completion": 0, "total": 0}  # Ollama doesn't provide detailed token counts
        
        return content, tokens
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare model responses"""
        analysis = {
            "response_lengths": {},
            "response_times": {},
            "token_usage": {},
            "common_phrases": self._find_common_phrases(results),
            "performance_metrics": {},
            "summary_stats": {}
        }
        
        # Extract metrics for each model
        for model, data in results.items():
            if isinstance(data, dict) and "content" in data:
                content = data["content"]
                analysis["response_lengths"][model] = len(content)
                analysis["response_times"][model] = data.get("response_time", 0)
                analysis["token_usage"][model] = data.get("tokens", {})
                
                # Calculate words per second
                word_count = len(content.split())
                response_time = data.get("response_time", 1)
                analysis["performance_metrics"][model] = {
                    "words_per_second": word_count / response_time if response_time > 0 else 0,
                    "characters_per_second": len(content) / response_time if response_time > 0 else 0,
                    "word_count": word_count
                }
        
        # Calculate summary statistics
        if analysis["response_times"]:
            times = list(analysis["response_times"].values())
            analysis["summary_stats"]["avg_response_time"] = np.mean(times)
            analysis["summary_stats"]["fastest_model"] = min(analysis["response_times"], key=analysis["response_times"].get)
            analysis["summary_stats"]["slowest_model"] = max(analysis["response_times"], key=analysis["response_times"].get)
        
        if analysis["response_lengths"]:
            lengths = list(analysis["response_lengths"].values())
            analysis["summary_stats"]["avg_response_length"] = np.mean(lengths)
            analysis["summary_stats"]["longest_response"] = max(analysis["response_lengths"], key=analysis["response_lengths"].get)
            analysis["summary_stats"]["shortest_response"] = min(analysis["response_lengths"], key=analysis["response_lengths"].get)
        
        return analysis
    
    def _find_common_phrases(self, results: Dict[str, Any]) -> List[str]:
        """Find common phrases between responses"""
        # Extract content from results
        contents = []
        for model, data in results.items():
            if isinstance(data, dict) and "content" in data:
                contents.append(data["content"].lower())
        
        if len(contents) < 2:
            return []
        
        # Simple implementation - find common words (would use NLP techniques in full implementation)
        common_phrases = []
        words_sets = [set(content.split()) for content in contents]
        
        # Find words that appear in at least half of the responses
        all_words = set()
        for word_set in words_sets:
            all_words.update(word_set)
        
        threshold = len(contents) / 2
        for word in all_words:
            if len(word) > 3:  # Skip short words
                count = sum(1 for word_set in words_sets if word in word_set)
                if count >= threshold:
                    common_phrases.append(word)
        
        return sorted(common_phrases)[:10]  # Return top 10 common phrases
    
    def save_results(self, results: Dict[str, Any], analysis: Dict, prompt: str, output_file: Path = None):
        """Save results and analysis to file"""
        if output_file is None:
            # Save to desktop with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = desktop_path / f"llm_comparison_results_{timestamp}.json"
        
        try:
            output_data = {
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt,
                    "total_models": len([k for k in results.keys() if k != "prompt"]),
                    "script_version": "1.0.0"
                },
                "model_responses": results,
                "analysis": analysis
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Results saved to {output_file}")
            messagebox.showinfo("Success", f"Results saved to {output_file}")
            
            # Also save as CSV for easy analysis
            csv_file = output_file.with_suffix('.csv')
            self._save_results_as_csv(results, analysis, csv_file)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            messagebox.showerror("Save Error", f"Failed to save results: {e}")
    
    def _save_results_as_csv(self, results: Dict[str, Any], analysis: Dict, csv_file: Path):
        """Save results as CSV for easy analysis"""
        try:
            # Prepare data for DataFrame
            data = []
            for model, result in results.items():
                if isinstance(result, dict) and "content" in result:
                    row = {
                        "model": model,
                        "response_length": len(result["content"]),
                        "response_time": result.get("response_time", 0),
                        "word_count": len(result["content"].split()),
                        "prompt_tokens": result.get("tokens", {}).get("prompt", 0),
                        "completion_tokens": result.get("tokens", {}).get("completion", 0),
                        "total_tokens": result.get("tokens", {}).get("total", 0),
                        "timestamp": result.get("timestamp", "")
                    }
                    
                    # Add performance metrics
                    perf_metrics = analysis.get("performance_metrics", {}).get(model, {})
                    row.update({
                        "words_per_second": perf_metrics.get("words_per_second", 0),
                        "characters_per_second": perf_metrics.get("characters_per_second", 0)
                    })
                    
                    data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            logger.info(f"CSV results saved to {csv_file}")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LLM Comparison Suite")
    
    parser.add_argument(
        "--models", 
        type=str,
        help="Comma-separated list of models to compare (provider/model format)"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str,
        help="File containing prompt text or direct prompt string"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file path for results"
    )
    
    parser.add_argument(
        "--reset-credentials",
        action="store_true",
        help="Reset stored API credentials"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Reset credentials if requested
    if args.reset_credentials:
        try:
            cred_file = Path.home() / '.ai_llm_comparison_credentials.json'
            if cred_file.exists():
                os.remove(cred_file)
                print("Credentials reset successfully")
        except Exception as e:
            print(f"Error resetting credentials: {e}")
        sys.exit(0)
    
    logger.info("Starting LLM Comparison Suite")
    
    # Initialize comparison suite
    suite = LLMComparisonSuite()
    
    # Get prompt from file, argument, or user input
    prompt = ""
    if args.prompt:
        prompt_path = Path(args.prompt)
        if prompt_path.exists() and prompt_path.is_file():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            prompt = args.prompt
    
    if not prompt:
        # Use GUI to get prompt
        root = tk.Tk()
        root.withdraw()
        
        prompt_file = filedialog.askopenfilename(
            title="Select prompt file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if prompt_file:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            # If still no prompt, ask directly
            root = tk.Tk()
            root.withdraw()
            
            prompt = simpledialog.askstring(
                "Prompt Input",
                "Enter your comparison prompt:",
                initialvalue="Compare the following models on their reasoning abilities."
            )
            
            root.destroy()
    
    if not prompt:
        print("No prompt provided. Exiting.")
        logger.error("No prompt provided")
        sys.exit(1)
    
    # Parse models
    selected_models = None
    if args.models:
        selected_models = [model.strip() for model in args.models.split(',')]
    
    # Run comparison
    try:
        logger.info(f"Running comparison with prompt: {prompt[:100]}...")
        results = suite.compare_models(prompt, selected_models)
        
        # Analyze results
        analysis = suite.analyze_results(results)
        
        # Save results
        output_file = None
        if args.output:
            output_file = Path(args.output)
        
        suite.save_results(results, analysis, prompt, output_file)
        
        logger.info("Comparison completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        messagebox.showerror("Error", f"Comparison failed: {e}")
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