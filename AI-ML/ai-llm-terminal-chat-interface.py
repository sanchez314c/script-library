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
# Script Name: ai-llm-terminal-chat-interface.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Universal terminal interface for interacting with multiple LLM APIs    
#              including OpenAI, Anthropic, Google AI, Perplexity, Groq, Mistral,      
#              Jamba, xAI, and local Ollama. Features conversation management,        
#              streaming responses, and comprehensive multi-provider support.                                               
#
# Usage: python ai-llm-terminal-chat-interface.py [--model provider/model] [--system file] 
#
# Dependencies: openai, anthropic, google-generativeai, requests, rich, python-dotenv,   
#               tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: API keys loaded from environment variables or .env file. Supports all major       
#        foundation model providers with unified interface and local Ollama integration.   
#        in script directory.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
Universal Terminal Chat Interface for Foundation Models

A comprehensive terminal-based chat interface that supports all major LLM providers
including OpenAI, Anthropic, Google AI, Perplexity, Groq, Mistral, Jamba, xAI, and Ollama.
Features conversation management, streaming responses, and secure credential handling.
"""

# Standard library imports
import os
import sys
import json
import time
import signal
import argparse
import logging
import readline
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Third-party imports
try:
    import openai
    import anthropic
    import google.generativeai as genai
    import requests
    from rich.console import Console
    from rich.markdown import Markdown
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using:")
    print("pip install openai anthropic google-generativeai requests rich python-dotenv")
    sys.exit(1)

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'ai-llm-terminal-chat-interface.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('ai-llm-terminal-chat-interface')

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
        self.credentials_file = Path.home() / '.ai_llm_terminal_chat_credentials.json'
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
    """Handles file selection with macOS native dialogs"""
    
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

class ConversationHistory:
    """Manages conversation history and persistence"""
    
    def __init__(self, history_file: Optional[Path] = None):
        self.messages = []
        self.history_file = history_file or Path.home() / '.ai_llm_terminal_chat_history.json'
        self.token_count = 0
        
    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        self.token_count += tokens
        
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []
        self.token_count = 0
        
    def save_history(self):
        """Save conversation history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "token_count": self.token_count,
                    "timestamp": time.time()
                }, f)
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
            
    def load_history(self) -> bool:
        """Load conversation history from file"""
        if not self.history_file.exists():
            return False
            
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                self.token_count = data.get("token_count", 0)
            return True
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            return False
            
    def get_formatted_messages(self):
        """Get messages in format required by API calls"""
        return self.messages.copy()

class ChatProvider:
    """Base class for chat API providers"""
    
    def __init__(self, credential_manager):
        self.credential_manager = credential_manager
        
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to the chat API"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from the chat API"""
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(ChatProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "OPENAI_API_KEY",
            "Please enter your OpenAI API key:"
        )
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models"""
        return ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o1-preview", "o1-mini"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            return {
                "content": response.choices[0].message.content,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from OpenAI API"""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            
            content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    content += content_chunk
                    yield content_chunk, content
                    
        except Exception as e:
            logger.error(f"OpenAI API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class AnthropicProvider(ChatProvider):
    """Anthropic API provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "ANTHROPIC_API_KEY",
            "Please enter your Anthropic API key:"
        )
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models"""
        return ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", 
                "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Anthropic API"""
        try:
            # Extract system message if present
            system_message = ""
            filtered_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    filtered_messages.append(msg)
            
            kwargs = {
                "model": model,
                "max_tokens": 4000,
                "messages": filtered_messages
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            response = self.client.messages.create(**kwargs)
            return {
                "content": response.content[0].text,
                "tokens": {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Anthropic API"""
        try:
            # Extract system message if present
            system_message = ""
            filtered_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    filtered_messages.append(msg)
            
            kwargs = {
                "model": model,
                "max_tokens": 4000,
                "messages": filtered_messages,
                "stream": True
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            stream = self.client.messages.create(**kwargs)
            
            content = ""
            for chunk in stream:
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    content_chunk = chunk.delta.text
                    content += content_chunk
                    yield content_chunk, content
                    
        except Exception as e:
            logger.error(f"Anthropic API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class GoogleAIProvider(ChatProvider):
    """Google AI (Gemini) provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "GOOGLE_AI_API_KEY",
            "Please enter your Google AI API key:"
        )
        if self.api_key:
            genai.configure(api_key=self.api_key)
            
    def get_available_models(self) -> List[str]:
        """Get list of available Google AI models"""
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Google AI API"""
        try:
            model_obj = genai.GenerativeModel(model)
            
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            response = model_obj.generate_content(prompt)
            return {
                "content": response.text,
                "tokens": {
                    "prompt": 0,  # Gemini doesn't provide detailed token counts
                    "completion": 0,
                    "total": 0
                }
            }
        except Exception as e:
            logger.error(f"Google AI API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Google AI API"""
        try:
            model_obj = genai.GenerativeModel(model)
            prompt = self._convert_messages_to_prompt(messages)
            
            response = model_obj.generate_content(prompt, stream=True)
            
            content = ""
            for chunk in response:
                if chunk.text:
                    content += chunk.text
                    yield chunk.text, content
                    
        except Exception as e:
            logger.error(f"Google AI API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to Gemini prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

class PerplexityProvider(ChatProvider):
    """Perplexity API provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "PERPLEXITY_API_KEY",
            "Please enter your Perplexity API key:"
        )
        self.base_url = "https://api.perplexity.ai"
            
    def get_available_models(self) -> List[str]:
        """Get list of available Perplexity models"""
        return ["llama-3.1-sonar-small-128k-online", "llama-3.1-sonar-large-128k-online", 
                "llama-3.1-sonar-huge-128k-online", "llama-3.1-8b-instruct", "llama-3.1-70b-instruct"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Perplexity API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "tokens": {
                    "prompt": result.get("usage", {}).get("prompt_tokens", 0),
                    "completion": result.get("usage", {}).get("completion_tokens", 0),
                    "total": result.get("usage", {}).get("total_tokens", 0)
                }
            }
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Perplexity API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "stream": True
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line)
                            if chunk_data["choices"][0]["delta"].get("content"):
                                content_chunk = chunk_data["choices"][0]["delta"]["content"]
                                content += content_chunk
                                yield content_chunk, content
                        except json.JSONDecodeError:
                            continue
                    
        except Exception as e:
            logger.error(f"Perplexity API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class GroqProvider(ChatProvider):
    """Groq API provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "GROQ_API_KEY",
            "Please enter your Groq API key:"
        )
        if self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
    def get_available_models(self) -> List[str]:
        """Get list of available Groq models"""
        return ["llama-3.1-405b-reasoning", "llama-3.1-70b-versatile", "llama-3.1-8b-instant",
                "mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            return {
                "content": response.choices[0].message.content,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Groq API"""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            
            content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    content += content_chunk
                    yield content_chunk, content
                    
        except Exception as e:
            logger.error(f"Groq API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class MistralProvider(ChatProvider):
    """Mistral AI provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "MISTRAL_API_KEY",
            "Please enter your Mistral API key:"
        )
        self.base_url = "https://api.mistral.ai/v1"
            
    def get_available_models(self) -> List[str]:
        """Get list of available Mistral models"""
        return ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
                "codestral-latest", "mistral-embed"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Mistral API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "tokens": {
                    "prompt": result.get("usage", {}).get("prompt_tokens", 0),
                    "completion": result.get("usage", {}).get("completion_tokens", 0),
                    "total": result.get("usage", {}).get("total_tokens", 0)
                }
            }
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Mistral API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "stream": True
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line)
                            if chunk_data["choices"][0]["delta"].get("content"):
                                content_chunk = chunk_data["choices"][0]["delta"]["content"]
                                content += content_chunk
                                yield content_chunk, content
                        except json.JSONDecodeError:
                            continue
                    
        except Exception as e:
            logger.error(f"Mistral API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class JambaProvider(ChatProvider):
    """AI21 Jamba provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "AI21_API_KEY",
            "Please enter your AI21 API key:"
        )
        self.base_url = "https://api.ai21.com/studio/v1"
            
    def get_available_models(self) -> List[str]:
        """Get list of available Jamba models"""
        return ["jamba-1.5-large", "jamba-1.5-mini", "jamba-instruct"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Jamba API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "tokens": {
                    "prompt": result.get("usage", {}).get("prompt_tokens", 0),
                    "completion": result.get("usage", {}).get("completion_tokens", 0),
                    "total": result.get("usage", {}).get("total_tokens", 0)
                }
            }
        except Exception as e:
            logger.error(f"Jamba API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Jamba API"""
        # Note: Streaming may not be available for all Jamba models
        try:
            result = self.send_message(messages, model)
            if "error" in result:
                yield f"Error: {result['error']}", f"Error: {result['error']}"
            else:
                content = result["content"]
                # Simulate streaming by yielding chunks
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    yield chunk, content[:i+chunk_size]
                    time.sleep(0.05)  # Small delay to simulate streaming
                    
        except Exception as e:
            logger.error(f"Jamba API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class XAIProvider(ChatProvider):
    """xAI Grok provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.api_key = self.credential_manager.get_credential(
            "XAI_API_KEY",
            "Please enter your xAI API key:"
        )
        if self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
            
    def get_available_models(self) -> List[str]:
        """Get list of available xAI models"""
        return ["grok-beta", "grok-vision-beta"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to xAI API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
            )
            return {
                "content": response.choices[0].message.content,
                "tokens": {
                    "prompt": getattr(response.usage, 'prompt_tokens', 0),
                    "completion": getattr(response.usage, 'completion_tokens', 0),
                    "total": getattr(response.usage, 'total_tokens', 0)
                }
            }
        except Exception as e:
            logger.error(f"xAI API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from xAI API"""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            
            content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    content += content_chunk
                    yield content_chunk, content
                    
        except Exception as e:
            logger.error(f"xAI API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class OllamaProvider(ChatProvider):
    """Ollama local model provider implementation"""
    
    def __init__(self, credential_manager):
        super().__init__(credential_manager)
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return ["llama3.1", "llama3.2", "mistral", "codellama", "deepseek-coder"]
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")
            return ["llama3.1", "llama3.2", "mistral", "codellama", "deepseek-coder"]
        
    def send_message(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a message to Ollama API"""
        try:
            data = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/chat", json=data)
            response.raise_for_status()
            result = response.json()
            
            return {
                "content": result["message"]["content"],
                "tokens": {
                    "prompt": 0,  # Ollama doesn't provide detailed token counts
                    "completion": 0,
                    "total": 0
                }
            }
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return {"error": str(e)}
            
    def stream_message(self, messages: List[Dict[str, str]], model: str):
        """Stream a message from Ollama API"""
        try:
            data = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            
            response = requests.post(f"{self.base_url}/api/chat", json=data, stream=True)
            response.raise_for_status()
            
            content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        if chunk_data.get("message", {}).get("content"):
                            content_chunk = chunk_data["message"]["content"]
                            content += content_chunk
                            yield content_chunk, content
                        
                        if chunk_data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                    
        except Exception as e:
            logger.error(f"Ollama API streaming error: {e}")
            yield f"Error: {e}", f"Error: {e}"

class TerminalChat:
    """Main terminal chat interface class"""
    
    def __init__(self):
        # Initialize components
        self.credential_manager = CredentialManager()
        self.conversation = ConversationHistory()
        self.console = Console()
        
        # Set up providers
        self.providers = {
            "openai": OpenAIProvider(self.credential_manager),
            "anthropic": AnthropicProvider(self.credential_manager),
            "google": GoogleAIProvider(self.credential_manager),
            "perplexity": PerplexityProvider(self.credential_manager),
            "groq": GroqProvider(self.credential_manager),
            "mistral": MistralProvider(self.credential_manager),
            "jamba": JambaProvider(self.credential_manager),
            "xai": XAIProvider(self.credential_manager),
            "ollama": OllamaProvider(self.credential_manager)
        }
        
        # Default settings
        self.current_provider = "openai"
        self.current_model = "gpt-4o-mini"
        self.system_prompt = "You are a helpful assistant."
        
        # Register signal handlers for graceful exit
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signal (Ctrl+C)"""
        print("\nExiting gracefully...")
        self.conversation.save_history()
        sys.exit(0)
        
    def initialize(self, args: Optional[argparse.Namespace] = None):
        """Initialize chat with command line arguments"""
        if args:
            # Set model if provided
            if args.model:
                if '/' in args.model:
                    provider, model = args.model.split('/', 1)
                    if provider in self.providers and model in self.providers[provider].get_available_models():
                        self.current_provider = provider
                        self.current_model = model
                else:
                    # Assume OpenAI model if no provider specified
                    if args.model in self.providers["openai"].get_available_models():
                        self.current_model = args.model
            
            # Load system prompt if provided
            if args.system:
                system_path = Path(args.system)
                if system_path.exists() and system_path.is_file():
                    with open(system_path, 'r') as f:
                        self.system_prompt = f.read().strip()
        
        # Add system prompt to conversation
        self.conversation.add_message("system", self.system_prompt)
        
        # Try to load conversation history
        if args and args.continue_conversation:
            self.conversation.load_history()
        
    def run(self):
        """Run the chat interface"""
        # Print welcome banner
        self._print_banner()
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() in ["/exit", "/quit", "/q"]:
                    print("Goodbye!")
                    self.conversation.save_history()
                    break
                    
                if user_input.lower() == "/help":
                    self._print_help()
                    continue
                    
                if user_input.lower() == "/clear":
                    self.conversation.clear_history()
                    self.conversation.add_message("system", self.system_prompt)
                    print("Conversation history cleared.")
                    continue
                    
                if user_input.lower().startswith("/model"):
                    self._handle_model_command(user_input)
                    continue
                    
                if user_input.lower() == "/history":
                    self._print_history()
                    continue
                    
                if not user_input:
                    continue
                
                # Add user message to history
                self.conversation.add_message("user", user_input)
                
                # Send message to provider
                provider = self.providers.get(self.current_provider)
                if not provider:
                    print(f"Error: Provider {self.current_provider} not available")
                    continue
                
                # Print assistant label
                print("\nAssistant: ", end="", flush=True)
                
                # Stream response
                full_content = ""
                for chunk, content in provider.stream_message(
                    self.conversation.get_formatted_messages(), 
                    self.current_model
                ):
                    print(chunk, end="", flush=True)
                    full_content = content
                
                print("\n")  # End line after response
                
                # Add assistant response to history
                self.conversation.add_message("assistant", full_content)
                
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\nAn error occurred: {e}")
    
    def _print_banner(self):
        """Print welcome banner"""
        model_info = f"{self.current_provider.capitalize()}/{self.current_model}"
        self.console.print(f"\n[bold cyan]Universal LLM Terminal Chat Interface v1.0.0[/bold cyan]")
        self.console.print(f"[cyan]Using model: {model_info}[/cyan]")
        self.console.print("[cyan]Type /help for available commands[/cyan]")
        self.console.print(f"[dim]Logs saved to: {log_file}[/dim]")
    
    def _print_help(self):
        """Print help information"""
        help_text = """
Available commands:
  /help       - Show this help message
  /exit, /quit, /q - Exit the chat
  /clear      - Clear conversation history
  /model      - Show current model
  /model list - List available models for all providers
  /model [provider/name] - Switch to specified model
  /history    - Show conversation history

Supported providers:
  - openai: OpenAI GPT models
  - anthropic: Anthropic Claude models
  - google: Google Gemini models
  - perplexity: Perplexity Sonar models
  - groq: Groq LLaMA and Mixtral models
  - mistral: Mistral AI models
  - jamba: AI21 Jamba models
  - xai: xAI Grok models
  - ollama: Local Ollama models
"""
        self.console.print(Markdown(help_text))
    
    def _handle_model_command(self, command: str):
        """Handle model-related commands"""
        parts = command.split()
        
        # Just /model - show current model
        if len(parts) == 1:
            self.console.print(f"Current model: [bold]{self.current_provider}/{self.current_model}[/bold]")
            return
            
        # /model list - list available models
        if parts[1].lower() == "list":
            self.console.print("\nAvailable models:")
            for provider_name, provider in self.providers.items():
                try:
                    models = provider.get_available_models()
                    self.console.print(f"\n[bold]{provider_name.capitalize()}[/bold]:")
                    for model in models:
                        self.console.print(f"  - {model}")
                except Exception as e:
                    self.console.print(f"\n[bold]{provider_name.capitalize()}[/bold]: [red]Error loading models: {e}[/red]")
            return
            
        # /model provider/name - switch model
        if "/" in parts[1]:
            provider, model = parts[1].split("/", 1)
            if provider in self.providers:
                try:
                    available_models = self.providers[provider].get_available_models()
                    if model in available_models:
                        self.current_provider = provider
                        self.current_model = model
                        self.console.print(f"Switched to model: [bold]{provider}/{model}[/bold]")
                    else:
                        self.console.print(f"Invalid model. Available {provider} models: {', '.join(available_models)}")
                except Exception as e:
                    self.console.print(f"Error accessing {provider} models: {e}")
            else:
                self.console.print(f"Invalid provider. Available providers: {', '.join(self.providers.keys())}")
            return
            
        # Assume OpenAI model if no provider specified
        model = parts[1]
        try:
            available_models = self.providers["openai"].get_available_models()
            if model in available_models:
                self.current_provider = "openai"
                self.current_model = model
                self.console.print(f"Switched to model: [bold]openai/{model}[/bold]")
            else:
                self.console.print(f"Invalid model. Use '/model list' to see available models.")
        except Exception as e:
            self.console.print(f"Error accessing OpenAI models: {e}")
    
    def _print_history(self):
        """Print conversation history"""
        if not self.conversation.messages:
            self.console.print("No conversation history.")
            return
            
        self.console.print("\n[bold]Conversation History:[/bold]")
        for msg in self.conversation.messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            
            if role == "System":
                self.console.print(f"\n[dim italic]System: {content}[/dim italic]")
            elif role == "User":
                self.console.print(f"\n[bold]You:[/bold] {content}")
            elif role == "Assistant":
                self.console.print(f"\n[bold cyan]Assistant:[/bold cyan] {content}")
        
        # Print token usage information
        self.console.print(f"\n[dim]Token count: {self.conversation.token_count}[/dim]")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Universal LLM Terminal Chat Interface")
    
    parser.add_argument(
        "--model", 
        type=str,
        help="Model to use (e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-20241022')"
    )
    
    parser.add_argument(
        "--system", 
        type=str,
        help="Path to file containing system prompt"
    )
    
    parser.add_argument(
        "--continue", 
        dest="continue_conversation",
        action="store_true",
        help="Continue previous conversation"
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
            cred_file = Path.home() / '.ai_llm_terminal_chat_credentials.json'
            if cred_file.exists():
                os.remove(cred_file)
                print("Credentials reset successfully")
        except Exception as e:
            print(f"Error resetting credentials: {e}")
        sys.exit(0)
    
    # Initialize and run chat interface
    chat = TerminalChat()
    chat.initialize(args)
    chat.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        print(f"\nCritical error: {e}")
        sys.exit(1)