#!/usr/bin/env python3

####################################################################################
#                                                                                  #
#    ██████╗ ██████╗ ██████╗ ███████╗██╗  ██╗    ████████╗ ██████╗                #
#   ██╔════╝██╔═══██╗██╔══██╗██╔════╝╚██╗██╔╝    ╚══██╔══╝██╔═══██╗               #
#   ██║     ██║   ██║██║  ██║█████╗   ╚███╔╝        ██║   ██║   ██║               #
#   ██║     ██║   ██║██║  ██║██╔══╝   ██╔██╗        ██║   ██║   ██║               #
#   ╚██████╗╚██████╔╝██████╔╝███████╗██╔╝ ██╗       ██║   ╚██████╔╝               #
#    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝                #
#                                                                                  #
#     ██████╗ ██████╗ ███████╗███╗   ██╗ █████╗ ██╗                               #
#    ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██║                               #
#    ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║                               #
#    ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║                               #
#    ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║                               #
#     ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝                               #
#                                                                                  #
####################################################################################
#
# Script Name: openai-codex-to-chatgpt-converter.py                                                 
# 
# Author: Claude Code Assistant  
#                                              
# Date Created: 2025-05-30                                                       
#
# Last Modified: 2025-05-30                                                      
#
# Description: OpenAI Codex to ChatGPT Format Converter - Converts OpenAI Codex
#              conversation exports (JSON format) to OpenAI ChatGPT compatible 
#              JSON format with comprehensive error handling and structure validation.
#
# Version: 1.0.0
#
####################################################################################

import sys
import os
import subprocess
import json
import time
import platform
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import hashlib
import uuid

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'tkinter',
        'python-dateutil'
    ]
    
    for package in dependencies:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--upgrade", "--quiet"
                ])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                return False
    return True

# Install dependencies before importing
if not install_dependencies():
    print("Failed to install required dependencies. Exiting.")
    sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dateutil import parser as date_parser

class CodexToOpenAIConverter:
    """Advanced OpenAI Codex to OpenAI ChatGPT format converter."""
    
    def __init__(self):
        """Initialize the converter."""
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"openai_codex_to_chatgpt_converter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'conversations_converted': 0,
            'skipped_files': 0,
            'errors_encountered': 0,
            'processing_time': 0
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"OpenAI Codex to ChatGPT Converter Log - {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
    
    def _log(self, message: str, level: str = "INFO"):
        """Log message to file and optionally print to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass  # Fail silently for logging issues
        
        print(f"[{level}] {message}")
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            'system': platform.system(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def select_source_directory(self) -> Optional[str]:
        """Select source directory containing OpenAI Codex files."""
        self._log("Opening source directory selection dialog")
        
        try:
            root = tk.Tk()
            root.withdraw()
            
            source_dir = filedialog.askdirectory(
                title="Select Directory Containing OpenAI Codex JSON Files",
                initialdir=str(Path.home())
            )
            
            if source_dir:
                self._log(f"Selected source directory: {source_dir}")
                root.destroy()
                return source_dir
            
            root.destroy()
            return None
            
        except Exception as e:
            self._log(f"Error selecting source directory: {e}", "ERROR")
            return None
    
    def select_output_directory(self) -> Optional[str]:
        """Select output directory for converted files."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Converted Files",
                initialdir=str(self.desktop_path)
            )
            
            if output_dir:
                self._log(f"Selected output directory: {output_dir}")
                root.destroy()
                return output_dir
            
            root.destroy()
            return None
            
        except Exception as e:
            self._log(f"Error selecting output directory: {e}", "ERROR")
            return None
    
    def find_json_files(self, directory: str) -> List[str]:
        """Find all JSON and JSONL files recursively."""
        self._log(f"Scanning for JSON/JSONL files in: {directory}")
        
        json_files = []
        try:
            # Find JSON files
            for path in Path(directory).rglob("*.json"):
                json_files.append(str(path))
            
            # Find JSONL files
            for path in Path(directory).rglob("*.jsonl"):
                json_files.append(str(path))
            
            self._log(f"Found {len(json_files)} JSON/JSONL files")
            return json_files
            
        except Exception as e:
            self._log(f"Error scanning directory: {e}", "ERROR")
            return []
    
    def load_file_content(self, file_path: str) -> Tuple[Optional[Any], Optional[str]]:
        """Load and parse JSON/JSONL file with error handling."""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix == '.jsonl':
                # Handle JSONL files
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                self._log(f"JSON decode error on line {line_num} in {file_path}: {e}", "WARNING")
                return data, None
            else:
                # Handle JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data, None
                
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in {file_path}: {e}"
            return None, error_msg
        except Exception as e:
            error_msg = f"Error loading {file_path}: {e}"
            return None, error_msg
    
    def is_openai_codex_data(self, data: Any) -> bool:
        """Determine if data is OpenAI Codex format."""
        try:
            # Check for OpenAI Codex session format
            if isinstance(data, dict):
                # OpenAI Codex session structure
                if 'session' in data and 'items' in data:
                    session = data.get('session', {})
                    items = data.get('items', [])
                    # Verify it's OpenAI Codex format by checking content structure
                    if items and len(items) > 0:
                        first_item = items[0]
                        if isinstance(first_item, dict) and 'content' in first_item:
                            content = first_item.get('content', [])
                            if isinstance(content, list) and len(content) > 0:
                                first_content = content[0]
                                if isinstance(first_content, dict) and first_content.get('type') == 'input_text':
                                    return True
                # Check for OpenAI ChatGPT format (already converted)
                if 'conversation_id' in data and 'mapping' in data:
                    return False  # Skip already converted files
            
            # OpenAI Codex data is typically single JSON objects, not arrays
            # But check if it's an array of OpenAI Codex sessions
            if isinstance(data, list):
                if len(data) > 0:
                    first_item = data[0]
                    if isinstance(first_item, dict) and 'session' in first_item and 'items' in first_item:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def convert_codex_to_chatgpt(self, data: Any, source_file: str) -> List[Dict[str, Any]]:
        """Convert OpenAI Codex data to OpenAI ChatGPT format."""
        conversations = []
        
        try:
            if isinstance(data, dict):
                # Check if it's OpenAI Codex format
                if 'session' in data and 'items' in data:
                    conv = self._convert_codex_session(data, source_file)
                    if conv:
                        conversations.append(conv)
            
            elif isinstance(data, list):
                # Multiple OpenAI Codex sessions
                for session_data in data:
                    if isinstance(session_data, dict) and 'session' in session_data and 'items' in session_data:
                        conv = self._convert_codex_session(session_data, source_file)
                        if conv:
                            conversations.append(conv)
            
        except Exception as e:
            self._log(f"Error converting {source_file}: {e}", "ERROR")
        
        return conversations
    
    def _group_jsonl_by_session(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group JSONL data by session ID."""
        sessions = {}
        
        for item in data:
            if isinstance(item, dict):
                session_id = item.get('session_id', item.get('id', 'default'))
                
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session': {
                            'id': session_id,
                            'timestamp': item.get('timestamp', item.get('ts', time.time())),
                            'instructions': item.get('instructions', '')
                        },
                        'items': []
                    }
                
                # Add item to session
                if 'role' in item or 'type' in item:
                    sessions[session_id]['items'].append(item)
        
        return list(sessions.values())
    
    def _convert_codex_session(self, session_data: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
        """Convert an OpenAI Codex session to OpenAI ChatGPT format."""
        try:
            # Extract session metadata
            session_info = session_data.get('session', {})
            items = session_data.get('items', [])
            
            if not items:
                return None
            
            # Parse session details
            conversation_id = session_info.get('id', str(uuid.uuid4()))
            timestamp_str = session_info.get('timestamp', datetime.now().isoformat())
            
            # Parse ISO timestamp
            create_time = self._parse_timestamp(timestamp_str)
            
            # Extract title from first user message
            title = self._extract_codex_title(items)
            if not title:
                title = f"OpenAI Codex Conversation {datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M')}"
            
            # Convert items to OpenAI mapping format
            mapping, current_node = self._convert_codex_items_to_mapping(items, create_time)
            
            if not mapping:
                return None
            
            # Create OpenAI format conversation
            openai_conversation = {
                'conversation_id': conversation_id,
                'title': title,
                'create_time': create_time,
                'update_time': create_time,
                'mapping': mapping,
                'current_node': current_node,
                'moderation_results': [],
                'is_archived': False,
                'metadata': {
                    'converted_from': 'OpenAI Codex',
                    'source_file': source_file,
                    'conversion_date': datetime.now().isoformat(),
                    'original_instructions': session_info.get('instructions', ''),
                    'message_count': len([m for m in mapping.values() if m.get('message')])
                }
            }
            
            return openai_conversation
            
        except Exception as e:
            self._log(f"Error converting OpenAI Codex session from {source_file}: {e}", "ERROR")
            return None
    
    def _convert_single_session(self, session_data: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
        """Convert a single CODEX session to OpenAI format."""
        try:
            # Extract session metadata
            session_info = session_data.get('session', {})
            items = session_data.get('items', [])
            
            # Handle direct session format (no nested session object)
            if not session_info and ('timestamp' in session_data or 'id' in session_data):
                session_info = {
                    'id': session_data.get('id', str(uuid.uuid4())),
                    'timestamp': session_data.get('timestamp', time.time()),
                    'instructions': session_data.get('instructions', '')
                }
                # Items might be directly in the data
                if 'role' in session_data:
                    items = [session_data]
                else:
                    items = session_data.get('items', [])
            
            # Generate conversation metadata
            conversation_id = session_info.get('id', str(uuid.uuid4()))
            create_time = self._parse_timestamp(session_info.get('timestamp', time.time()))
            
            # Extract title from items or use default
            title = self._extract_codex_title(items) or f"OpenAI Codex Conversation {datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M')}"
            
            # Convert messages to OpenAI mapping format
            mapping, current_node = self._convert_codex_items_to_mapping(items, create_time)
            
            if not mapping:
                return None
            
            # Create OpenAI format conversation
            openai_conversation = {
                'conversation_id': conversation_id,
                'title': title,
                'create_time': create_time,
                'update_time': create_time,
                'mapping': mapping,
                'current_node': current_node,
                'moderation_results': [],
                'is_archived': False,
                'metadata': {
                    'converted_from': 'OpenAI Codex',
                    'source_file': source_file,
                    'conversion_date': datetime.now().isoformat(),
                    'original_instructions': session_info.get('instructions', ''),
                    'message_count': len([m for m in mapping.values() if m.get('message')])
                }
            }
            
            return openai_conversation
            
        except Exception as e:
            self._log(f"Error converting session from {source_file}: {e}", "ERROR")
            return None
    
    def _extract_codex_title(self, items: List[Dict[str, Any]]) -> Optional[str]:
        """Extract a title from the first user message in OpenAI Codex format."""
        try:
            for item in items:
                if isinstance(item, dict):
                    role = item.get('role', '')
                    if role == 'user':
                        content = item.get('content', [])
                        if isinstance(content, list) and len(content) > 0:
                            first_content = content[0]
                            if isinstance(first_content, dict):
                                # OpenAI Codex uses 'text' field within 'input_text' type objects
                                text = first_content.get('text', '')
                            else:
                                text = str(first_content)
                            
                            if text:
                                # Clean and truncate title
                                title = re.sub(r'[^\w\s\-\.\(\)]+', '', text)
                                title = re.sub(r'\s+', ' ', title).strip()
                                return title[:100] + "..." if len(title) > 100 else title
            
            return None
            
        except Exception:
            return None
    
    def _convert_codex_items_to_mapping(self, items: List[Dict[str, Any]], base_timestamp: float) -> Tuple[Dict[str, Any], str]:
        """Convert OpenAI Codex items to OpenAI ChatGPT mapping format."""
        mapping = {}
        node_ids = []
        current_timestamp = base_timestamp
        
        try:
            message_index = 0
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # Process different types of items with enhanced extraction
                node_id = str(uuid.uuid4())
                message = None
                
                if item.get('type') == 'message' and 'role' in item:
                    # OpenAI Codex message format
                    role = item.get('role', 'assistant')
                    content = self._extract_codex_content_from_item(item)
                    
                    if content and any(part.strip() for part in content):  # Has actual content
                        message = self._create_openai_message(item, role, content, current_timestamp + message_index * 10)
                        
                elif item.get('type') == 'function_call':
                    # Function call
                    name = item.get('name', 'function')
                    arguments = item.get('arguments', '{}')
                    
                    content = [f"[Function Call: {name}]\n{arguments}"]
                    message = self._create_openai_message(item, 'assistant', content, current_timestamp + message_index * 10)
                    
                elif item.get('type') == 'function_call_output':
                    # Function call output
                    output = item.get('output', '{}')
                    if isinstance(output, str):
                        try:
                            output_data = json.loads(output)
                            if isinstance(output_data, dict) and 'output' in output_data:
                                content = [f"[Function Output]\n{output_data['output']}"]
                            else:
                                content = [f"[Function Output]\n{output}"]
                        except:
                            content = [f"[Function Output]\n{output}"]
                    else:
                        content = [f"[Function Output]\n{json.dumps(output, indent=2)}"]
                    
                    message = self._create_openai_message(item, 'assistant', content, current_timestamp + message_index * 10)
                
                # Add message to mapping if we created one
                if message:
                    node_ids.append(node_id)
                    
                    # Set parent relationship
                    parent_id = node_ids[message_index-1] if message_index > 0 else None
                    
                    mapping[node_id] = {
                        'id': node_id,
                        'message': message,
                        'parent': parent_id,
                        'children': []
                    }
                    
                    message_index += 1
            
            # Update children references
            for i, node_id in enumerate(node_ids):
                if i < len(node_ids) - 1:
                    mapping[node_id]['children'] = [node_ids[i + 1]]
                else:
                    mapping[node_id]['children'] = []
            
            current_node = node_ids[-1] if node_ids else ''
            return mapping, current_node
            
        except Exception as e:
            self._log(f"Error converting items to mapping: {e}", "ERROR")
            return {}, ''
    
    
    def _extract_codex_content_from_item(self, item: Dict[str, Any]) -> List[str]:
        """Extract content from OpenAI Codex item format."""
        parts = []
        
        try:
            content = item.get('content', [])
            
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        # OpenAI Codex format uses 'input_text' and 'output_text' types
                        if c.get('type') == 'input_text':
                            text = c.get('text', '')
                        elif c.get('type') == 'output_text':
                            text = c.get('text', '')
                        else:
                            text = c.get('text', str(c))
                        
                        if text:
                            parts.append(str(text))
                    elif isinstance(c, str):
                        parts.append(c)
            elif isinstance(content, str):
                parts = [content]
            elif isinstance(content, dict):
                text = content.get('text', str(content))
                parts = [text]
            
            # Ensure we have content
            if not parts:
                parts = ['']
                
        except Exception as e:
            self._log(f"Error extracting OpenAI Codex content: {e}", "WARNING")
            parts = ['']
        
        return parts
    
    def _create_openai_message(self, item: Dict[str, Any], role: str, content: List[str], message_time: float) -> Dict[str, Any]:
        """Create OpenAI format message."""
        return {
            'id': item.get('id', str(uuid.uuid4())),
            'author': {
                'role': role,
                'name': None,
                'metadata': {}
            },
            'create_time': message_time,
            'update_time': message_time,
            'content': {
                'content_type': 'text',
                'parts': content
            },
            'status': item.get('status', 'completed'),
            'end_turn': True,
            'weight': 1.0,
            'metadata': {},
            'recipient': 'all'
        }
    
    
    
    
    def _parse_timestamp(self, timestamp: Union[str, int, float, None]) -> float:
        """Parse various timestamp formats to float."""
        if timestamp is None:
            return time.time()
        
        if isinstance(timestamp, (int, float)):
            # Handle milliseconds
            if timestamp > 1e12:
                timestamp = timestamp / 1000
            return float(timestamp)
        
        if isinstance(timestamp, str):
            try:
                # Try parsing as ISO format first (for Claude Code)
                from dateutil import parser as date_parser
                dt = date_parser.parse(timestamp)
                return dt.timestamp()
            except:
                # Try parsing as float
                try:
                    ts = float(timestamp)
                    if ts > 1e12:
                        ts = ts / 1000
                    return ts
                except:
                    return time.time()
        
        return time.time()
    
    def save_converted_file(self, conversations: List[Dict[str, Any]], original_path: str, output_dir: str) -> bool:
        """Save converted conversations to output directory."""
        try:
            original_file = Path(original_path)
            output_path = Path(output_dir)
            
            # Create output filename with -chatgpt_format suffix
            base_name = original_file.stem
            new_filename = f"{base_name}-chatgpt_format.json"
            output_file = output_path / new_filename
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save conversations
            output_data = {
                'conversations': conversations,
                'metadata': {
                    'source_file': str(original_path),
                    'conversion_date': datetime.now().isoformat(),
                    'converter_version': '1.0.0',
                    'total_conversations': len(conversations)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            
            self._log(f"Saved converted file: {output_file}")
            return True
            
        except Exception as e:
            self._log(f"Error saving converted file for {original_path}: {e}", "ERROR")
            return False
    
    def process_files(self, source_dir: str, output_dir: str) -> bool:
        """Main processing function."""
        try:
            self._log("Starting OpenAI Codex to ChatGPT conversion")
            
            # Find all JSON/JSONL files
            json_files = self.find_json_files(source_dir)
            if not json_files:
                self._log("No JSON/JSONL files found", "ERROR")
                return False
            
            # Create progress window
            progress_window = self._create_progress_window(len(json_files))
            
            def update_progress(current, total, filename):
                if progress_window:
                    progress_window.update_progress(current, total, filename)
            
            try:
                # Process each file
                for i, file_path in enumerate(json_files):
                    update_progress(i + 1, len(json_files), Path(file_path).name)
                    
                    self.stats['files_processed'] += 1
                    
                    # Load file content
                    data, error = self.load_file_content(file_path)
                    if error:
                        self._log(f"Skipping {file_path}: {error}", "WARNING")
                        self.stats['skipped_files'] += 1
                        continue
                    
                    # Check if it's OpenAI Codex data
                    if not self.is_openai_codex_data(data):
                        self._log(f"Skipping non-OpenAI Codex file: {Path(file_path).name}", "INFO")
                        self.stats['skipped_files'] += 1
                        continue
                    
                    # Convert to OpenAI ChatGPT format
                    conversations = self.convert_codex_to_chatgpt(data, file_path)
                    
                    if conversations:
                        # Filter out conversations with minimal content
                        valid_conversations = []
                        for conv in conversations:
                            message_count = conv.get('metadata', {}).get('message_count', 0)
                            if message_count > 1:  # More than just system prompt
                                valid_conversations.append(conv)
                        
                        if valid_conversations:
                            # Save converted file
                            if self.save_converted_file(valid_conversations, file_path, output_dir):
                                self.stats['conversations_converted'] += len(valid_conversations)
                            else:
                                self.stats['errors_encountered'] += 1
                        else:
                            self._log(f"Skipping minimal conversation: {Path(file_path).name}", "INFO")
                            self.stats['skipped_files'] += 1
                    else:
                        self._log(f"No conversations found in {Path(file_path).name}", "WARNING")
                        self.stats['skipped_files'] += 1
            
            finally:
                if progress_window:
                    progress_window.close()
            
            self._log("OpenAI Codex to ChatGPT conversion completed")
            return True
            
        except Exception as e:
            self._log(f"Error in main processing: {e}", "ERROR")
            return False
    
    def _create_progress_window(self, total_files: int):
        """Create progress tracking window."""
        try:
            root = tk.Tk()
            root.title("Converting OpenAI Codex to ChatGPT Format")
            root.geometry("500x200")
            root.resizable(False, False)
            
            # Progress info
            info_label = tk.Label(root, text=f"Processing {total_files} files...", font=("Arial", 12))
            info_label.pack(pady=10)
            
            # Progress bar
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=400)
            progress_bar.pack(pady=10)
            
            # Status label
            status_label = tk.Label(root, text="Starting...", font=("Arial", 10))
            status_label.pack(pady=5)
            
            # File label
            file_label = tk.Label(root, text="", font=("Arial", 9), fg="gray")
            file_label.pack(pady=5)
            
            class ProgressWindow:
                def __init__(self, root, progress_var, status_label, file_label):
                    self.root = root
                    self.progress_var = progress_var
                    self.status_label = status_label
                    self.file_label = file_label
                
                def update_progress(self, current, total, filename):
                    progress = (current / total) * 100
                    self.progress_var.set(progress)
                    self.status_label.config(text=f"Processing {current}/{total} files")
                    self.file_label.config(text=f"Current: {filename}")
                    self.root.update()
                
                def close(self):
                    self.root.destroy()
            
            return ProgressWindow(root, progress_var, status_label, file_label)
            
        except Exception as e:
            self._log(f"Error creating progress window: {e}", "ERROR")
            return None
    
    def generate_statistics_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing statistics."""
        processing_time = time.time() - self.start_time
        self.stats['processing_time'] = processing_time
        
        report = {
            'processing_summary': self.stats,
            'platform_info': self.platform_info,
            'processing_details': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'log_file': str(self.log_file)
            }
        }
        
        return report
    
    def run_interactive_conversion(self):
        """Run interactive conversion with GUI dialogs."""
        try:
            # Welcome message
            root = tk.Tk()
            root.withdraw()
            
            messagebox.showinfo(
                "OpenAI Codex to ChatGPT Converter",
                "Welcome to OpenAI Codex to ChatGPT Format Converter v1.0.0\n\n"
                "This tool will convert OpenAI Codex conversation exports to OpenAI ChatGPT format.\n\n"
                "Click OK to begin..."
            )
            
            # Select source directory
            source_dir = self.select_source_directory()
            if not source_dir:
                messagebox.showwarning("Cancelled", "No source directory selected. Exiting.")
                return False
            
            # Select output directory
            output_dir = self.select_output_directory()
            if not output_dir:
                messagebox.showwarning("Cancelled", "No output directory selected. Exiting.")
                return False
            
            # Confirm processing
            json_files = self.find_json_files(source_dir)
            confirm = messagebox.askyesno(
                "Confirm Conversion",
                f"Found {len(json_files)} JSON files to process.\n\n"
                f"Source: {source_dir}\n"
                f"Output: {output_dir}\n\n"
                f"Files will be saved with '-chatgpt_format' suffix.\n\n"
                f"Continue with conversion?"
            )
            
            if not confirm:
                messagebox.showinfo("Cancelled", "Conversion cancelled by user.")
                return False
            
            # Process files
            success = self.process_files(source_dir, output_dir)
            
            if success:
                stats = self.generate_statistics_report()
                messagebox.showinfo(
                    "Conversion Complete",
                    f"OpenAI Codex to ChatGPT conversion completed successfully!\n\n"
                    f"Files processed: {stats['processing_summary']['files_processed']}\n"
                    f"Conversations converted: {stats['processing_summary']['conversations_converted']}\n"
                    f"Files skipped: {stats['processing_summary']['skipped_files']}\n"
                    f"Errors: {stats['processing_summary']['errors_encountered']}\n"
                    f"Processing time: {stats['processing_summary']['processing_time']:.2f} seconds\n\n"
                    f"Output saved to: {output_dir}\n"
                    f"Log file: {self.log_file}"
                )
            else:
                messagebox.showerror(
                    "Conversion Failed",
                    f"OpenAI Codex to ChatGPT conversion failed.\n\n"
                    f"Check the log file for details: {self.log_file}"
                )
            
            return success
            
        except Exception as e:
            self._log(f"Error in interactive conversion: {e}", "ERROR")
            try:
                messagebox.showerror("Error", f"An error occurred: {e}")
            except:
                pass
            return False

def main():
    """Main function to run OpenAI Codex to ChatGPT converter."""
    print("Starting OpenAI Codex to ChatGPT Converter...")
    
    try:
        # Initialize converter
        converter = CodexToOpenAIConverter()
        
        # Run interactive conversion
        success = converter.run_interactive_conversion()
        
        if success:
            # Show macOS notification if available
            try:
                if platform.system() == "Darwin":
                    subprocess.run([
                        "osascript", "-e",
                        f'display notification "OpenAI Codex to ChatGPT conversion completed!" with title "Codex Converter"'
                    ], check=False)
            except:
                pass
        
    except KeyboardInterrupt:
        print("\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()