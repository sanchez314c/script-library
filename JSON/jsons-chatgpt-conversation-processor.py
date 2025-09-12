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
# Script Name: jsons-chatgpt-conversation-processor.py                                                 
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Description: Enhanced ChatGPT Conversation Processor - Advanced conversation 
#              analysis and processing with comprehensive standardization and 
#              professional reporting.
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

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'tkinter',
        'pandas',
        'keyring',
        'beautifulsoup4',
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
import pandas as pd
import keyring
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

class ChatGPTConversationProcessor:
    """Advanced ChatGPT conversation processing and standardization suite."""
    
    def __init__(self):
        """Initialize the ChatGPT conversation processor."""
        self.results = {}
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"chatgpt_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Processing statistics
        self.stats = {
            'files_processed': 0,
            'conversations_found': 0,
            'conversations_deduplicated': 0,
            'errors_encountered': 0,
            'processing_time': 0
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"ChatGPT Conversation Processor Log - {datetime.now()}\n")
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
        """Select source directory containing ChatGPT JSON files."""
        self._log("Opening directory selection dialog")
        
        try:
            root = tk.Tk()
            root.withdraw()
            
            # Try to load saved path
            saved_source = None
            try:
                saved_source = keyring.get_password("chatgpt_processor_v2", "source_folder")
            except:
                pass
            
            # Ask if user wants to use saved path
            if saved_source and Path(saved_source).exists():
                use_saved = messagebox.askyesno(
                    "Use Previous Directory?",
                    f"Use previously selected directory?\n\n{saved_source}"
                )
                if use_saved:
                    root.destroy()
                    return saved_source
            
            # Select new directory
            source_dir = filedialog.askdirectory(
                title="Select Directory Containing ChatGPT JSON Files",
                initialdir=saved_source if saved_source and Path(saved_source).exists() else str(Path.home())
            )
            
            if source_dir:
                # Save for future use
                try:
                    keyring.set_password("chatgpt_processor_v2", "source_folder", source_dir)
                except:
                    pass
                
                self._log(f"Selected source directory: {source_dir}")
                root.destroy()
                return source_dir
            
            root.destroy()
            return None
            
        except Exception as e:
            self._log(f"Error selecting directory: {e}", "ERROR")
            return None
    
    def select_output_directory(self) -> Optional[str]:
        """Select output directory for processed files."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            # Try to load saved path
            saved_output = None
            try:
                saved_output = keyring.get_password("chatgpt_processor_v2", "output_folder")
            except:
                pass
            
            # Ask if user wants to use saved path
            if saved_output and Path(saved_output).exists():
                use_saved = messagebox.askyesno(
                    "Use Previous Output Directory?",
                    f"Use previously selected output directory?\n\n{saved_output}"
                )
                if use_saved:
                    root.destroy()
                    return saved_output
            
            # Select new directory
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Processed Files",
                initialdir=saved_output if saved_output and Path(saved_output).exists() else str(self.desktop_path)
            )
            
            if output_dir:
                # Save for future use
                try:
                    keyring.set_password("chatgpt_processor_v2", "output_folder", output_dir)
                except:
                    pass
                
                self._log(f"Selected output directory: {output_dir}")
                root.destroy()
                return output_dir
            
            root.destroy()
            return None
            
        except Exception as e:
            self._log(f"Error selecting output directory: {e}", "ERROR")
            return None
    
    def find_json_files(self, directory: str) -> List[str]:
        """Find all JSON files in directory and subdirectories."""
        self._log(f"Scanning for JSON files in: {directory}")
        
        json_files = []
        try:
            for path in Path(directory).rglob("*.json"):
                json_files.append(str(path))
            
            self._log(f"Found {len(json_files)} JSON files")
            return json_files
            
        except Exception as e:
            self._log(f"Error scanning directory: {e}", "ERROR")
            return []
    
    def load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load and parse JSON file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
            
        except json.JSONDecodeError as e:
            self._log(f"JSON decode error in {file_path}: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return None
        except Exception as e:
            self._log(f"Error loading {file_path}: {e}", "ERROR")
            self.stats['errors_encountered'] += 1
            return None
    
    def standardize_conversation_format(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize conversation to consistent format."""
        try:
            # Generate conversation ID if missing
            if 'conversation_id' not in conversation:
                # Create hash from mapping content
                mapping_str = json.dumps(conversation.get('mapping', {}), sort_keys=True)
                conversation_id = hashlib.md5(mapping_str.encode()).hexdigest()
                conversation['conversation_id'] = conversation_id
            
            # Standardize timestamps
            create_time = self._parse_timestamp(conversation.get('create_time'))
            update_time = self._parse_timestamp(conversation.get('update_time', create_time))
            
            # Format title
            title = self._format_title(conversation.get('title', 'Untitled Conversation'))
            formatted_date = self._format_date(create_time)
            
            # Combine date and title
            if not title.startswith(formatted_date):
                title = f"{formatted_date} - {title}"
            
            # Create standardized structure
            standardized = {
                'conversation_id': conversation['conversation_id'],
                'title': title,
                'create_time': create_time,
                'update_time': update_time,
                'mapping': self._process_mapping(conversation.get('mapping', {})),
                'current_node': conversation.get('current_node', ''),
                'moderation_results': conversation.get('moderation_results', []),
                'is_archived': conversation.get('is_archived', False),
                'metadata': {
                    'processed_by': 'ChatGPT Processor v1.0.0',
                    'processed_date': datetime.now().isoformat(),
                    'message_count': len(conversation.get('mapping', {})),
                    'original_file': conversation.get('source_file', 'unknown')
                }
            }
            
            return standardized
            
        except Exception as e:
            self._log(f"Error standardizing conversation: {e}", "ERROR")
            return None
    
    def _parse_timestamp(self, timestamp: Union[str, int, float, None]) -> float:
        """Parse various timestamp formats to float."""
        if timestamp is None:
            return datetime.now().timestamp()
        
        if isinstance(timestamp, (int, float)):
            # Handle milliseconds
            if timestamp > 1e12:
                timestamp = timestamp / 1000
            return float(timestamp)
        
        if isinstance(timestamp, str):
            try:
                # Try parsing as ISO format
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
                    return datetime.now().timestamp()
        
        return datetime.now().timestamp()
    
    def _format_date(self, timestamp: float) -> str:
        """Format timestamp to readable date."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%y-%m-%d')
        except:
            return datetime.now().strftime('%y-%m-%d')
    
    def _format_title(self, title: str) -> str:
        """Clean and format conversation title."""
        if not title or title.strip() == '':
            return "Untitled Conversation"
        
        # Clean unwanted characters
        clean_title = re.sub(r'[^\w\s\-\.\(\)]+', '', title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        # Limit length
        if len(clean_title) > 100:
            clean_title = clean_title[:97] + "..."
        
        return clean_title
    
    def _process_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Process and standardize mapping structure."""
        processed_mapping = {}
        
        for node_id, node_data in mapping.items():
            if not isinstance(node_data, dict):
                continue
            
            message = node_data.get('message')
            if message is None:
                # Keep non-message nodes as-is
                processed_mapping[node_id] = node_data
                continue
            
            # Process message content
            processed_message = {
                'id': message.get('id', node_id),
                'author': self._process_author(message.get('author', {})),
                'create_time': self._parse_timestamp(message.get('create_time')),
                'update_time': self._parse_timestamp(message.get('update_time')),
                'content': self._process_content(message.get('content', {})),
                'status': message.get('status', 'finished_successfully'),
                'end_turn': message.get('end_turn', True),
                'weight': message.get('weight', 1.0),
                'metadata': message.get('metadata', {}),
                'recipient': message.get('recipient', 'all')
            }
            
            processed_mapping[node_id] = {
                'id': node_data.get('id', node_id),
                'message': processed_message,
                'parent': node_data.get('parent'),
                'children': node_data.get('children', [])
            }
        
        return processed_mapping
    
    def _process_author(self, author: Dict[str, Any]) -> Dict[str, Any]:
        """Process author information."""
        return {
            'role': author.get('role', 'assistant'),
            'name': author.get('name'),
            'metadata': author.get('metadata', {})
        }
    
    def _process_content(self, content: Union[Dict, str, List]) -> Dict[str, Any]:
        """Process message content to standard format."""
        if isinstance(content, str):
            return {
                'content_type': 'text',
                'parts': [content]
            }
        
        if isinstance(content, list):
            return {
                'content_type': 'text',
                'parts': [str(item) for item in content]
            }
        
        if isinstance(content, dict):
            return {
                'content_type': content.get('content_type', 'text'),
                'parts': content.get('parts', [''])
            }
        
        return {
            'content_type': 'text',
            'parts': ['']
        }
    
    def deduplicate_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate conversations based on content hash."""
        self._log("Deduplicating conversations")
        
        seen_hashes = set()
        unique_conversations = []
        
        for conv in conversations:
            # Create content hash
            mapping_str = json.dumps(conv.get('mapping', {}), sort_keys=True)
            content_hash = hashlib.md5(mapping_str.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_conversations.append(conv)
                self.stats['conversations_found'] += 1
            else:
                self.stats['conversations_deduplicated'] += 1
        
        self._log(f"Kept {len(unique_conversations)} unique conversations, removed {self.stats['conversations_deduplicated']} duplicates")
        return unique_conversations
    
    def sort_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort conversations by creation time."""
        try:
            return sorted(conversations, key=lambda x: x.get('create_time', 0))
        except Exception as e:
            self._log(f"Error sorting conversations: {e}", "ERROR")
            return conversations
    
    def save_json_output(self, data: Any, file_path: str) -> bool:
        """Save data to JSON file with error handling."""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            self._log(f"Saved JSON: {file_path}")
            return True
            
        except Exception as e:
            self._log(f"Error saving JSON {file_path}: {e}", "ERROR")
            return False
    
    def generate_html_output(self, conversations: List[Dict[str, Any]], output_dir: str) -> bool:
        """Generate HTML files for conversations."""
        self._log("Generating HTML output")
        
        try:
            html_dir = Path(output_dir) / "html"
            html_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate individual HTML files
            for conv in conversations:
                html_content = self._conversation_to_html(conv)
                html_file = html_dir / f"{conv['title']}.html"
                
                try:
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                except Exception as e:
                    self._log(f"Error saving HTML for {conv['title']}: {e}", "ERROR")
            
            # Generate combined HTML
            combined_html = self._generate_combined_html(conversations)
            combined_file = html_dir / "all_conversations.html"
            
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(combined_html)
            
            self._log(f"Generated HTML files in: {html_dir}")
            return True
            
        except Exception as e:
            self._log(f"Error generating HTML: {e}", "ERROR")
            return False
    
    def _conversation_to_html(self, conversation: Dict[str, Any]) -> str:
        """Convert single conversation to HTML."""
        title = conversation.get('title', 'Untitled Conversation')
        create_time = conversation.get('create_time', time.time())
        
        # Format date
        try:
            date_str = datetime.fromtimestamp(create_time).strftime('%B %d, %Y at %I:%M %p')
        except:
            date_str = "Unknown Date"
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #212121;
            color: #ededec;
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #404040;
            padding-bottom: 20px;
        }}
        .title {{
            font-size: 1.8em;
            margin-bottom: 10px;
            color: #ffffff;
        }}
        .date {{
            color: #aaaaaa;
            font-size: 0.9em;
        }}
        .message {{
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            position: relative;
        }}
        .user-message {{
            background-color: #2f2f2f;
            margin-left: 10%;
            border-left: 3px solid #0084ff;
        }}
        .assistant-message {{
            background-color: #1a1a1a;
            margin-right: 10%;
            border-left: 3px solid #00d084;
        }}
        .author {{
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .user-author {{
            color: #0084ff;
        }}
        .assistant-author {{
            color: #00d084;
        }}
        .content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .timestamp {{
            font-size: 0.8em;
            color: #888888;
            margin-top: 10px;
        }}
        @media (max-width: 600px) {{
            .user-message, .assistant-message {{
                margin-left: 0;
                margin-right: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">{title}</h1>
        <div class="date">{date_str}</div>
    </div>
    <div class="conversation">
"""
        
        # Add messages
        mapping = conversation.get('mapping', {})
        messages = self._extract_messages_from_mapping(mapping)
        
        for message in messages:
            author_role = message.get('author', {}).get('role', 'assistant')
            content_parts = message.get('content', {}).get('parts', [''])
            content = '\n'.join(str(part) for part in content_parts if part)
            
            if not content.strip():
                continue
            
            # Format timestamp
            msg_time = message.get('create_time', create_time)
            try:
                time_str = datetime.fromtimestamp(msg_time).strftime('%I:%M %p')
            except:
                time_str = ""
            
            author_class = "user" if author_role == "user" else "assistant"
            author_display = "You" if author_role == "user" else "ChatGPT"
            
            html_content += f"""
        <div class="message {author_class}-message">
            <div class="author {author_class}-author">{author_display}</div>
            <div class="content">{content}</div>
            <div class="timestamp">{time_str}</div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _extract_messages_from_mapping(self, mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract messages from mapping in chronological order."""
        messages = []
        
        # Find all message nodes
        for node_id, node_data in mapping.items():
            if isinstance(node_data, dict) and 'message' in node_data:
                message = node_data['message']
                if message and isinstance(message, dict):
                    messages.append(message)
        
        # Sort by creation time
        try:
            messages.sort(key=lambda x: x.get('create_time', 0))
        except:
            pass
        
        return messages
    
    def _generate_combined_html(self, conversations: List[Dict[str, Any]]) -> str:
        """Generate combined HTML file for all conversations."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All ChatGPT Conversations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #212121;
            color: #ededec;
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .conversation-separator {
            border-top: 2px solid #404040;
            margin: 40px 0;
            padding-top: 30px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #404040;
            padding-bottom: 20px;
        }
        .title {
            font-size: 1.8em;
            margin-bottom: 10px;
            color: #ffffff;
        }
        .date {
            color: #aaaaaa;
            font-size: 0.9em;
        }
        .message {
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            position: relative;
        }
        .user-message {
            background-color: #2f2f2f;
            margin-left: 10%;
            border-left: 3px solid #0084ff;
        }
        .assistant-message {
            background-color: #1a1a1a;
            margin-right: 10%;
            border-left: 3px solid #00d084;
        }
        .author {
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        .user-author {
            color: #0084ff;
        }
        .assistant-author {
            color: #00d084;
        }
        .content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .timestamp {
            font-size: 0.8em;
            color: #888888;
            margin-top: 10px;
        }
        @media (max-width: 600px) {
            .user-message, .assistant-message {
                margin-left: 0;
                margin-right: 0;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">All ChatGPT Conversations</h1>
        <div class="date">Generated on """ + datetime.now().strftime('%B %d, %Y at %I:%M %p') + """</div>
    </div>
"""
        
        for i, conv in enumerate(conversations):
            if i > 0:
                html_content += '<div class="conversation-separator"></div>'
            
            # Add individual conversation HTML (without head/body tags)
            conv_html = self._conversation_to_html(conv)
            # Extract just the content between body tags
            start = conv_html.find('<div class="header">')
            end = conv_html.find('</body>')
            if start != -1 and end != -1:
                html_content += conv_html[start:end]
        
        html_content += """
</body>
</html>
"""
        
        return html_content
    
    def generate_statistics_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing statistics."""
        processing_time = time.time() - self.start_time
        
        report = {
            'processing_summary': {
                'total_processing_time': processing_time,
                'files_processed': self.stats['files_processed'],
                'conversations_found': self.stats['conversations_found'],
                'conversations_deduplicated': self.stats['conversations_deduplicated'],
                'errors_encountered': self.stats['errors_encountered'],
                'success_rate': (self.stats['files_processed'] - self.stats['errors_encountered']) / max(1, self.stats['files_processed']) * 100
            },
            'platform_info': self.platform_info,
            'processing_details': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'log_file': str(self.log_file)
            }
        }
        
        return report
    
    def process_conversations(self, source_dir: str, output_dir: str) -> bool:
        """Main processing function."""
        try:
            self._log("Starting ChatGPT conversation processing")
            
            # Find JSON files
            json_files = self.find_json_files(source_dir)
            if not json_files:
                self._log("No JSON files found", "ERROR")
                return False
            
            # Create output directories
            output_path = Path(output_dir)
            json_output_dir = output_path / "json"
            individual_dir = json_output_dir / "individual"
            
            json_output_dir.mkdir(parents=True, exist_ok=True)
            individual_dir.mkdir(parents=True, exist_ok=True)
            
            # Process files with progress tracking
            all_conversations = []
            
            # Create progress window
            progress_window = self._create_progress_window(len(json_files))
            
            def update_progress(current, total, filename):
                if progress_window:
                    progress_window.update_progress(current, total, filename)
            
            # Process files
            for i, file_path in enumerate(json_files):
                update_progress(i + 1, len(json_files), Path(file_path).name)
                
                data = self.load_json_file(file_path)
                if data is None:
                    continue
                
                self.stats['files_processed'] += 1
                
                # Process based on data structure
                conversations = self._extract_conversations_from_data(data, file_path)
                
                for conv in conversations:
                    standardized = self.standardize_conversation_format(conv)
                    if standardized:
                        all_conversations.append(standardized)
            
            # Close progress window
            if progress_window:
                progress_window.close()
            
            if not all_conversations:
                self._log("No valid conversations found", "ERROR")
                return False
            
            # Deduplicate and sort
            unique_conversations = self.deduplicate_conversations(all_conversations)
            sorted_conversations = self.sort_conversations(unique_conversations)
            
            # Save individual JSON files
            self._log("Saving individual conversation files")
            for conv in sorted_conversations:
                filename = f"{conv['title']}.json"
                # Clean filename
                filename = re.sub(r'[^\w\s\-\.]', '', filename)
                file_path = individual_dir / filename
                self.save_json_output(conv, str(file_path))
            
            # Save combined JSON
            combined_file = json_output_dir / "all_conversations.json"
            self.save_json_output(sorted_conversations, str(combined_file))
            
            # Generate HTML output
            self.generate_html_output(sorted_conversations, str(output_path))
            
            # Save statistics
            stats_report = self.generate_statistics_report()
            stats_file = output_path / "processing_statistics.json"
            self.save_json_output(stats_report, str(stats_file))
            
            self._log("ChatGPT conversation processing completed successfully")
            return True
            
        except Exception as e:
            self._log(f"Error in main processing: {e}", "ERROR")
            return False
    
    def _extract_conversations_from_data(self, data: Any, source_file: str) -> List[Dict[str, Any]]:
        """Extract conversations from various data structures."""
        conversations = []
        
        try:
            if isinstance(data, list):
                # List of conversations
                for item in data:
                    if isinstance(item, dict) and 'mapping' in item:
                        item['source_file'] = source_file
                        conversations.append(item)
            
            elif isinstance(data, dict):
                if 'mapping' in data:
                    # Single conversation
                    data['source_file'] = source_file
                    conversations.append(data)
                else:
                    # Dictionary with conversations as values
                    for value in data.values():
                        if isinstance(value, dict) and 'mapping' in value:
                            value['source_file'] = source_file
                            conversations.append(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and 'mapping' in item:
                                    item['source_file'] = source_file
                                    conversations.append(item)
        
        except Exception as e:
            self._log(f"Error extracting conversations from {source_file}: {e}", "ERROR")
        
        return conversations
    
    def _create_progress_window(self, total_files: int):
        """Create progress tracking window."""
        try:
            root = tk.Tk()
            root.title("Processing ChatGPT Conversations")
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
    
    def run_interactive_processing(self):
        """Run interactive processing with GUI dialogs."""
        try:
            # Welcome message
            root = tk.Tk()
            root.withdraw()
            
            welcome = messagebox.showinfo(
                "ChatGPT Conversation Processor",
                "Welcome to ChatGPT Conversation Processor v1.0.0\n\n"
                "This tool will help you process and standardize ChatGPT conversation exports.\n\n"
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
                "Confirm Processing",
                f"Found {len(json_files)} JSON files to process.\n\n"
                f"Source: {source_dir}\n"
                f"Output: {output_dir}\n\n"
                f"Continue with processing?"
            )
            
            if not confirm:
                messagebox.showinfo("Cancelled", "Processing cancelled by user.")
                return False
            
            # Process conversations
            success = self.process_conversations(source_dir, output_dir)
            
            if success:
                stats = self.generate_statistics_report()
                messagebox.showinfo(
                    "Processing Complete",
                    f"ChatGPT conversation processing completed successfully!\n\n"
                    f"Files processed: {stats['processing_summary']['files_processed']}\n"
                    f"Conversations found: {stats['processing_summary']['conversations_found']}\n"
                    f"Duplicates removed: {stats['processing_summary']['conversations_deduplicated']}\n"
                    f"Processing time: {stats['processing_summary']['total_processing_time']:.2f} seconds\n\n"
                    f"Output saved to: {output_dir}\n"
                    f"Log file: {self.log_file}"
                )
            else:
                messagebox.showerror(
                    "Processing Failed",
                    f"ChatGPT conversation processing failed.\n\n"
                    f"Check the log file for details: {self.log_file}"
                )
            
            return success
            
        except Exception as e:
            self._log(f"Error in interactive processing: {e}", "ERROR")
            try:
                messagebox.showerror("Error", f"An error occurred: {e}")
            except:
                pass
            return False

def main():
    """Main function to run ChatGPT conversation processor."""
    print("Starting ChatGPT Conversation Processor Suite...")
    
    try:
        # Initialize processor
        processor = ChatGPTConversationProcessor()
        
        # Run interactive processing
        success = processor.run_interactive_processing()
        
        if success:
            # Show macOS notification if available
            try:
                if platform.system() == "Darwin":
                    subprocess.run([
                        "osascript", "-e",
                        f'display notification "ChatGPT conversation processing completed!" with title "GET SWIFTY - ChatGPT Processor"'
                    ], check=False)
            except:
                pass
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()