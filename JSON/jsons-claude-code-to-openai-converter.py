#!/usr/bin/env python3

####################################################################################
#                                                                                  #
#     ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗         ██╗███████╗ ██████╗ ███╗   ██╗#
#    ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝         ██║██╔════╝██╔═══██╗████╗  ██║#
#    ██║     ██║     ███████║██║   ██║██║  ██║█████╗           ██║███████╗██║   ██║██╔██╗ ██║#
#    ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝      ██   ██║╚════██║██║   ██║██║╚██╗██║#
#    ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗    ╚█████╔╝███████║╚██████╔╝██║ ╚████║#
#     ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝#
#                                                                                  #
#     ████████╗ ██████╗      ██████╗ ██████╗ ███████╗███╗   ██╗ █████╗ ██╗         #
#     ╚══██╔══╝██╔═══██╗    ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██║         #
#        ██║   ██║   ██║    ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║         #
#        ██║   ██║   ██║    ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║         #
#        ██║   ╚██████╔╝    ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║         #
#        ╚═╝    ╚═════╝      ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝         #
#                                                                                  #
####################################################################################
#
# Script Name: claude-json-to-openai-converter.py                                                 
# 
# Author: Claude Code Assistant  
#                                              
# Date Created: 2025-05-30                                                       
#
# Last Modified: 2025-05-30                                                      
#
# Description: Claude JSON to OpenAI ChatGPT Format Converter - Converts Claude
#              conversation exports (JSONL format) to OpenAI ChatGPT compatible 
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

class ClaudeJSONToOpenAIConverter:
    """Advanced Claude JSON to OpenAI ChatGPT format converter."""
    
    def __init__(self):
        """Initialize the converter."""
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"claude_json_to_openai_converter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
                f.write(f"Claude JSON to OpenAI Converter Log - {datetime.now()}\n")
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
        """Select source directory containing Claude JSON files."""
        self._log("Opening source directory selection dialog")
        
        try:
            root = tk.Tk()
            root.withdraw()
            
            source_dir = filedialog.askdirectory(
                title="Select Directory Containing Claude JSON/JSONL Files",
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
    
    def find_jsonl_files(self, directory: str) -> List[str]:
        """Find all JSONL files recursively."""
        self._log(f"Scanning for JSONL files in: {directory}")
        
        jsonl_files = []
        try:
            # Find JSONL files
            for path in Path(directory).rglob("*.jsonl"):
                jsonl_files.append(str(path))
            
            self._log(f"Found {len(jsonl_files)} JSONL files")
            return jsonl_files
            
        except Exception as e:
            self._log(f"Error scanning directory: {e}", "ERROR")
            return []
    
    def load_claude_jsonl(self, file_path: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """Load and parse Claude JSONL file."""
        try:
            messages = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            message = json.loads(line)
                            messages.append(message)
                        except json.JSONDecodeError as e:
                            self._log(f"JSON decode error on line {line_num} in {file_path}: {e}", "WARNING")
            
            if not messages:
                return None, "No valid messages found"
            
            return messages, None
                
        except Exception as e:
            error_msg = f"Error loading {file_path}: {e}"
            return None, error_msg
    
    def is_claude_conversation_data(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if data is Claude conversation format."""
        try:
            if not messages or not isinstance(messages, list):
                return False
            
            # Check for Claude message format
            first_message = messages[0]
            if isinstance(first_message, dict):
                # Check for Claude-specific fields
                if ('sessionId' in first_message and 
                    'uuid' in first_message and 
                    'message' in first_message and
                    'timestamp' in first_message):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def convert_claude_to_openai(self, messages: List[Dict[str, Any]], source_file: str) -> Optional[Dict[str, Any]]:
        """Convert Claude JSONL to OpenAI format."""
        try:
            if not messages:
                return None
            
            # Get session info from first message
            session_id = messages[0].get('sessionId', str(uuid.uuid4()))
            first_timestamp = messages[0].get('timestamp', datetime.now().isoformat())
            
            # Parse timestamp
            create_time = self._parse_timestamp(first_timestamp)
            
            # Extract title from first user message
            title = self._extract_claude_title(messages)
            if not title:
                title = f"Claude Conversation {datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M')}"
            
            # Build conversation thread based on parentUuid relationships
            mapping, current_node = self._build_claude_thread_mapping(messages, create_time)
            
            if not mapping:
                return None
            
            # Create OpenAI format conversation
            openai_conversation = {
                'conversation_id': session_id,
                'title': title,
                'create_time': create_time,
                'update_time': create_time,
                'mapping': mapping,
                'current_node': current_node,
                'moderation_results': [],
                'is_archived': False,
                'metadata': {
                    'converted_from': 'Claude JSON Export',
                    'source_file': source_file,
                    'conversion_date': datetime.now().isoformat(),
                    'total_messages': len(messages),
                    'message_count': len([m for m in mapping.values() if m.get('message')])
                }
            }
            
            return openai_conversation
            
        except Exception as e:
            self._log(f"Error converting Claude messages from {source_file}: {e}", "ERROR")
            return None
    
    def _extract_claude_title(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract title from Claude messages."""
        try:
            for message in messages:
                if isinstance(message, dict):
                    msg_data = message.get('message', {})
                    if msg_data.get('role') == 'user':
                        content = msg_data.get('content', '')
                        if isinstance(content, str) and content.strip():
                            # Clean title
                            title = content.strip()
                            # Skip meta messages and commands
                            if (not title.startswith('<') and 
                                not title.startswith('Caveat:') and
                                len(title) > 10):
                                # Truncate and clean
                                title = re.sub(r'[^\w\s\-\.\(\)]+', '', title)
                                title = re.sub(r'\s+', ' ', title).strip()
                                return title[:100] + "..." if len(title) > 100 else title
            
            return None
            
        except Exception:
            return None
    
    def _build_claude_thread_mapping(self, messages: List[Dict[str, Any]], base_timestamp: float) -> Tuple[Dict[str, Any], str]:
        """Build OpenAI mapping from Claude message thread."""
        mapping = {}
        uuid_to_node_id = {}
        node_ids = []
        
        try:
            # Filter out meta messages and organize by UUID
            conversation_messages = []
            for msg in messages:
                if (isinstance(msg, dict) and 
                    not msg.get('isMeta', False) and
                    msg.get('type') in ['user', 'assistant'] and
                    'message' in msg):
                    conversation_messages.append(msg)
            
            if not conversation_messages:
                return {}, ''
            
            # Create mapping entries
            for i, claude_msg in enumerate(conversation_messages):
                node_id = str(uuid.uuid4())
                msg_uuid = claude_msg.get('uuid')
                
                node_ids.append(node_id)
                uuid_to_node_id[msg_uuid] = node_id
                
                # Extract message content
                msg_data = claude_msg.get('message', {})
                role = msg_data.get('role', 'assistant')
                content = self._extract_claude_content(msg_data)
                
                # Parse timestamp
                msg_timestamp = claude_msg.get('timestamp', base_timestamp + i * 10)
                message_time = self._parse_timestamp(msg_timestamp)
                
                # Create OpenAI message format
                message = {
                    'id': claude_msg.get('uuid', str(uuid.uuid4())),
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
                    'status': 'finished_successfully',
                    'end_turn': True,
                    'weight': 1.0,
                    'metadata': {
                        'model': msg_data.get('model', ''),
                        'cost_usd': claude_msg.get('costUSD', 0),
                        'duration_ms': claude_msg.get('durationMs', 0)
                    },
                    'recipient': 'all'
                }
                
                # Find parent based on parentUuid
                parent_uuid = claude_msg.get('parentUuid')
                parent_id = uuid_to_node_id.get(parent_uuid) if parent_uuid else None
                
                mapping[node_id] = {
                    'id': node_id,
                    'message': message,
                    'parent': parent_id,
                    'children': []
                }
            
            # Update children references
            for node_id, node_data in mapping.items():
                parent_id = node_data.get('parent')
                if parent_id and parent_id in mapping:
                    mapping[parent_id]['children'].append(node_id)
            
            current_node = node_ids[-1] if node_ids else ''
            return mapping, current_node
            
        except Exception as e:
            self._log(f"Error building Claude thread mapping: {e}", "ERROR")
            return {}, ''
    
    def _extract_claude_content(self, msg_data: Dict[str, Any]) -> List[str]:
        """Extract content from Claude message data."""
        parts = []
        
        try:
            content = msg_data.get('content', '')
            
            if isinstance(content, str):
                if content.strip():
                    parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            if text:
                                parts.append(text)
                        elif item.get('type') == 'tool_use':
                            # Include tool use information
                            tool_name = item.get('name', 'tool')
                            tool_input = item.get('input', {})
                            tool_text = f"[Tool: {tool_name}]\n{json.dumps(tool_input, indent=2)}"
                            parts.append(tool_text)
                    elif isinstance(item, str):
                        parts.append(item)
            
            # Handle tool results
            if 'toolUseResult' in msg_data:
                result = msg_data['toolUseResult']
                if isinstance(result, dict):
                    if 'stdout' in result:
                        parts.append(f"[Tool Output]\n{result['stdout']}")
                    elif result:
                        parts.append(f"[Tool Result]\n{json.dumps(result, indent=2)}")
                elif isinstance(result, str):
                    parts.append(f"[Tool Result]\n{result}")
            
            # Ensure we have at least one part
            if not parts:
                parts = ['']
                
        except Exception as e:
            self._log(f"Error extracting Claude content: {e}", "WARNING")
            parts = ['']
        
        return parts
    
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
                # Try parsing as ISO format first
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
    
    def save_converted_file(self, conversation: Dict[str, Any], original_path: str, output_dir: str) -> bool:
        """Save converted conversation to output directory."""
        try:
            original_file = Path(original_path)
            output_path = Path(output_dir)
            
            # Create output filename with -claude-oai-format suffix
            base_name = original_file.stem
            new_filename = f"{base_name}-claude-oai-format.json"
            output_file = output_path / new_filename
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save conversation
            output_data = {
                'conversations': [conversation],
                'metadata': {
                    'source_file': str(original_path),
                    'conversion_date': datetime.now().isoformat(),
                    'converter_version': '1.0.0',
                    'total_conversations': 1
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
            self._log("Starting Claude JSON to OpenAI conversion")
            
            # Find all JSONL files
            jsonl_files = self.find_jsonl_files(source_dir)
            if not jsonl_files:
                self._log("No JSONL files found", "ERROR")
                return False
            
            # Create progress window
            progress_window = self._create_progress_window(len(jsonl_files))
            
            def update_progress(current, total, filename):
                if progress_window:
                    progress_window.update_progress(current, total, filename)
            
            try:
                # Process each file
                for i, file_path in enumerate(jsonl_files):
                    update_progress(i + 1, len(jsonl_files), Path(file_path).name)
                    
                    self.stats['files_processed'] += 1
                    
                    # Load file content
                    messages, error = self.load_claude_jsonl(file_path)
                    if error:
                        self._log(f"Skipping {file_path}: {error}", "WARNING")
                        self.stats['skipped_files'] += 1
                        continue
                    
                    # Check if it's Claude conversation data
                    if not self.is_claude_conversation_data(messages):
                        self._log(f"Skipping non-Claude file: {Path(file_path).name}", "INFO")
                        self.stats['skipped_files'] += 1
                        continue
                    
                    # Convert to OpenAI format
                    conversation = self.convert_claude_to_openai(messages, file_path)
                    
                    if conversation:
                        # Save converted file
                        if self.save_converted_file(conversation, file_path, output_dir):
                            self.stats['conversations_converted'] += 1
                        else:
                            self.stats['errors_encountered'] += 1
                    else:
                        self._log(f"No conversation found in {Path(file_path).name}", "WARNING")
                        self.stats['skipped_files'] += 1
            
            finally:
                if progress_window:
                    progress_window.close()
            
            self._log("Claude JSON to OpenAI conversion completed")
            return True
            
        except Exception as e:
            self._log(f"Error in main processing: {e}", "ERROR")
            return False
    
    def _create_progress_window(self, total_files: int):
        """Create progress tracking window."""
        try:
            root = tk.Tk()
            root.title("Converting Claude JSON to OpenAI Format")
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
                "Claude JSON to OpenAI Converter",
                "Welcome to Claude JSON to OpenAI ChatGPT Format Converter v1.0.0\n\n"
                "This tool will convert Claude conversation exports to OpenAI ChatGPT format.\n\n"
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
            jsonl_files = self.find_jsonl_files(source_dir)
            confirm = messagebox.askyesno(
                "Confirm Conversion",
                f"Found {len(jsonl_files)} JSONL files to process.\n\n"
                f"Source: {source_dir}\n"
                f"Output: {output_dir}\n\n"
                f"Files will be saved with '-claude-oai-format' suffix.\n\n"
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
                    f"Claude JSON to OpenAI conversion completed successfully!\n\n"
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
                    f"Claude JSON to OpenAI conversion failed.\n\n"
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
    """Main function to run Claude JSON to OpenAI converter."""
    print("Starting Claude JSON to OpenAI Converter...")
    
    try:
        # Initialize converter
        converter = ClaudeJSONToOpenAIConverter()
        
        # Run interactive conversion
        success = converter.run_interactive_conversion()
        
        if success:
            # Show macOS notification if available
            try:
                if platform.system() == "Darwin":
                    subprocess.run([
                        "osascript", "-e",
                        f'display notification "Claude JSON to OpenAI conversion completed!" with title "Claude Converter"'
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