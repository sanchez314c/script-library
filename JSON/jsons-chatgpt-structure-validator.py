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
# Script Name: jsons-chatgpt-structure-validator.py                                                 
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Description: Enhanced ChatGPT Structure Validator - Advanced JSON structure 
#              validation and analysis with comprehensive quality assessment.
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
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union, Set
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'tkinter',
        'jsonschema',
        'deepdiff',
        'matplotlib',
        'pandas'
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
from jsonschema import validate, ValidationError, Draft7Validator
from deepdiff import DeepDiff
import matplotlib.pyplot as plt
import pandas as pd

class ChatGPTStructureValidator:
    """Advanced ChatGPT JSON structure validation and analysis suite."""
    
    def __init__(self):
        """Initialize the ChatGPT structure validator."""
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"chatgpt_structure_validator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Validation results
        self.validation_results = {}
        self.statistics = {
            'files_analyzed': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors_found': 0,
            'warnings_found': 0
        }
        
        # ChatGPT schema definition
        self.chatgpt_schema = self._define_chatgpt_schema()
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"ChatGPT Structure Validator Log - {datetime.now()}\n")
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
    
    def _define_chatgpt_schema(self) -> Dict[str, Any]:
        """Define the expected ChatGPT conversation JSON schema."""
        return {
            "type": "object",
            "required": ["title", "create_time", "update_time", "mapping"],
            "properties": {
                "title": {"type": "string"},
                "create_time": {"type": "number"},
                "update_time": {"type": "number"},
                "mapping": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "message": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "id": {"type": "string"},
                                        "author": {
                                            "type": "object",
                                            "properties": {
                                                "role": {"type": "string"},
                                                "name": {"type": ["string", "null"]},
                                                "metadata": {"type": "object"}
                                            },
                                            "required": ["role"]
                                        },
                                        "create_time": {"type": "number"},
                                        "update_time": {"type": "number"},
                                        "content": {
                                            "type": "object",
                                            "properties": {
                                                "content_type": {"type": "string"},
                                                "parts": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                }
                                            },
                                            "required": ["content_type", "parts"]
                                        },
                                        "status": {"type": "string"},
                                        "end_turn": {"type": "boolean"},
                                        "weight": {"type": "number"},
                                        "metadata": {"type": "object"},
                                        "recipient": {"type": "string"}
                                    },
                                    "required": ["id", "author", "create_time", "content"]
                                },
                                "parent": {"type": ["string", "null"]},
                                "children": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["id"]
                        }
                    }
                },
                "moderation_results": {"type": "array"},
                "current_node": {"type": "string"},
                "conversation_id": {"type": "string"},
                "is_archived": {"type": "boolean"}
            }
        }
    
    def select_files_for_validation(self) -> List[str]:
        """Select files for validation using GUI dialog."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            files = filedialog.askopenfilenames(
                title="Select ChatGPT JSON Files to Validate",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=str(Path.home())
            )
            
            root.destroy()
            
            if files:
                self._log(f"Selected {len(files)} files for validation")
                return list(files)
            
            return []
            
        except Exception as e:
            self._log(f"Error selecting files: {e}", "ERROR")
            return []
    
    def select_directory_for_validation(self) -> Optional[str]:
        """Select directory for bulk validation."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            directory = filedialog.askdirectory(
                title="Select Directory Containing ChatGPT JSON Files",
                initialdir=str(Path.home())
            )
            
            root.destroy()
            
            if directory:
                self._log(f"Selected directory for validation: {directory}")
                return directory
            
            return None
            
        except Exception as e:
            self._log(f"Error selecting directory: {e}", "ERROR")
            return None
    
    def find_json_files_in_directory(self, directory: str) -> List[str]:
        """Find all JSON files in directory and subdirectories."""
        json_files = []
        try:
            for path in Path(directory).rglob("*.json"):
                json_files.append(str(path))
            
            self._log(f"Found {len(json_files)} JSON files in directory")
            return json_files
            
        except Exception as e:
            self._log(f"Error scanning directory: {e}", "ERROR")
            return []
    
    def load_and_parse_json(self, file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Load and parse JSON file, return data and error message if any."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, None
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            return None, error_msg
        except Exception as e:
            error_msg = f"File read error: {e}"
            return None, error_msg
    
    def validate_conversation_structure(self, data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Validate conversation structure against ChatGPT schema."""
        validation_result = {
            'file_path': file_path,
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'structure_analysis': {},
            'completeness_score': 0
        }
        
        try:
            # Schema validation
            validator = Draft7Validator(self.chatgpt_schema)
            schema_errors = list(validator.iter_errors(data))
            
            if schema_errors:
                validation_result['is_valid'] = False
                for error in schema_errors:
                    validation_result['errors'].append({
                        'type': 'schema_violation',
                        'message': error.message,
                        'path': list(error.path),
                        'schema_path': list(error.schema_path)
                    })
            
            # Structure analysis
            validation_result['structure_analysis'] = self._analyze_conversation_structure(data)
            
            # Completeness analysis
            validation_result['completeness_score'] = self._calculate_completeness_score(data)
            
            # Additional validations
            self._validate_timestamps(data, validation_result)
            self._validate_mapping_consistency(data, validation_result)
            self._validate_content_integrity(data, validation_result)
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append({
                'type': 'validation_error',
                'message': f"Validation process failed: {e}",
                'path': [],
                'schema_path': []
            })
        
        return validation_result
    
    def _analyze_conversation_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of a conversation."""
        analysis = {
            'total_nodes': 0,
            'message_nodes': 0,
            'empty_nodes': 0,
            'user_messages': 0,
            'assistant_messages': 0,
            'system_messages': 0,
            'conversation_depth': 0,
            'total_characters': 0,
            'has_title': bool(data.get('title')),
            'has_timestamps': bool(data.get('create_time')) and bool(data.get('update_time')),
            'node_types': {}
        }
        
        try:
            mapping = data.get('mapping', {})
            analysis['total_nodes'] = len(mapping)
            
            for node_id, node_data in mapping.items():
                if not isinstance(node_data, dict):
                    continue
                
                message = node_data.get('message')
                if message is None:
                    analysis['empty_nodes'] += 1
                    continue
                
                analysis['message_nodes'] += 1
                
                # Analyze author role
                author = message.get('author', {})
                role = author.get('role', 'unknown')
                
                if role == 'user':
                    analysis['user_messages'] += 1
                elif role == 'assistant':
                    analysis['assistant_messages'] += 1
                elif role == 'system':
                    analysis['system_messages'] += 1
                
                # Count content characters
                content = message.get('content', {})
                parts = content.get('parts', [])
                for part in parts:
                    if isinstance(part, str):
                        analysis['total_characters'] += len(part)
                
                # Track node types
                node_type = f"{role}_message" if message else "empty_node"
                analysis['node_types'][node_type] = analysis['node_types'].get(node_type, 0) + 1
            
            # Calculate conversation depth (simple heuristic)
            analysis['conversation_depth'] = max(
                analysis['user_messages'],
                analysis['assistant_messages']
            )
            
        except Exception as e:
            self._log(f"Error analyzing structure: {e}", "ERROR")
        
        return analysis
    
    def _calculate_completeness_score(self, data: Dict[str, Any]) -> float:
        """Calculate a completeness score (0-100) for the conversation."""
        score = 0
        max_score = 100
        
        # Required fields (40 points)
        required_fields = ['title', 'create_time', 'update_time', 'mapping']
        for field in required_fields:
            if field in data and data[field] is not None:
                score += 10
        
        # Optional but important fields (20 points)
        optional_fields = ['conversation_id', 'current_node']
        for field in optional_fields:
            if field in data and data[field] is not None:
                score += 10
        
        # Mapping completeness (40 points)
        mapping = data.get('mapping', {})
        if mapping:
            total_nodes = len(mapping)
            complete_nodes = 0
            
            for node_data in mapping.values():
                if isinstance(node_data, dict):
                    message = node_data.get('message')
                    if message and isinstance(message, dict):
                        # Check for essential message fields
                        essential_fields = ['id', 'author', 'create_time', 'content']
                        if all(field in message for field in essential_fields):
                            complete_nodes += 1
            
            if total_nodes > 0:
                mapping_completeness = (complete_nodes / total_nodes) * 40
                score += mapping_completeness
        
        return min(score, max_score)
    
    def _validate_timestamps(self, data: Dict[str, Any], validation_result: Dict[str, Any]):
        """Validate timestamp consistency and format."""
        try:
            create_time = data.get('create_time')
            update_time = data.get('update_time')
            
            # Check timestamp types
            if create_time is not None and not isinstance(create_time, (int, float)):
                validation_result['warnings'].append({
                    'type': 'timestamp_format',
                    'message': f"create_time should be numeric, got {type(create_time).__name__}",
                    'field': 'create_time'
                })
            
            if update_time is not None and not isinstance(update_time, (int, float)):
                validation_result['warnings'].append({
                    'type': 'timestamp_format',
                    'message': f"update_time should be numeric, got {type(update_time).__name__}",
                    'field': 'update_time'
                })
            
            # Check timestamp logic
            if (isinstance(create_time, (int, float)) and 
                isinstance(update_time, (int, float)) and 
                update_time < create_time):
                validation_result['warnings'].append({
                    'type': 'timestamp_logic',
                    'message': "update_time is earlier than create_time",
                    'field': 'timestamps'
                })
            
            # Validate message timestamps
            mapping = data.get('mapping', {})
            for node_id, node_data in mapping.items():
                if isinstance(node_data, dict):
                    message = node_data.get('message')
                    if message and isinstance(message, dict):
                        msg_create = message.get('create_time')
                        msg_update = message.get('update_time')
                        
                        if (isinstance(msg_create, (int, float)) and 
                            isinstance(msg_update, (int, float)) and 
                            msg_update < msg_create):
                            validation_result['warnings'].append({
                                'type': 'message_timestamp_logic',
                                'message': f"Message {node_id} update_time earlier than create_time",
                                'field': f'mapping.{node_id}.message.timestamps'
                            })
        
        except Exception as e:
            validation_result['errors'].append({
                'type': 'timestamp_validation_error',
                'message': f"Error validating timestamps: {e}",
                'path': [],
                'schema_path': []
            })
    
    def _validate_mapping_consistency(self, data: Dict[str, Any], validation_result: Dict[str, Any]):
        """Validate mapping consistency and relationships."""
        try:
            mapping = data.get('mapping', {})
            if not mapping:
                return
            
            # Check parent-child relationships
            all_node_ids = set(mapping.keys())
            referenced_parents = set()
            referenced_children = set()
            
            for node_id, node_data in mapping.items():
                if isinstance(node_data, dict):
                    parent = node_data.get('parent')
                    children = node_data.get('children', [])
                    
                    # Check parent exists
                    if parent is not None:
                        referenced_parents.add(parent)
                        if parent not in all_node_ids:
                            validation_result['errors'].append({
                                'type': 'missing_parent',
                                'message': f"Node {node_id} references non-existent parent {parent}",
                                'field': f'mapping.{node_id}.parent'
                            })
                    
                    # Check children exist
                    for child in children:
                        referenced_children.add(child)
                        if child not in all_node_ids:
                            validation_result['errors'].append({
                                'type': 'missing_child',
                                'message': f"Node {node_id} references non-existent child {child}",
                                'field': f'mapping.{node_id}.children'
                            })
            
            # Check for orphaned nodes (nodes with parents that don't reference them as children)
            for node_id, node_data in mapping.items():
                if isinstance(node_data, dict):
                    parent_id = node_data.get('parent')
                    if parent_id and parent_id in mapping:
                        parent_node = mapping[parent_id]
                        if isinstance(parent_node, dict):
                            parent_children = parent_node.get('children', [])
                            if node_id not in parent_children:
                                validation_result['warnings'].append({
                                    'type': 'orphaned_node',
                                    'message': f"Node {node_id} has parent {parent_id} but is not listed as child",
                                    'field': f'mapping.{node_id}'
                                })
        
        except Exception as e:
            validation_result['errors'].append({
                'type': 'mapping_validation_error',
                'message': f"Error validating mapping: {e}",
                'path': [],
                'schema_path': []
            })
    
    def _validate_content_integrity(self, data: Dict[str, Any], validation_result: Dict[str, Any]):
        """Validate content integrity and format."""
        try:
            mapping = data.get('mapping', {})
            
            for node_id, node_data in mapping.items():
                if isinstance(node_data, dict):
                    message = node_data.get('message')
                    if message and isinstance(message, dict):
                        content = message.get('content', {})
                        
                        # Check content structure
                        if isinstance(content, dict):
                            content_type = content.get('content_type')
                            parts = content.get('parts', [])
                            
                            # Validate content type
                            if content_type not in ['text', 'code', 'execution_output']:
                                validation_result['warnings'].append({
                                    'type': 'unknown_content_type',
                                    'message': f"Unknown content_type: {content_type} in node {node_id}",
                                    'field': f'mapping.{node_id}.message.content.content_type'
                                })
                            
                            # Validate parts
                            if not isinstance(parts, list):
                                validation_result['errors'].append({
                                    'type': 'invalid_content_parts',
                                    'message': f"content.parts should be array in node {node_id}",
                                    'field': f'mapping.{node_id}.message.content.parts'
                                })
                            else:
                                for i, part in enumerate(parts):
                                    if not isinstance(part, str):
                                        validation_result['warnings'].append({
                                            'type': 'non_string_content_part',
                                            'message': f"content.parts[{i}] is not string in node {node_id}",
                                            'field': f'mapping.{node_id}.message.content.parts[{i}]'
                                        })
                        
                        # Check author information
                        author = message.get('author', {})
                        if isinstance(author, dict):
                            role = author.get('role')
                            if role not in ['user', 'assistant', 'system', 'tool']:
                                validation_result['warnings'].append({
                                    'type': 'unknown_author_role',
                                    'message': f"Unknown author role: {role} in node {node_id}",
                                    'field': f'mapping.{node_id}.message.author.role'
                                })
        
        except Exception as e:
            validation_result['errors'].append({
                'type': 'content_validation_error',
                'message': f"Error validating content: {e}",
                'path': [],
                'schema_path': []
            })
    
    def compare_conversations(self, file1: str, file2: str) -> Dict[str, Any]:
        """Compare two conversation files and identify differences."""
        self._log(f"Comparing conversations: {Path(file1).name} vs {Path(file2).name}")
        
        comparison_result = {
            'file1': file1,
            'file2': file2,
            'are_identical': False,
            'differences': {},
            'structural_differences': {},
            'content_differences': {}
        }
        
        try:
            # Load both files
            data1, error1 = self.load_and_parse_json(file1)
            data2, error2 = self.load_and_parse_json(file2)
            
            if error1 or error2:
                comparison_result['errors'] = {
                    'file1_error': error1,
                    'file2_error': error2
                }
                return comparison_result
            
            # Perform deep comparison
            diff = DeepDiff(data1, data2, ignore_order=True, report_type='text')
            
            if not diff:
                comparison_result['are_identical'] = True
            else:
                comparison_result['differences'] = dict(diff)
            
            # Structural comparison
            struct1 = self._analyze_conversation_structure(data1)
            struct2 = self._analyze_conversation_structure(data2)
            
            structural_diff = DeepDiff(struct1, struct2, ignore_order=True)
            if structural_diff:
                comparison_result['structural_differences'] = dict(structural_diff)
            
            # Content comparison (message counts, character counts, etc.)
            comparison_result['content_differences'] = {
                'message_count_diff': struct1['message_nodes'] - struct2['message_nodes'],
                'character_count_diff': struct1['total_characters'] - struct2['total_characters'],
                'user_message_diff': struct1['user_messages'] - struct2['user_messages'],
                'assistant_message_diff': struct1['assistant_messages'] - struct2['assistant_messages']
            }
            
        except Exception as e:
            comparison_result['error'] = f"Comparison failed: {e}"
        
        return comparison_result
    
    def validate_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Validate multiple files and generate comprehensive report."""
        self._log(f"Starting validation of {len(file_paths)} files")
        
        # Create progress window
        progress_window = self._create_progress_window(len(file_paths), "Validating Files")
        
        results = {}
        
        def update_progress(current, total, filename):
            if progress_window:
                progress_window.update_progress(current, total, filename)
        
        try:
            for i, file_path in enumerate(file_paths):
                update_progress(i + 1, len(file_paths), Path(file_path).name)
                
                self.statistics['files_analyzed'] += 1
                
                # Load and parse file
                data, parse_error = self.load_and_parse_json(file_path)
                
                if parse_error:
                    results[file_path] = {
                        'parse_error': parse_error,
                        'is_valid': False
                    }
                    self.statistics['invalid_files'] += 1
                    self.statistics['errors_found'] += 1
                    continue
                
                # Validate structure
                validation_result = self.validate_conversation_structure(data, file_path)
                results[file_path] = validation_result
                
                # Update statistics
                if validation_result['is_valid']:
                    self.statistics['valid_files'] += 1
                else:
                    self.statistics['invalid_files'] += 1
                
                self.statistics['errors_found'] += len(validation_result['errors'])
                self.statistics['warnings_found'] += len(validation_result['warnings'])
        
        finally:
            if progress_window:
                progress_window.close()
        
        self.validation_results = results
        self._log(f"Validation completed. {self.statistics['valid_files']} valid, {self.statistics['invalid_files']} invalid files")
        
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        processing_time = time.time() - self.start_time
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'platform_info': self.platform_info,
                'log_file': str(self.log_file)
            },
            'statistics': self.statistics,
            'validation_summary': {
                'total_files': self.statistics['files_analyzed'],
                'success_rate': (self.statistics['valid_files'] / max(1, self.statistics['files_analyzed'])) * 100,
                'error_rate': (self.statistics['errors_found'] / max(1, self.statistics['files_analyzed'])),
                'warning_rate': (self.statistics['warnings_found'] / max(1, self.statistics['files_analyzed']))
            },
            'detailed_results': self.validation_results
        }
        
        # Analyze common issues
        common_errors = {}
        common_warnings = {}
        
        for result in self.validation_results.values():
            if isinstance(result, dict):
                for error in result.get('errors', []):
                    error_type = error.get('type', 'unknown')
                    common_errors[error_type] = common_errors.get(error_type, 0) + 1
                
                for warning in result.get('warnings', []):
                    warning_type = warning.get('type', 'unknown')
                    common_warnings[warning_type] = common_warnings.get(warning_type, 0) + 1
        
        report['common_issues'] = {
            'errors': common_errors,
            'warnings': common_warnings
        }
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any]) -> str:
        """Save validation report to JSON file."""
        try:
            report_file = self.desktop_path / f"chatgpt_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self._log(f"Validation report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self._log(f"Error saving report: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, report: Dict[str, Any]) -> str:
        """Create visualizations for validation results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ChatGPT Structure Validation Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Validation Overview
            ax1 = axes[0, 0]
            labels = ['Valid Files', 'Invalid Files']
            sizes = [self.statistics['valid_files'], self.statistics['invalid_files']]
            colors = ['#2ecc71', '#e74c3c']
            
            if sum(sizes) > 0:
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('File Validation Results')
            
            # Plot 2: Error Distribution
            ax2 = axes[0, 1]
            common_errors = report['common_issues']['errors']
            
            if common_errors:
                error_types = list(common_errors.keys())
                error_counts = list(common_errors.values())
                
                bars = ax2.bar(error_types, error_counts, color='#e74c3c', alpha=0.7)
                ax2.set_title('Common Error Types')
                ax2.set_xlabel('Error Type')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, error_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_counts)*0.01,
                            f'{value}', ha='center', va='bottom')
            
            # Plot 3: Warning Distribution
            ax3 = axes[1, 0]
            common_warnings = report['common_issues']['warnings']
            
            if common_warnings:
                warning_types = list(common_warnings.keys())
                warning_counts = list(common_warnings.values())
                
                bars = ax3.bar(warning_types, warning_counts, color='#f39c12', alpha=0.7)
                ax3.set_title('Common Warning Types')
                ax3.set_xlabel('Warning Type')
                ax3.set_ylabel('Count')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, warning_counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(warning_counts)*0.01,
                            f'{value}', ha='center', va='bottom')
            
            # Plot 4: Completeness Score Distribution
            ax4 = axes[1, 1]
            completeness_scores = []
            
            for result in self.validation_results.values():
                if isinstance(result, dict) and 'completeness_score' in result:
                    completeness_scores.append(result['completeness_score'])
            
            if completeness_scores:
                ax4.hist(completeness_scores, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
                ax4.set_title('Completeness Score Distribution')
                ax4.set_xlabel('Completeness Score (%)')
                ax4.set_ylabel('Number of Files')
                ax4.axvline(x=sum(completeness_scores)/len(completeness_scores), 
                           color='red', linestyle='--', label=f'Average: {sum(completeness_scores)/len(completeness_scores):.1f}%')
                ax4.legend()
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.desktop_path / f"chatgpt_validation_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"Visualization saved: {viz_file}")
            return str(viz_file)
            
        except Exception as e:
            self._log(f"Error creating visualizations: {e}", "ERROR")
            return ""
    
    def _create_progress_window(self, total_items: int, title: str = "Processing"):
        """Create progress tracking window."""
        try:
            root = tk.Tk()
            root.title(title)
            root.geometry("500x200")
            root.resizable(False, False)
            
            # Progress info
            info_label = tk.Label(root, text=f"Processing {total_items} items...", font=("Arial", 12))
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
    
    def run_interactive_validation(self):
        """Run interactive validation with GUI dialogs."""
        try:
            # Welcome message
            root = tk.Tk()
            root.withdraw()
            
            choice = messagebox.askyesnocancel(
                "ChatGPT Structure Validator",
                "Welcome to ChatGPT Structure Validator v1.0.0\n\n"
                "Choose validation mode:\n"
                "• Yes: Validate specific files\n"
                "• No: Validate entire directory\n"
                "• Cancel: Exit"
            )
            
            if choice is None:  # Cancel
                return False
            
            files_to_validate = []
            
            if choice:  # Yes - specific files
                files_to_validate = self.select_files_for_validation()
            else:  # No - directory
                directory = self.select_directory_for_validation()
                if directory:
                    files_to_validate = self.find_json_files_in_directory(directory)
            
            if not files_to_validate:
                messagebox.showwarning("No Files", "No files selected for validation.")
                return False
            
            # Confirm validation
            confirm = messagebox.askyesno(
                "Confirm Validation",
                f"Ready to validate {len(files_to_validate)} files.\n\n"
                f"Continue with validation?"
            )
            
            if not confirm:
                messagebox.showinfo("Cancelled", "Validation cancelled by user.")
                return False
            
            # Validate files
            validation_results = self.validate_files(files_to_validate)
            
            # Generate report
            report = self.generate_validation_report()
            
            # Save report
            report_file = self.save_validation_report(report)
            
            # Create visualizations
            viz_file = self.create_visualizations(report)
            
            # Show results
            stats = report['statistics']
            messagebox.showinfo(
                "Validation Complete",
                f"ChatGPT structure validation completed!\n\n"
                f"Files analyzed: {stats['files_analyzed']}\n"
                f"Valid files: {stats['valid_files']}\n"
                f"Invalid files: {stats['invalid_files']}\n"
                f"Errors found: {stats['errors_found']}\n"
                f"Warnings found: {stats['warnings_found']}\n\n"
                f"Report saved: {report_file}\n"
                f"Visualization: {viz_file}\n"
                f"Log file: {self.log_file}"
            )
            
            return True
            
        except Exception as e:
            self._log(f"Error in interactive validation: {e}", "ERROR")
            try:
                messagebox.showerror("Error", f"An error occurred: {e}")
            except:
                pass
            return False

def main():
    """Main function to run ChatGPT structure validator."""
    print("Starting ChatGPT Structure Validator Suite...")
    
    try:
        # Initialize validator
        validator = ChatGPTStructureValidator()
        
        # Run interactive validation
        success = validator.run_interactive_validation()
        
        if success:
            # Show macOS notification if available
            try:
                if platform.system() == "Darwin":
                    subprocess.run([
                        "osascript", "-e",
                        f'display notification "ChatGPT structure validation completed!" with title "GET SWIFTY - Structure Validator"'
                    ], check=False)
            except:
                pass
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during validation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()