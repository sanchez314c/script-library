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
# Script Name: images-media-metadata-reporter.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible comprehensive media metadata scanning and reporting
#     tool that analyzes metadata fields with focus on date/time tags and generates
#     detailed statistical reports with CSV export capabilities.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Parallel processing using all available CPU cores
#     - Multiple format support (.jpg, .jpeg, .png, .heic, .tiff, etc.)
#     - Native macOS directory selection dialogs
#     - Progress tracking with visual feedback
#     - Comprehensive metadata field reporting
#     - CSV export with customizable filters
#     - Tag pattern matching and statistical analysis
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed via brew)
#     - tkinter (standard library)
#
# Usage:
#     python images-media-metadata-reporter.py
#
####################################################################################

import os
import sys
import re
import logging
import argparse
import csv
import subprocess
import importlib
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Counter, Union, Tuple, Callable
from collections import defaultdict, Counter
import json
import time

# Setup internal utility functions to replace external dependency
class ExifToolWrapper:
    """Wrapper for interacting with ExifTool."""
    
    @staticmethod
    def check_exiftool() -> bool:
        """Check if exiftool is installed and available."""
        try:
            subprocess.run(
                ['exiftool', '-ver'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_metadata(file_path: str) -> Dict[str, Any]:
        """Get all metadata from a file."""
        try:
            result = subprocess.run(
                ['exiftool', '-j', '-g', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            # Parse JSON result
            data = json.loads(result.stdout)
            if not data:
                return {}
                
            # ExifTool returns a list with one item per file
            return data[0]
        except Exception as e:
            logging.error(f"Error getting metadata: {e}")
            return {}

class LoggingManager:
    """Manages logging setup."""
    
    @staticmethod
    def setup(log_file: str, log_level: int = logging.INFO) -> logging.Logger:
        """Setup logging with file and console handlers."""
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
        
        return logger

class UIHelper:
    """UI helper functions."""
    
    @staticmethod
    def show_info(title: str, message: str) -> None:
        """Show an information dialog."""
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    
    @staticmethod
    def show_error(title: str, message: str) -> None:
        """Show an error dialog."""
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    
    @staticmethod
    def ask_yes_no(title: str, message: str) -> bool:
        """Show a yes/no dialog."""
        root = tk.Tk()
        root.withdraw()
        result = messagebox.askyesno(title, message)
        root.destroy()
        return result
    
    @staticmethod
    def create_progress_window(title: str, message: str, maximum: int) -> Tuple[tk.Toplevel, ttk.Progressbar, ttk.Label]:
        """Create a progress window."""
        window = tk.Toplevel()
        window.title(title)
        window.geometry("400x150")
        window.resizable(False, False)
        
        # Center on screen
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Status label
        status_label = ttk.Label(window, text=message, font=("Helvetica", 12))
        status_label.pack(pady=(20, 10))
        
        # Progress bar
        progress_bar = ttk.Progressbar(
            window,
            orient="horizontal",
            length=350,
            mode="determinate",
            maximum=maximum
        )
        progress_bar.pack(pady=10, padx=25)
        
        return window, progress_bar, status_label

class FileSelector:
    """File selection helper functions."""
    
    @staticmethod
    def select_directory(title: str = "Select Directory", multiple: bool = False) -> Union[List[str], None]:
        """Select one or more directories."""
        root = tk.Tk()
        root.withdraw()
        
        if multiple:
            # Custom implementation for multiple directory selection
            # since tkinter doesn't support it natively
            directories = []
            while True:
                directory = filedialog.askdirectory(title=title)
                if not directory:
                    break
                directories.append(directory)
                
                if not UIHelper.ask_yes_no("Select Directory", "Add another directory?"):
                    break
            
            root.destroy()
            return directories if directories else None
        else:
            directory = filedialog.askdirectory(title=title)
            root.destroy()
            return [directory] if directory else None
    
    @staticmethod
    def select_output_directory(title: str = "Select Output Directory") -> Optional[str]:
        """Select an output directory."""
        root = tk.Tk()
        root.withdraw()
        
        directory = filedialog.askdirectory(title=title)
        
        root.destroy()
        return directory if directory else None

class ProcessingManager:
    """Manages parallel processing."""
    
    @staticmethod
    def get_optimal_workers(cpu_bound: bool = True, io_bound: bool = False) -> int:
        """Get optimal number of workers based on task type."""
        cpu_count = multiprocessing.cpu_count()
        
        if cpu_bound:
            # CPU-bound tasks: leave one core for system
            return max(1, cpu_count - 1)
        elif io_bound:
            # IO-bound tasks: can use more workers
            return cpu_count * 2
        else:
            # Default
            return cpu_count

class MetadataReporter:
    """Main class for scanning and reporting media file metadata."""
    
    def __init__(
        self,
        source_directories: Optional[List[str]] = None,
        output_directory: Optional[str] = None,
        date_tags_only: bool = False,
        tag_pattern: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the metadata reporter.
        
        Args:
            source_directories: List of source directories to process
            output_directory: Directory for report outputs
            date_tags_only: Whether to limit analysis to date-related tags
            tag_pattern: Regular expression to filter tags
            verbose: Enable verbose logging
        """
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = utils.LoggingManager.setup(
            log_file="media-metadata-reporter.log",
            log_level=log_level
        )
        
        self.source_directories = source_directories or []
        self.output_directory = output_directory
        self.date_tags_only = date_tags_only
        self.tag_pattern = re.compile(tag_pattern) if tag_pattern else None
        
        # File format support - media files that typically have metadata
        self.supported_extensions: Set[str] = {
            '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.heic', '.heif', 
            '.dng', '.cr2', '.arw', '.nef', '.orf', '.rw2', '.pef', '.mp4', 
            '.mov', '.avi', '.mkv'
        }
        
        # Regex patterns for identifying date/time tags
        self.datetime_patterns = [
            re.compile(r'date', re.IGNORECASE),
            re.compile(r'time', re.IGNORECASE),
            re.compile(r'create', re.IGNORECASE),
            re.compile(r'modify', re.IGNORECASE),
            re.compile(r'origin', re.IGNORECASE)
        ]
        
        # Data collection
        self.all_tags: Dict[str, int] = Counter()
        self.date_tags: Dict[str, int] = Counter()
        self.tag_values: Dict[str, Dict[str, int]] = defaultdict(Counter)
        self.file_types: Dict[str, int] = Counter()
        self.files_processed = 0
        self.files_with_errors = 0
    
    def is_date_tag(self, tag: str) -> bool:
        """
        Check if a tag appears to be related to dates/times.
        
        Args:
            tag: Tag name to check
            
        Returns:
            bool: True if tag is related to dates/times
        """
        return any(pattern.search(tag) for pattern in self.datetime_patterns)
    
    def process_file(self, file_path: str) -> bool:
        """
        Process a single file to extract metadata.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all metadata as JSON
            metadata = utils.ExifToolWrapper.get_metadata(file_path)
            
            if not metadata:
                self.logger.warning(f"No metadata found for {file_path}")
                return False
            
            # Record file type
            file_ext = os.path.splitext(file_path)[1].lower()
            self.file_types[file_ext] += 1
            
            # Process each metadata tag
            for tag, value in metadata.items():
                # Skip binary data or null values
                if value is None or isinstance(value, bytes):
                    continue
                
                # Convert value to string for counting
                value_str = str(value)
                
                # Only process tags that match the pattern if specified
                if self.tag_pattern and not self.tag_pattern.search(tag):
                    continue
                
                # Record all tags
                self.all_tags[tag] += 1
                
                # Record value distribution for this tag (limit long values)
                if len(value_str) < 100:  # Skip very long values
                    self.tag_values[tag][value_str] += 1
                
                # Check if it's a date-related tag
                if self.is_date_tag(tag):
                    self.date_tags[tag] += 1
            
            self.files_processed += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            self.files_with_errors += 1
            return False
    
    def find_files(self) -> List[str]:
        """
        Find all supported files in the source directories.
        
        Returns:
            List of file paths
        """
        files_to_process = []
        
        for directory in self.source_directories:
            for ext in self.supported_extensions:
                files_to_process.extend(
                    [str(p) for p in Path(directory).rglob(f"*{ext}")]
                )
                files_to_process.extend(
                    [str(p) for p in Path(directory).rglob(f"*{ext.upper()}")]
                )
        
        return files_to_process
    
    def scan_directories(self) -> None:
        """Scan all directories and collect metadata statistics."""
        # Find all files
        files_to_process = self.find_files()
        total_files = len(files_to_process)
        
        if total_files == 0:
            utils.UIHelper.show_info(
                "No Files Found",
                "No supported media files found in the source directories."
            )
            return
        
        # Ask for confirmation
        if not utils.UIHelper.ask_yes_no(
            "Confirm Scanning",
            f"Found {total_files} files to scan for metadata.\n"
            f"Continue?"
        ):
            return
        
        # Create progress window
        window, progress_bar, status_label = utils.UIHelper.create_progress_window(
            title="Scanning Metadata",
            message=f"Scanning {total_files} files...",
            maximum=total_files
        )
        
        # Reset counters
        self.all_tags.clear()
        self.date_tags.clear()
        self.tag_values.clear()
        self.file_types.clear()
        self.files_processed = 0
        self.files_with_errors = 0
        
        # Define function to update progress UI
        def update_progress(success: bool) -> None:
            # Update progress and status asynchronously
            window.after(0, lambda: progress_bar.step(1))
            window.after(0, lambda: status_label.config(
                text=f"Processed: {self.files_processed} | Errors: {self.files_with_errors}"
            ))
            window.update_idletasks()
        
        # Process files in parallel
        max_workers = utils.ProcessingManager.get_optimal_workers(cpu_bound=False, io_bound=True)
        
        # Define worker function
        def process_file_with_progress(file_path: str) -> bool:
            result = self.process_file(file_path)
            
            # Update progress UI
            window.after(0, lambda: update_progress(result))
            
            return result
        
        # Process files
        with utils.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file_with_progress, file) for file in files_to_process]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in thread: {e}")
                    self.files_with_errors += 1
        
        # Close progress window
        window.destroy()
        
        # Generate reports
        self.generate_reports()
        
        # Show results summary
        utils.UIHelper.show_info(
            "Scanning Complete",
            f"Successfully processed: {self.files_processed}\n"
            f"Errors: {self.files_with_errors}\n"
            f"Unique metadata tags found: {len(self.all_tags)}\n"
            f"Date-related tags: {len(self.date_tags)}\n"
            f"Reports saved to: {self.output_directory or 'current directory'}"
        )
    
    def generate_reports(self) -> None:
        """Generate various reports based on collected metadata."""
        # Create output directory if needed
        output_dir = self.output_directory or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. All tags report
        all_tags_file = os.path.join(output_dir, f"metadata_all_tags_{timestamp}.csv")
        with open(all_tags_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Tag", "Occurrence Count", "Is Date Tag"])
            
            # Sort by occurrence count (descending)
            for tag, count in sorted(self.all_tags.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([tag, count, "Yes" if tag in self.date_tags else "No"])
        
        # 2. Date tags report
        if self.date_tags:
            date_tags_file = os.path.join(output_dir, f"metadata_date_tags_{timestamp}.csv")
            with open(date_tags_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Date Tag", "Occurrence Count"])
                
                # Sort by occurrence count (descending)
                for tag, count in sorted(self.date_tags.items(), key=lambda x: x[1], reverse=True):
                    writer.writerow([tag, count])
        
        # 3. File types report
        file_types_file = os.path.join(output_dir, f"metadata_file_types_{timestamp}.csv")
        with open(file_types_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["File Extension", "Count"])
            
            # Sort by count (descending)
            for ext, count in sorted(self.file_types.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([ext, count])
        
        # 4. Value distribution for common tags (top 10 tags, top 10 values each)
        values_file = os.path.join(output_dir, f"metadata_tag_values_{timestamp}.csv")
        with open(values_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Tag", "Value", "Occurrence Count"])
            
            # Get top 10 most common tags
            top_tags = [tag for tag, _ in sorted(self.all_tags.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]]
            
            # For each top tag, show top 10 values
            for tag in top_tags:
                if tag in self.tag_values:
                    top_values = self.tag_values[tag].most_common(10)
                    for value, count in top_values:
                        writer.writerow([tag, value, count])
                    
                    # Add an empty row between tags
                    writer.writerow([])
        
        self.logger.info(f"Reports generated in {output_dir}")
    
    def select_directories(self) -> bool:
        """
        Prompt for source and output directories.
        
        Returns:
            bool: True if directories were selected, False otherwise
        """
        # Select source directories
        source_dirs = utils.FileSelector.select_directory(
            title="Select Source Directories to Scan",
            multiple=True
        )
        
        if not source_dirs:
            return False
            
        self.source_directories = source_dirs
        
        # Ask if user wants to save reports
        if utils.UIHelper.ask_yes_no(
            "Save Reports",
            "Would you like to save reports to a specific directory?"
        ):
            output_dir = utils.FileSelector.select_output_directory(
                title="Select Output Directory for Reports"
            )
            
            if output_dir:
                self.output_directory = output_dir
        
        return True
    
    def run(self) -> None:
        """Main execution flow."""
        # Check dependencies
        if not utils.ExifToolWrapper.check_exiftool():
            utils.UIHelper.show_error(
                "Error",
                "ExifTool is not installed or not in PATH. Please install ExifTool."
            )
            return
        
        # Select directories if not specified
        if not self.source_directories:
            if not self.select_directories():
                self.logger.info("Directory selection cancelled. Exiting.")
                return
        
        # Scan directories
        self.scan_directories()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan and report media file metadata"
    )
    parser.add_argument(
        "-s", "--source",
        nargs="+",
        help="Source directories to scan"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for reports"
    )
    parser.add_argument(
        "-d", "--date-tags-only",
        action="store_true",
        help="Only analyze date-related tags"
    )
    parser.add_argument(
        "-p", "--pattern",
        help="Regular expression pattern to filter tags"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()

def main() -> None:
    """Main entry point for the script."""
    # Parse arguments
    args = parse_arguments()
    
    # Create reporter instance
    reporter = MetadataReporter(
        source_directories=args.source,
        output_directory=args.output,
        date_tags_only=args.date_tags_only,
        tag_pattern=args.pattern,
        verbose=args.verbose
    )
    
    # Run reporter
    reporter.run()

if __name__ == "__main__":
    main()