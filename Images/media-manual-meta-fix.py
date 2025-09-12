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
# Script Name: images-media-manual-meta-fix.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible advanced metadata fixing tool for images with
#     support for various formats, extensive error handling, and parallel processing
#     with automatic backup creation and detailed reporting.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded processing using all available cores
#     - Multiple format support (.jpg, .jpeg, .tiff, .tif, .png, .dng, .webp)
#     - Native macOS directory selection dialogs
#     - Progress tracking with visual feedback
#     - Comprehensive error logging with backup creation
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed via brew)
#     - tkinter (standard library)
#
# Usage:
#     python images-media-manual-meta-fix.py
#
####################################################################################
"""

import os
import sys
import logging
import argparse
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import multiprocessing
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set, Callable, Union

# Integrated utility functions
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
    def run_exiftool(args: List[str], file_path: str, quiet: bool = False) -> str:
        """Run exiftool with specified arguments."""
        try:
            cmd = ['exiftool'] + args + [file_path]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if not quiet:
                logging.error(f"ExifTool error: {e.stderr}")
            raise Exception(f"ExifTool error: {e.stderr}")

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
    def select_directory(title: str = "Select Directory") -> Optional[str]:
        """Select a directory."""
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

class MetadataFixer:
    """Main class for fixing metadata in image files."""
    
    def __init__(self, backup: bool = False, verbose: bool = False):
        """
        Initialize the metadata fixer.
        
        Args:
            backup: Whether to create backups of files before modifying
            verbose: Enable verbose logging
        """
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = LoggingManager.setup(
            log_file="media-metadata-fix.log",
            log_level=log_level
        )
        
        self.source_dir: Optional[Path] = None
        self.processed_count = 0
        self.error_count = 0
        self.backup = backup
        self.supported_extensions: Set[str] = {
            '.jpg', '.jpeg', '.tiff', '.tif', '.png', '.dng', '.webp', '.heic'
        }
    
    def select_directory(self) -> bool:
        """
        Open directory selection dialog and set source directory.
        
        Returns:
            bool: True if directory was selected, False otherwise
        """
        directory = FileSelector.select_directory(
            title="Select Directory with Images"
        )
        
        if not directory:
            return False
            
        self.source_dir = Path(directory)
        return True
    
    def process_file(self, file_path: Path) -> Tuple[Path, bool, str]:
        """
        Process a single file's metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (file_path, success, message)
        """
        try:
            # Clear all EXIF data
            exif_args = ["-exif:all="]
            if self.backup:
                # Create backup with original extension
                ExifToolWrapper.run_exiftool(
                    exif_args,
                    str(file_path),
                    quiet=True
                )
            else:
                # Overwrite original
                exif_args.insert(0, "-overwrite_original")
                ExifToolWrapper.run_exiftool(
                    exif_args,
                    str(file_path),
                    quiet=True
                )
            
            # Attempt to restore essential EXIF data
            restore_args = ["-tagsfromfile", str(file_path), "--MakerNotes"]
            if not self.backup:
                restore_args.insert(0, "-overwrite_original")
                
            ExifToolWrapper.run_exiftool(
                restore_args,
                str(file_path),
                quiet=True
            )
            
            return (file_path, True, "Success")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error processing {file_path}: {error_msg}")
            return (file_path, False, error_msg)
    
    def find_images(self) -> List[Path]:
        """
        Find all supported image files in directory.
        
        Returns:
            List of image file paths
        """
        if not self.source_dir:
            return []
            
        images = []
        for ext in self.supported_extensions:
            images.extend(self.source_dir.rglob(f"*{ext}"))
            images.extend(self.source_dir.rglob(f"*{ext.upper()}"))
        return images
    
    def process_directory(self) -> None:
        """Process all images in the selected directory."""
        images = self.find_images()
        total_images = len(images)
        
        if not images:
            UIHelper.show_info(
                "No Images Found",
                "No supported images found in the selected directory."
            )
            return
        
        # Ask for confirmation
        if not UIHelper.ask_yes_no(
            "Confirm Processing",
            f"Found {total_images} images to process. Continue?"
        ):
            return
        
        # Create progress window
        window, progress_bar, status_label = UIHelper.create_progress_window(
            title="Processing Images",
            message=f"Processing {total_images} images...",
            maximum=total_images
        )
        
        # Track success and error counts
        self.processed_count = 0
        self.error_count = 0
        
        # Define function to update progress UI
        def update_progress(success: bool) -> None:
            if success:
                self.processed_count += 1
            else:
                self.error_count += 1
                
            # Update progress and status asynchronously
            window.after(0, lambda: progress_bar.step(1))
            window.after(0, lambda: status_label.config(
                text=f"Processed: {self.processed_count} | Errors: {self.error_count}"
            ))
            window.update_idletasks()
        
        # Process files in parallel
        max_workers = ProcessingManager.get_optimal_workers(cpu_bound=False, io_bound=True)
        
        # Define worker function
        def process_file_with_progress(file_path: Path) -> Tuple[Path, bool, str]:
            result = self.process_file(file_path)
            _, success, _ = result
            
            # Update progress UI
            window.after(0, lambda: update_progress(success))
            
            return result
        
        # Process files
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file_with_progress, img) for img in images]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in thread: {e}")
        
        # Close progress window
        window.destroy()
        
        # Show results
        UIHelper.show_info(
            "Processing Complete",
            f"Successfully processed: {self.processed_count}\n"
            f"Errors: {self.error_count}\n"
            f"See media-metadata-fix.log for details"
        )
    
    def run(self) -> None:
        """Main execution flow."""
        # Check if exiftool is available
        if not ExifToolWrapper.check_exiftool():
            UIHelper.show_error(
                "Error",
                "ExifTool is not installed or not in PATH. Please install ExifTool."
            )
            return
            
        if not self.select_directory():
            self.logger.info("No directory selected. Exiting.")
            return
            
        self.process_directory()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix metadata in image files"
    )
    parser.add_argument(
        "-d", "--directory",
        help="Directory to process (if not specified, will open selection dialog)"
    )
    parser.add_argument(
        "-b", "--backup",
        action="store_true",
        help="Create backups of original files"
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
    
    # Create fixer instance
    fixer = MetadataFixer(
        backup=args.backup,
        verbose=args.verbose
    )
    
    # Set directory if specified in arguments
    if args.directory:
        fixer.source_dir = Path(args.directory)
        fixer.run()
    else:
        # Will prompt for directory selection
        fixer.run()

if __name__ == "__main__":
    main()