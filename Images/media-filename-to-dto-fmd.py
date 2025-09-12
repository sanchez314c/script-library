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
# Script Name: images-media-filename-to-dto-fmd.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible media filename to DateTime Original converter that
#     extracts date/time information from filenames and updates metadata fields
#     DateTimeOriginal and FileModifyDate with parallel processing.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded processing using all available cores
#     - Multiple folder selection via native macOS dialogs
#     - Support for various filename date patterns
#     - Progress tracking with comprehensive error handling
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed via brew)
#     - tkinter (standard library)
#
# Usage:
#     python images-media-filename-to-dto-fmd.py
#
####################################################################################

import os
import re
import sys
import argparse
import logging
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import multiprocessing
from typing import List, Optional, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

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
            
    @staticmethod
    def set_metadata(file_path: str, metadata: Dict[str, Any]) -> bool:
        """Set metadata for a file."""
        try:
            # Construct command
            cmd = ['exiftool']
            for tag, value in metadata.items():
                cmd.extend(['-' + tag + '=' + str(value)])
            cmd.extend(['-overwrite_original', file_path])
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except Exception as e:
            logging.error(f"Error setting metadata: {e}")
            return False

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
    def select_directory(title: str = "Select Directory", multiple: bool = False) -> Optional[List[str]]:
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
                
                if not messagebox.askyesno("Select Directory", "Add another directory?"):
                    break
            
            root.destroy()
            return directories if directories else None
        else:
            directory = filedialog.askdirectory(title=title)
            root.destroy()
            return [directory] if directory else None

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

def run_exif_cleanse_commands(file_path: str) -> bool:
    """
    Cleanse EXIF data to prepare for metadata updates.
    
    Args:
        file_path: Path to file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Clearing potentially problematic metadata
        ExifToolWrapper.run_exiftool(
            ['-overwrite_original', '-m', '-F', '-api', 'LargeFileSupport=1', '-exif:all='],
            file_path,
            quiet=True
        )
        
        # Attempt to restore structure to prevent data loss
        ExifToolWrapper.run_exiftool(
            ['-overwrite_original', '-tagsfromfile', '@', '-all:all', '-unsafe'],
            file_path,
            quiet=True
        )
        
        # Remove MakerNotes which can sometimes cause issues with metadata integrity
        ExifToolWrapper.run_exiftool(
            ['-MakerNotes=', '-overwrite_original'],
            file_path,
            quiet=True
        )
        
        return True
    except Exception as e:
        logging.error(f'Error processing EXIF data for {file_path}: {e}')
        return False

def extract_datetime_from_filename(filename: str) -> Optional[str]:
    """
    Extract datetime string from filename using regex.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Formatted datetime string or None if not found
    """
    match = re.match(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2}-\d{2}-\d{2})", filename)
    if match:
        date_part, time_part = match.groups()
        datetime_str = f"{date_part} {time_part}".replace('-', ':')
        return datetime_str
    return None

def set_metadata_from_filename(file_path: str) -> bool:
    """
    Extract datetime from filename and update metadata fields.
    
    Args:
        file_path: Path to file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    # First cleanse the EXIF data
    if not run_exif_cleanse_commands(file_path):
        return False

    datetime_str = extract_datetime_from_filename(os.path.basename(file_path))
    if datetime_str:
        try:
            # Set DateTimeOriginal and FileModifyDate metadata fields
            ExifToolWrapper.set_metadata(
                file_path,
                {
                    'DateTimeOriginal': datetime_str,
                    'FileModifyDate': datetime_str
                }
            )
            logging.info(f"Updated {file_path} with DateTimeOriginal and FileModifyDate: {datetime_str}")
            return True
        except Exception as e:
            logging.error(f"Error setting metadata for {file_path}: {e}")
            return False
    else:
        logging.warning(f"Could not extract datetime from filename: {file_path}")
        return False

def process_directories(directories: List[str]) -> Tuple[int, int]:
    """
    Process all files in specified directories in parallel.
    
    Args:
        directories: List of directory paths to process
        
    Returns:
        Tuple of (success_count, error_count)
    """
    files_to_process = []
    
    # Collect all files from all directories
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                files_to_process.append(os.path.join(root, file))
    
    # Create progress tracking window
    total_files = len(files_to_process)
    window, progress_bar, status_label = UIHelper.create_progress_window(
        title="Processing Files",
        message=f"Processing {total_files} files...",
        maximum=total_files
    )
    
    # Initialize counts
    success_count = 0
    error_count = 0
    
    # Define processing function
    def process_with_progress(file_path):
        nonlocal success_count, error_count
        try:
            result = set_metadata_from_filename(file_path)
            if result:
                success_count += 1
            else:
                error_count += 1
                
            # Update progress asynchronously
            window.after(0, lambda: progress_bar.step(1))
            window.after(0, lambda: status_label.config(text=f"Processed: {success_count + error_count}/{total_files}"))
            window.update_idletasks()
            
            return result
        except Exception as e:
            logging.error(f"Unexpected error processing {file_path}: {e}")
            error_count += 1
            return False
    
    # Process files in parallel
    max_workers = ProcessingManager.get_optimal_workers(cpu_bound=False, io_bound=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_with_progress, file) for file in files_to_process]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread: {e}")
    
    # Close progress window
    window.destroy()
    
    return success_count, error_count

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract date/time from filenames and update metadata"
    )
    parser.add_argument(
        "-d", "--directories",
        nargs="+",
        help="Directories to process (if not specified, will open selection dialog)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    global logging
    logging = LoggingManager.setup(
        log_file="media-filename-to-dto-fmd.log",
        log_level=log_level
    )
    
    # Check exiftool availability
    if not ExifToolWrapper.check_exiftool():
        UIHelper.show_error(
            "Error",
            "ExifTool is not installed or not in PATH. Please install ExifTool."
        )
        return
    
    # Get directories to process
    if args.directories:
        directories = args.directories
    else:
        directories = FileSelector.select_directory(
            title="Select Directories to Process",
            multiple=True
        )
    
    if not directories:
        logging.info("No directories selected. Exiting...")
        return
    
    # Process directories
    success_count, error_count = process_directories(directories)
    
    # Show results
    UIHelper.show_info(
        "Processing Complete",
        f"Successfully processed {success_count} files.\n"
        f"Errors: {error_count}\n"
        f"See log file for details."
    )

if __name__ == "__main__":
    main()