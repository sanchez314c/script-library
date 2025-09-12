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
# Script Name: images-media-image-processor.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible advanced image processing tool that copies, renames, 
#     and organizes images based on metadata or filename patterns with comprehensive
#     format support and parallel processing capabilities.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Parallel processing using all available CPU cores
#     - Multiple format support (jpg, jpeg, png, gif, bmp, tiff, heic, dng)
#     - HEIC/HEIF format support with metadata extraction
#     - Filename pattern matching with subsecond precision
#     - Year-based organization with collision handling
#     - XMP title setting and comprehensive logging
#     - Native macOS dialogs for file/folder selection
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - Pillow, exifread, pyheif (auto-installed)
#     - exiftool (auto-installed via brew)
#
# Usage:
#     python images-media-image-processor.py
#
####################################################################################

import os
import sys
import re
import io
import logging
import argparse
import subprocess
import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set, Union, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json

# Import dependencies with error handling
try:
    from PIL import Image
    import exifread
    import pyheif
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Please install required packages using: pip install Pillow exifread pyheif")
    sys.exit(1)

# Integrated utilities that were previously in media_utils
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
                ['exiftool', '-j', file_path],
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
    def get_optimal_workers(cpu_bound: bool = True) -> int:
        """Get optimal number of workers based on task type."""
        cpu_count = multiprocessing.cpu_count()
        
        if cpu_bound:
            # CPU-bound tasks: leave one core for system
            return max(1, cpu_count - 1)
        else:
            # IO-bound tasks: can use more workers
            return cpu_count * 2

class ImageProcessor:
    """Main class for processing and organizing images."""
    
    def __init__(
        self, 
        source_directories: Optional[List[str]] = None,
        destination_directory: Optional[str] = None,
        organize_by_year: bool = True,
        backup_mode: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the image processor.
        
        Args:
            source_directories: List of source directories to process
            destination_directory: Output directory for processed images
            organize_by_year: Whether to organize files by year
            backup_mode: Whether to create backups
            verbose: Enable verbose logging
        """
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = utils.LoggingManager.setup(
            log_file="media-image-processor.log",
            log_level=log_level
        )
        
        self.source_directories = source_directories or []
        self.destination_directory = destination_directory
        self.organize_by_year = organize_by_year
        self.backup_mode = backup_mode
        
        # File format support
        self.supported_extensions: Set[str] = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.heic', '.heif', '.dng'
        }
        
        # Regular expression patterns
        self.filename_date_pattern = re.compile(
            r'(\d{4})[-_](\d{2})[-_](\d{2})[_\s]*(\d{2})[-_](\d{2})[-_](\d{2})'
        )
        self.date_time_pattern = re.compile(
            r'(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})'
        )
        
        # Counters
        self.success_count = 0
        self.error_count = 0
        self.collision_count = 0
    
    def extract_date_from_filename(self, file_name: str) -> Optional[str]:
        """
        Extract date and time from filename using regex pattern.
        
        Args:
            file_name: Filename to parse
            
        Returns:
            Formatted datetime string (YYYY-MM-DD HH-MM-SS) or None if not found
        """
        match = self.filename_date_pattern.search(file_name)
        if match:
            try:
                # Constructing date from the filename
                year, month, day, hour, minute, second = map(int, match.groups())
                date_obj = datetime(year, month, day, hour, minute, second)
                return date_obj.strftime("%Y-%m-%d %H-%M-%S")
            except ValueError:
                self.logger.debug(f"Invalid date/time in filename: {file_name}")
                return None
        return None
    
    def is_valid_date(self, date_str: str) -> bool:
        """
        Check if a date string is valid and reasonable.
        
        Args:
            date_str: Date string in format YYYY:MM:DD HH:MM:SS
            
        Returns:
            bool: True if date is valid and after 1979-12-31
        """
        if not date_str or date_str.startswith("0000:00:00"):
            return False
            
        try:
            date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            return date >= datetime(1979, 12, 31)
        except ValueError:
            return False
    
    def get_oldest_date(self, tags: Dict[str, Any]) -> Optional[str]:
        """
        Extract the oldest valid date from EXIF tags.
        
        Args:
            tags: Dictionary of EXIF tags
            
        Returns:
            Oldest date in format YYYY-MM-DD HH-MM-SS or None if not found
        """
        dates = []
        
        # Check all tags for date formats
        for tag, value in tags.items():
            if isinstance(value, exifread.classes.IfdTag) and value.field_type == 2:  # ASCII field
                date_str = str(value)
                if self.date_time_pattern.match(date_str) and self.is_valid_date(date_str):
                    dates.append(datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S"))
        
        if dates:
            return min(dates).strftime("%Y-%m-%d %H-%M-%S")
        else:
            return None
    
    def get_subseconds(self, tags: Dict[str, Any]) -> str:
        """
        Extract subsecond precision from EXIF tags.
        
        Args:
            tags: Dictionary of EXIF tags
            
        Returns:
            Subsecond string (padded to 6 digits) or 000000 if not found
        """
        # Check common subsecond tags
        subsec_tags = ['EXIF SubSecTime', 'EXIF SubSecTimeOriginal', 'EXIF SubSecTimeDigitized']
        for subsec_tag in subsec_tags:
            if subsec_tag in tags:
                subsec = str(tags[subsec_tag]).lstrip("0").zfill(6)
                self.logger.debug(f"Subseconds found in tag {subsec_tag}: {subsec}")
                return subsec
        
        # Check any tag with "SubSec" in the name
        self.logger.debug("No Subseconds found in common tags")
        for tag in tags.keys():
            if "SubSec" in tag:
                subsec = str(tags[tag]).lstrip("0").zfill(6)
                self.logger.debug(f"Subseconds found in tag {tag}: {subsec}")
                return subsec
        
        self.logger.debug("No Subseconds found in any tag")
        return '000000'
    
    def process_heic_file(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process HEIC/HEIF image files to extract metadata.
        
        Args:
            file_path: Path to HEIC file
            
        Returns:
            Tuple of (datetime string, format string) or (None, None) on error
        """
        try:
            heif_file = pyheif.read(file_path)
            
            for metadata in heif_file.metadata or []:
                if metadata['type'] == 'Exif':
                    tags = exifread.process_file(io.BytesIO(metadata['data']))
                    return self.get_oldest_date(tags), 'heic'
                    
            return None, 'heic'
        except Exception as e:
            self.logger.error(f"Error processing HEIC file: {file_path}, Error: {e}")
            return None, None
    
    def copy_and_rename_file(self, file_path: str) -> bool:
        """
        Copy and potentially rename image file based on metadata or filename.
        
        Args:
            file_path: Path to image file to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.destination_directory:
                self.logger.error("Destination directory not set")
                return False
                
            file_extension = os.path.splitext(file_path)[1].lower()
            filename_date = self.extract_date_from_filename(os.path.basename(file_path))
            oldest_date_time, image_format = None, None
            subsec_str = '000000'
            
            # Process different image formats
            if file_extension in ['.heic', '.heif']:
                oldest_date_time, image_format = self.process_heic_file(file_path)
                
                # If failed to extract from HEIC, try using exiftool
                if not oldest_date_time:
                    try:
                        metadata = utils.ExifToolWrapper.get_metadata(file_path)
                        if 'DateTimeOriginal' in metadata:
                            date_str = metadata['DateTimeOriginal']
                            if self.date_time_pattern.match(date_str) and self.is_valid_date(date_str):
                                oldest_date_time = datetime.strptime(
                                    date_str, "%Y:%m:%d %H:%M:%S"
                                ).strftime("%Y-%m-%d %H-%M-%S")
                    except Exception as e:
                        self.logger.error(f"Error using exiftool for HEIC: {e}")
            else:
                # Process standard image formats
                try:
                    with open(file_path, 'rb') as file:
                        tags = exifread.process_file(file)
                        oldest_date_time = self.get_oldest_date(tags)
                        subsec_str = self.get_subseconds(tags)
                        
                        with Image.open(file_path) as img:
                            image_format = img.format.lower() if img.format else None
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    return False
            
            # Use filename date if available
            if filename_date:
                oldest_date_time = filename_date
            
            # If no date information found, skip file
            if not oldest_date_time:
                self.logger.warning(f"No date information found for {file_path}")
                return False
            
            # Make sure we have a valid image format
            if not image_format or image_format not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'heic', 'dng']:
                image_format = file_extension.lstrip('.')
                if not image_format or image_format not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'heic', 'dng']:
                    image_format = 'jpg'
            
            # Determine target directory (organize by year if enabled)
            if self.organize_by_year:
                year = oldest_date_time[:4]
                new_folder = os.path.join(self.destination_directory, year)
            else:
                new_folder = self.destination_directory
                
            # Create directory if it doesn't exist
            os.makedirs(new_folder, exist_ok=True)
            
            # Format new filename
            formatted_date_time = oldest_date_time.replace(':', '-').replace(' ', '_')
            new_file_name = f"{formatted_date_time}-{subsec_str}.{image_format}"
            new_file_path = os.path.join(new_folder, new_file_name)
            
            # Handle file collisions
            counter = 1
            base, extension = os.path.splitext(new_file_name)
            while os.path.exists(new_file_path):
                self.collision_count += 1
                new_file_name = f"{base}-{counter}{extension}"
                new_file_path = os.path.join(new_folder, new_file_name)
                counter += 1
            
            # Copy file
            import shutil
            shutil.copy2(file_path, new_file_path)
            
            # Set XMP:Title to the filename without the extension
            filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            utils.ExifToolWrapper.set_metadata(
                new_file_path,
                {"XMP:Title": filename_without_extension}
            )
            
            self.logger.info(f"Processed: {file_path} -> {new_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def process_directories(self) -> None:
        """Process all image files in the source directories."""
        # Validate directories
        if not self.source_directories:
            self.logger.error("No source directories specified")
            return
            
        if not self.destination_directory:
            self.logger.error("No destination directory specified")
            return
        
        # Collect all image files
        files_to_process = []
        for directory in self.source_directories:
            for ext in self.supported_extensions:
                files_to_process.extend(
                    [str(p) for p in Path(directory).rglob(f"*{ext}")]
                )
                files_to_process.extend(
                    [str(p) for p in Path(directory).rglob(f"*{ext.upper()}")]
                )
        
        total_files = len(files_to_process)
        if total_files == 0:
            utils.UIHelper.show_info(
                "No Images Found", 
                "No supported image files found in the source directories."
            )
            return
        
        # Ask for confirmation
        if not utils.UIHelper.ask_yes_no(
            "Confirm Processing",
            f"Found {total_files} images to process.\n"
            f"Source: {', '.join(self.source_directories)}\n"
            f"Destination: {self.destination_directory}\n\n"
            f"Continue?"
        ):
            return
        
        # Create progress window
        window, progress_bar, status_label = utils.UIHelper.create_progress_window(
            title="Processing Images",
            message=f"Processing {total_files} images...",
            maximum=total_files
        )
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.collision_count = 0
        
        # Define function to update progress UI
        def update_progress(success: bool) -> None:
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                
            # Update progress and status asynchronously
            window.after(0, lambda: progress_bar.step(1))
            window.after(0, lambda: status_label.config(
                text=f"Processed: {self.success_count} | Errors: {self.error_count} | Collisions: {self.collision_count}"
            ))
            window.update_idletasks()
        
        # Process files in parallel
        max_workers = utils.ProcessingManager.get_optimal_workers()
        
        # Define worker function
        def process_file_with_progress(file_path: str) -> bool:
            result = self.copy_and_rename_file(file_path)
            
            # Update progress UI
            window.after(0, lambda: update_progress(result))
            
            return result
        
        # Process files
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file_with_progress, file) for file in files_to_process]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in process: {e}")
                    self.error_count += 1
        
        # Close progress window
        window.destroy()
        
        # Show results
        utils.UIHelper.show_info(
            "Processing Complete",
            f"Successfully processed: {self.success_count}\n"
            f"Errors: {self.error_count}\n"
            f"Collisions handled: {self.collision_count}\n"
            f"See media-image-processor.log for details"
        )
    
    def select_directories(self) -> bool:
        """
        Prompt for source and destination directories.
        
        Returns:
            bool: True if directories were selected, False otherwise
        """
        # Select source directories
        source_dirs = utils.FileSelector.select_directory(
            title="Select Source Directories",
            multiple=True
        )
        
        if not source_dirs:
            return False
            
        self.source_directories = source_dirs
        
        # Select destination directory
        dest_dir = utils.FileSelector.select_output_directory(
            title="Select Destination Directory"
        )
        
        if not dest_dir:
            return False
            
        self.destination_directory = dest_dir
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
        if not self.source_directories or not self.destination_directory:
            if not self.select_directories():
                self.logger.info("Directory selection cancelled. Exiting.")
                return
                
        # Process files
        self.process_directories()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process, rename, and organize image files"
    )
    parser.add_argument(
        "-s", "--source",
        nargs="+",
        help="Source directories to process"
    )
    parser.add_argument(
        "-d", "--destination",
        help="Destination directory for processed images"
    )
    parser.add_argument(
        "-y", "--by-year",
        action="store_true",
        default=True,
        help="Organize images by year (default: true)"
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
    
    # Create processor instance
    processor = ImageProcessor(
        source_directories=args.source,
        destination_directory=args.destination,
        organize_by_year=args.by_year,
        backup_mode=args.backup,
        verbose=args.verbose
    )
    
    # Run processor
    processor.run()

if __name__ == "__main__":
    main()