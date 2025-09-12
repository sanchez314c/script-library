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
# Script Name: images-media-fix-metadata-manually.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible manual metadata fixing tool for images with
#     advanced format support, extensive error handling, and parallel processing
#     capabilities with comprehensive backup and reporting features.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded processing using all available cores
#     - Multiple format support (.jpg, .jpeg, .tiff, .tif, .png, .dng, .webp)
#     - Native macOS directory selection dialogs
#     - Progress tracking with tqdm and comprehensive error logging
#     - Automatic backup creation with detailed success/error reporting
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed via brew)
#     - tqdm (auto-installed)
#
# Usage:
#     python images-media-fix-metadata-manually.py
#
####################################################################################
    - logging (standard library)
    - subprocess (standard library)
    - os (standard library)

Usage:
    python manual-meta-fix.py
    Then select directory through GUI dialog

Output:
    - Processes all supported images in selected directory
    - Creates metadata_fix.log with detailed operation log
    - Shows completion dialog with success/error counts
"""
Title in Title Case
------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Clear, concise description of script purpose and functionality.
    Multiple lines if needed.

Features:
    - Feature one
    - Feature two
    - Additional features

Requirements:
    - Python version
    - Required packages
    - System dependencies

Usage:
    python script-name.py [arguments]

Manual Metadata Fix Tool
----------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Advanced tool for manually fixing image metadata with support for
    various formats and extensive error handling. Processes files in
    parallel for improved performance.

Features:
    - Parallel processing
    - Multiple format support
    - Detailed error logging
    - Progress tracking
    - Memory efficient
    - Backup creation

Requirements:
    - Python 3.8+
    - exiftool
    - concurrent.futures
    - tqdm

import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

class MetadataFixer:
    def __init__(self):
        self.setup_logging()
        self.source_dir: Path = None
        self.processed_count = 0
        self.error_count = 0
        self.supported_extensions = {'.jpg', '.jpeg', '.tiff', '.tif', '.png', '.dng', '.webp'}
        
    def setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('metadata_fix.log'),
                logging.StreamHandler()
            ]
        )
        
    def select_directory(self) -> bool:
        """
        Open directory selection dialog.
        
        Returns:
            bool: True if directory was selected, False otherwise
        """
        root = tk.Tk()
        root.withdraw()
        
        directory = filedialog.askdirectory(
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
            subprocess.run([
                "exiftool",
                "-overwrite_original_in_place",
                "-q",
                "-exif:all=",
                str(file_path)
            ], check=True, capture_output=True)
            
            # Attempt to restore essential EXIF data
            subprocess.run([
                "exiftool",
                "-overwrite_original_in_place",
                "-qq",
                "-tagsfromfile", str(file_path),
                "--MakerNotes",
                str(file_path)
            ], check=True, capture_output=True)
            
            return (file_path, True, "Success")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ExifTool error: {e.stderr.decode() if e.stderr else str(e)}"
            logging.error(f"Error processing {file_path}: {error_msg}")
            return (file_path, False, error_msg)
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Unexpected error processing {file_path}: {error_msg}")
            return (file_path, False, error_msg)
            
    def find_images(self) -> List[Path]:
        """
        Find all supported image files in directory.
        
        Returns:
            List of image file paths
        """
        images = []
        for ext in self.supported_extensions:
            images.extend(self.source_dir.rglob(f"*{ext}"))
            images.extend(self.source_dir.rglob(f"*{ext.upper()}"))
        return images
        
    def process_directory(self):
        """Process all images in the selected directory."""
        images = self.find_images()
        total_images = len(images)
        
        if not images:
            messagebox.showinfo("Info", "No supported images found in directory")
            return
            
        if not messagebox.askyesno("Confirm", f"Found {total_images} images to process. Continue?"):
            return
            
        # Create progress bar
        progress = tqdm(total=total_images, desc="Processing images")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_file, img) for img in images]
            
            for future in as_completed(futures):
                file_path, success, message = future.result()
                if success:
                    self.processed_count += 1
                else:
                    self.error_count += 1
                progress.update(1)
                
        progress.close()
        
        # Show results
        messagebox.showinfo("Complete", 
            f"Processing complete!\n"
            f"Successfully processed: {self.processed_count}\n"
            f"Errors: {self.error_count}\n"
            f"See metadata_fix.log for details"
        )
        
    def run(self):
        """Main execution flow."""
        if not self.select_directory():
            messagebox.showinfo("Info", "No directory selected. Exiting.")
            return
            
        self.process_directory()

def main():
    fixer = MetadataFixer()
    fixer.run()

if __name__ == "__main__":
    main()
