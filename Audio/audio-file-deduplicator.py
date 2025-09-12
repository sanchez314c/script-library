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
# Script Name: audio-file-deduplicator.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-performance audio file deduplicator using SHA-256 hashing to    
#              identify and safely manage duplicate audio files. Features multi-core      
#              processing, safe duplicate handling, and comprehensive reporting        
#              with progress tracking.                                               
#
# Usage: python audio-file-deduplicator.py [--source DIR] [--duplicates DIR] 
#
# Dependencies: tkinter (built-in)                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Uses SHA-256 file hashing for accurate duplicate detection. Safely moves       
#        duplicates to designated folder with collision handling. Optimized for   
#        multi-core processing with progress tracking GUI.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
Audio File Deduplicator

High-performance audio file deduplicator that identifies and safely manages
duplicate audio files using SHA-256 hashing with multi-core processing.
"""

import os
import sys
import hashlib
import logging
import shutil
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'audio-file-deduplicator.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('audio-file-deduplicator')

class AudioDeduplicator:
    """High-performance audio file deduplicator with multi-core processing"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.worker_count = max(1, self.cpu_count - 1)  # Leave 1 core free
        
        # Common audio file extensions
        self.audio_extensions = {
            '.mp3', '.m4a', '.aac', '.wav', '.flac', '.ogg', 
            '.wma', '.aiff', '.au', '.ra', '.3gp', '.amr', '.caff'
        }
        
        # Progress tracking
        self.progress_window = None
        self.progress_var = None
        self.status_var = None
        
        logger.info(f"Initialized deduplicator with {self.worker_count} workers")
    
    def select_directory(self, title: str) -> Optional[Path]:
        """Select directory using native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        directory = filedialog.askdirectory(title=title)
        root.destroy()
        
        return Path(directory) if directory else None
    
    def create_progress_window(self, title: str, max_value: int):
        """Create progress tracking window"""
        self.progress_window = tk.Toplevel()
        self.progress_window.title(title)
        self.progress_window.geometry("400x150")
        self.progress_window.resizable(False, False)
        self.progress_window.attributes('-topmost', True)
        
        # Center window
        self.progress_window.update_idletasks()
        width = self.progress_window.winfo_width()
        height = self.progress_window.winfo_height()
        x = (self.progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.progress_window.winfo_screenheight() // 2) - (height // 2)
        self.progress_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(self.progress_window, textvariable=self.status_var)
        status_label.pack(pady=20)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            self.progress_window,
            variable=self.progress_var,
            maximum=max_value,
            length=350
        )
        progress_bar.pack(pady=10)
        
        self.progress_window.update()
    
    def update_progress(self, current: int, status: str = None):
        """Update progress window"""
        if self.progress_window and self.progress_var:
            self.progress_var.set(current)
            if status and self.status_var:
                self.status_var.set(status)
            self.progress_window.update()
    
    def close_progress_window(self):
        """Close progress window"""
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
    
    def gather_audio_files(self, directory: Path) -> List[Path]:
        """Recursively gather all audio files"""
        audio_files = []
        
        try:
            for file_path in directory.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.audio_extensions):
                    audio_files.append(file_path)
            
            logger.info(f"Found {len(audio_files)} audio files to analyze")
            return audio_files
            
        except Exception as e:
            logger.error(f"Error gathering files: {e}")
            return []
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> Optional[Tuple[str, Path]]:
        """Calculate SHA-256 hash of a file"""
        try:
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                chunk_size = 8192
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest(), file_path
            
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None
    
    def find_duplicates(self, source_dir: Path) -> List[Tuple[Path, Path]]:
        """Find duplicate files using parallel processing"""
        # Gather all audio files
        all_files = self.gather_audio_files(source_dir)
        
        if not all_files:
            logger.warning("No audio files found")
            return []
        
        logger.info(f"Analyzing {len(all_files)} files for duplicates")
        
        # Create progress window
        self.create_progress_window("Finding Duplicates", len(all_files))
        
        # Hash tracking
        hash_map: Dict[str, Path] = {}
        duplicates: List[Tuple[Path, Path]] = []
        processed = 0
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all hashing tasks
            future_to_file = {
                executor.submit(self.calculate_file_hash, file_path): file_path 
                for file_path in all_files
            }
            
            # Process results as they complete
            for future in future_to_file:
                processed += 1
                
                # Update progress
                self.update_progress(
                    processed, 
                    f"Processed {processed}/{len(all_files)} files"
                )
                
                result = future.result()
                if result is None:
                    continue
                
                file_hash, file_path = result
                
                # Check for duplicate
                if file_hash in hash_map:
                    original_file = hash_map[file_hash]
                    duplicates.append((file_path, original_file))
                    logger.info(f"Found duplicate: {file_path.name} (original: {original_file.name})")
                else:
                    hash_map[file_hash] = file_path
        
        self.close_progress_window()
        
        logger.info(f"Found {len(duplicates)} duplicate files")
        return duplicates
    
    def move_duplicates(self, duplicates: List[Tuple[Path, Path]], 
                       destination: Path) -> Tuple[int, int, List[str]]:
        """Move duplicate files to destination directory"""
        destination.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        errors = []
        
        # Create progress window for moving files
        if duplicates:
            self.create_progress_window("Moving Duplicates", len(duplicates))
        
        for i, (duplicate_file, original_file) in enumerate(duplicates, 1):
            try:
                # Generate destination path
                dest_path = destination / duplicate_file.name
                
                # Handle filename collisions
                counter = 1
                while dest_path.exists():
                    stem = duplicate_file.stem
                    suffix = duplicate_file.suffix
                    dest_path = destination / f"{stem}_duplicate_{counter}{suffix}"
                    counter += 1
                
                # Move the file
                shutil.move(str(duplicate_file), str(dest_path))
                
                logger.info(f"Moved duplicate: {duplicate_file.name} → {dest_path.name}")
                successful += 1
                
                # Update progress
                self.update_progress(i, f"Moved {i}/{len(duplicates)} duplicates")
                
            except Exception as e:
                error_msg = f"Failed to move {duplicate_file.name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed += 1
        
        self.close_progress_window()
        
        return successful, failed, errors
    
    def run_interactive(self):
        """Run interactive mode with GUI dialogs"""
        logger.info("Starting interactive audio deduplication...")
        
        # Select source directory
        source_dir = self.select_directory("Select Source Directory (to scan for duplicates)")
        if not source_dir or not source_dir.exists():
            logger.info("No source directory selected")
            return
        
        # Select destination directory for duplicates
        dest_dir = self.select_directory("Select Destination Directory (for duplicate files)")
        if not dest_dir:
            logger.info("No destination directory selected")
            return
        
        try:
            # Find duplicates
            duplicates = self.find_duplicates(source_dir)
            
            if not duplicates:
                messagebox.showinfo(
                    "No Duplicates Found",
                    "No duplicate audio files were found in the selected directory."
                )
                logger.info("No duplicates found")
                return
            
            # Confirm before moving
            confirm_msg = (
                f"Found {len(duplicates)} duplicate files.\n\n"
                f"Move duplicates to: {dest_dir}?\n\n"
                f"Original files will remain in place."
            )
            
            if not messagebox.askyesno("Confirm Duplicate Removal", confirm_msg):
                logger.info("User cancelled duplicate removal")
                return
            
            # Move duplicates
            successful, failed, errors = self.move_duplicates(duplicates, dest_dir)
            
            # Show summary
            summary_msg = (
                f"Deduplication completed!\n\n"
                f"Total duplicates found: {len(duplicates)}\n"
                f"Successfully moved: {successful}\n"
                f"Failed to move: {failed}\n\n"
                f"Duplicates moved to: {dest_dir}"
            )
            
            if failed > 0:
                summary_msg += f"\n\nErrors occurred. Check log file for details."
                messagebox.showwarning("Deduplication Completed with Errors", summary_msg)
            else:
                messagebox.showinfo("Deduplication Completed", summary_msg)
            
            # Log final summary
            logger.info(f"""
Deduplication Summary:
- Source directory: {source_dir}
- Destination directory: {dest_dir}
- Total duplicates found: {len(duplicates)}
- Successfully moved: {successful}
- Failed to move: {failed}
""")
            
        except Exception as e:
            error_msg = f"Deduplication process failed: {e}"
            logger.error(error_msg)
            messagebox.showerror("Deduplication Error", error_msg)
        finally:
            self.close_progress_window()

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="High-performance audio file deduplicator"
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        help='Source directory to scan for duplicates'
    )
    
    parser.add_argument(
        '--duplicates', '-d',
        type=str,
        help='Destination directory for duplicate files'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of worker processes (default: CPU cores - 1)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Find duplicates but do not move them'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        deduplicator = AudioDeduplicator()
        
        # Override worker count if specified
        if args.workers:
            deduplicator.worker_count = max(1, min(args.workers, deduplicator.cpu_count))
            logger.info(f"Using {deduplicator.worker_count} workers (user specified)")
        
        # Command line mode
        if args.source:
            source_dir = Path(args.source)
            
            # Validate source path
            if not source_dir.exists() or not source_dir.is_dir():
                logger.error(f"Source directory does not exist: {source_dir}")
                return 1
            
            # Find duplicates
            duplicates = deduplicator.find_duplicates(source_dir)
            
            if not duplicates:
                print("No duplicate files found.")
                return 0
            
            print(f"\nFound {len(duplicates)} duplicate files:")
            for duplicate, original in duplicates:
                print(f"  Duplicate: {duplicate}")
                print(f"  Original:  {original}")
                print()
            
            # Handle dry run
            if args.dry_run:
                print("Dry run mode - no files were moved.")
                return 0
            
            # Move duplicates if destination specified
            if args.duplicates:
                dest_dir = Path(args.duplicates)
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                successful, failed, errors = deduplicator.move_duplicates(duplicates, dest_dir)
                
                print(f"\nDeduplication Summary:")
                print(f"Total duplicates: {len(duplicates)}")
                print(f"Successfully moved: {successful}")
                print(f"Failed to move: {failed}")
                
                if errors:
                    print(f"\nErrors:")
                    for error in errors[:5]:  # Show first 5 errors
                        print(f"  - {error}")
                    if len(errors) > 5:
                        print(f"  ... and {len(errors) - 5} more errors")
                
                return 0 if failed == 0 else 1
            else:
                print("Use --duplicates to specify where to move duplicate files.")
                return 0
        
        # Interactive mode
        else:
            deduplicator.run_interactive()
            return 0
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())