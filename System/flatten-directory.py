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
# Script Name: systems-flatten-directory.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible directory structure flattening utility that
#     intelligently flattens complex nested directories into single level while
#     preserving all files and handling collisions with advanced strategies.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Intelligent directory flattening with smart collision resolution
#     - Multi-threaded processing using all available cores
#     - Native macOS folder selection dialogs with progress tracking
#     - Dry-run mode for safe previewing and multiple collision strategies
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - shutil, pathlib (standard library)
#     - tkinter (standard library)
#
# Usage:
#     python systems-flatten-directory.py
#
####################################################################################

import argparse
import concurrent.futures
import os
import re
import shutil
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable

class CollisionHandler:
    """Handles filename collisions during directory flattening."""
    
    STRATEGIES = {
        "rename": "Add numeric suffix to duplicates (file.txt → file_1.txt)",
        "parent_prefix": "Add parent directory name (parent/file.txt → parent_file.txt)",
        "hash_suffix": "Add partial hash to filename (file.txt → file_a7f3b.txt)",
        "timestamp": "Add timestamp to filename (file.txt → file_20250303_123045.txt)",
        "skip": "Skip duplicate files (first file wins)",
        "overwrite": "Overwrite existing files (last file wins)"
    }
    
    def __init__(self, strategy: str = "rename"):
        """
        Initialize the collision handler with the specified strategy.
        
        Args:
            strategy: The collision handling strategy to use
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Invalid collision strategy: {strategy}")
        
        self.strategy = strategy
        self.existing_files = set()
        self.moved_count = 0
        self.collision_count = 0
        self.skipped_count = 0
    
    def get_target_path(self, source_path: Path, dest_dir: Path) -> Optional[Path]:
        """
        Determine the target path for a file, handling any collisions.
        
        Args:
            source_path: Source file path
            dest_dir: Destination directory path
            
        Returns:
            Target file path or None if file should be skipped
        """
        filename = source_path.name
        target_path = dest_dir / filename
        
        # If no collision, return the target path as is
        if str(target_path) not in self.existing_files and not target_path.exists():
            self.existing_files.add(str(target_path))
            return target_path
        
        # Handle collision based on strategy
        self.collision_count += 1
        
        if self.strategy == "skip":
            self.skipped_count += 1
            return None
        
        elif self.strategy == "overwrite":
            self.existing_files.add(str(target_path))
            return target_path
        
        elif self.strategy == "rename":
            counter = 1
            stem = target_path.stem
            suffix = target_path.suffix
            while True:
                new_path = dest_dir / f"{stem}_{counter}{suffix}"
                if str(new_path) not in self.existing_files and not new_path.exists():
                    self.existing_files.add(str(new_path))
                    return new_path
                counter += 1
        
        elif self.strategy == "parent_prefix":
            parent_name = source_path.parent.name
            if parent_name:
                new_filename = f"{parent_name}_{filename}"
                new_path = dest_dir / new_filename
                if str(new_path) not in self.existing_files and not new_path.exists():
                    self.existing_files.add(str(new_path))
                    return new_path
                
                # If still a collision, fall back to rename strategy
                counter = 1
                stem = new_path.stem
                suffix = new_path.suffix
                while True:
                    new_path = dest_dir / f"{stem}_{counter}{suffix}"
                    if str(new_path) not in self.existing_files and not new_path.exists():
                        self.existing_files.add(str(new_path))
                        return new_path
                    counter += 1
            else:
                # If no parent name, fall back to rename strategy
                return self.get_target_path(source_path, dest_dir)
        
        elif self.strategy == "hash_suffix":
            import hashlib
            
            # Create a hash of the file content
            file_hash = hashlib.md5(str(source_path).encode()).hexdigest()[:6]
            stem = target_path.stem
            suffix = target_path.suffix
            new_path = dest_dir / f"{stem}_{file_hash}{suffix}"
            
            if str(new_path) not in self.existing_files and not new_path.exists():
                self.existing_files.add(str(new_path))
                return new_path
            
            # In the unlikely event of a hash collision, fall back to rename strategy
            counter = 1
            while True:
                new_path = dest_dir / f"{stem}_{file_hash}_{counter}{suffix}"
                if str(new_path) not in self.existing_files and not new_path.exists():
                    self.existing_files.add(str(new_path))
                    return new_path
                counter += 1
        
        elif self.strategy == "timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = target_path.stem
            suffix = target_path.suffix
            new_path = dest_dir / f"{stem}_{timestamp}{suffix}"
            
            if str(new_path) not in self.existing_files and not new_path.exists():
                self.existing_files.add(str(new_path))
                return new_path
            
            # In the unlikely event of a timestamp collision, add milliseconds
            import time
            ms = int(time.time() * 1000) % 1000
            new_path = dest_dir / f"{stem}_{timestamp}_{ms}{suffix}"
            self.existing_files.add(str(new_path))
            return new_path
        
        # Default fallback - should never get here
        return None

class ProgressTracker:
    """Track and display progress for the flattening process."""
    
    def __init__(self, total_items: int = 0):
        """
        Initialize the progress tracker.
        
        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def update(self, items: int = 1) -> None:
        """
        Update the processed items count.
        
        Args:
            items: Number of items processed
        """
        with self.lock:
            self.processed_items += items
        
    def get_progress(self) -> Tuple[int, int, float]:
        """
        Return the current progress stats.
        
        Returns:
            Tuple of (processed_items, total_items, percentage)
        """
        with self.lock:
            processed = self.processed_items
            total = self.total_items
        
        if total == 0:
            percent = 0
        else:
            percent = (processed / total) * 100
        
        return processed, total, percent
    
    def get_eta(self) -> str:
        """
        Estimate the time remaining based on progress.
        
        Returns:
            String with the estimated time remaining
        """
        with self.lock:
            processed = self.processed_items
            total = self.total_items
        
        if processed == 0:
            return "calculating..."
        
        elapsed = time.time() - self.start_time
        if total == 0:
            return "unknown"
        
        items_per_second = processed / elapsed
        if items_per_second == 0:
            return "unknown"
            
        remaining_items = total - processed
        seconds_remaining = remaining_items / items_per_second
        
        if seconds_remaining < 60:
            return f"{int(seconds_remaining)} seconds"
        elif seconds_remaining < 3600:
            minutes = seconds_remaining / 60
            return f"{int(minutes)} minutes"
        else:
            hours = seconds_remaining / 3600
            return f"{hours:.1f} hours"

def select_directory(title: str = "Select Folder") -> Optional[str]:
    """
    Open a macOS native file dialog to select a directory.
    
    Args:
        title: Dialog window title
        
    Returns:
        Selected directory path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Ensure dialog appears on top
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected

def collect_files(directory: Path) -> List[Path]:
    """
    Recursively collect all files in a directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of file paths
    """
    files = []
    
    try:
        # Walk the directory tree
        for root, _, filenames in os.walk(directory):
            root_path = Path(root)
            for filename in filenames:
                # Skip hidden files (starting with .)
                if filename.startswith('.'):
                    continue
                files.append(root_path / filename)
    except (PermissionError, OSError) as e:
        print(f"Error scanning directory {directory}: {str(e)}")
    
    return files

def move_file(source_path: Path, 
             dest_dir: Path, 
             collision_handler: CollisionHandler,
             dry_run: bool = False) -> Tuple[bool, str]:
    """
    Move a file to the destination directory, handling collisions.
    
    Args:
        source_path: Source file path
        dest_dir: Destination directory path
        collision_handler: Collision handling strategy
        dry_run: If True, don't actually move files
        
    Returns:
        Tuple of (success, message)
    """
    try:
        target_path = collision_handler.get_target_path(source_path, dest_dir)
        
        if target_path is None:
            return False, f"Skipped (collision): {source_path}"
        
        if dry_run:
            return True, f"Would move: {source_path} → {target_path}"
        
        # Ensure the target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        shutil.move(str(source_path), str(target_path))
        collision_handler.moved_count += 1
        
        return True, f"Moved: {source_path} → {target_path}"
        
    except (PermissionError, OSError) as e:
        return False, f"Error moving {source_path}: {str(e)}"

def flatten_directory(source_dir: Path, 
                     dest_dir: Path, 
                     collision_strategy: str = "rename",
                     dry_run: bool = False,
                     progress_tracker: Optional[ProgressTracker] = None,
                     status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, int]:
    """
    Flatten a directory structure.
    
    Args:
        source_dir: Source directory
        dest_dir: Destination directory
        collision_strategy: Strategy for handling filename collisions
        dry_run: If True, don't actually move files
        progress_tracker: Progress tracking object
        status_callback: Callback function for status updates
        
    Returns:
        Statistics dictionary
    """
    # Create destination directory if it doesn't exist
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collision handler
    collision_handler = CollisionHandler(strategy=collision_strategy)
    
    # Get list of all files
    if status_callback:
        status_callback("Collecting files...")
    
    files = collect_files(source_dir)
    
    if progress_tracker:
        progress_tracker.total_items = len(files)
    
    if status_callback:
        status_callback(f"Found {len(files)} files to process")
    
    # Process each file
    error_count = 0
    
    for file_path in files:
        success, message = move_file(file_path, dest_dir, collision_handler, dry_run)
        
        if not success:
            error_count += 1
        
        if status_callback:
            status_callback(message)
        
        if progress_tracker:
            progress_tracker.update()
    
    # Compile statistics
    stats = {
        "total_files": len(files),
        "moved_files": collision_handler.moved_count,
        "collision_count": collision_handler.collision_count,
        "skipped_files": collision_handler.skipped_count,
        "error_count": error_count
    }
    
    return stats

def parallel_flatten_directory(source_dir: Path, 
                              dest_dir: Path, 
                              collision_strategy: str = "rename",
                              dry_run: bool = False,
                              num_workers: int = 4,
                              progress_tracker: Optional[ProgressTracker] = None,
                              status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, int]:
    """
    Flatten a directory structure using multiple threads.
    
    Args:
        source_dir: Source directory
        dest_dir: Destination directory
        collision_strategy: Strategy for handling filename collisions
        dry_run: If True, don't actually move files
        num_workers: Number of worker threads
        progress_tracker: Progress tracking object
        status_callback: Callback function for status updates
        
    Returns:
        Statistics dictionary
    """
    # Create destination directory if it doesn't exist
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collision handler
    collision_handler = CollisionHandler(strategy=collision_strategy)
    
    # Get list of all files
    if status_callback:
        status_callback("Collecting files...")
    
    files = collect_files(source_dir)
    
    if progress_tracker:
        progress_tracker.total_items = len(files)
    
    if status_callback:
        status_callback(f"Found {len(files)} files to process")
    
    # Define worker function
    def worker(file_path):
        success, message = move_file(file_path, dest_dir, collision_handler, dry_run)
        
        if status_callback:
            status_callback(message)
        
        if progress_tracker:
            progress_tracker.update()
        
        return success
    
    # Process files in parallel
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(worker, file_path): file_path
            for file_path in files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            success = future.result()
            if not success:
                error_count += 1
    
    # Compile statistics
    stats = {
        "total_files": len(files),
        "moved_files": collision_handler.moved_count,
        "collision_count": collision_handler.collision_count,
        "skipped_files": collision_handler.skipped_count,
        "error_count": error_count
    }
    
    return stats

def create_gui() -> None:
    """Create and run the GUI for the directory flattener."""
    # Create the main window
    root = tk.Tk()
    root.title("Directory Flattener")
    root.geometry("800x600")
    
    # Variables
    source_var = tk.StringVar()
    dest_var = tk.StringVar()
    strategy_var = tk.StringVar(value="rename")
    dry_run_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="Ready")
    
    # Function to browse for source directory
    def browse_source():
        directory = select_directory("Select Source Directory")
        if directory:
            source_var.set(directory)
    
    # Function to browse for destination directory
    def browse_dest():
        directory = select_directory("Select Destination Directory")
        if directory:
            dest_var.set(directory)
    
    # Function to update status
    def update_status(message):
        # This will be called from a worker thread, so we need to use after()
        # to update the UI safely
        root.after(0, lambda: status_var.set(message))
        root.after(0, lambda: log_text.insert(tk.END, message + "\n"))
        root.after(0, lambda: log_text.see(tk.END))
    
    # Function to start flattening
    def start_flatten():
        source = source_var.get()
        dest = dest_var.get()
        
        if not source:
            messagebox.showerror("Error", "Please select a source directory")
            return
        
        if not dest:
            messagebox.showerror("Error", "Please select a destination directory")
            return
        
        source_path = Path(source)
        dest_path = Path(dest)
        
        if not source_path.exists() or not source_path.is_dir():
            messagebox.showerror("Error", f"Source directory does not exist: {source}")
            return
        
        # Disable controls during processing
        for widget in controls:
            widget.config(state=tk.DISABLED)
        
        # Clear log
        log_text.delete(1.0, tk.END)
        
        # Reset progress bar
        progress_bar['value'] = 0
        root.update()
        
        # Get options
        strategy = strategy_var.get()
        dry_run = dry_run_var.get()
        
        # Create progress tracker
        progress = ProgressTracker()
        
        # Define progress update function
        def update_progress():
            if progress.total_items == 0:
                return
            
            processed, total, percent = progress.get_progress()
            eta = progress.get_eta()
            
            progress_bar['value'] = percent
            status_var.set(f"Processed {processed}/{total} files ({percent:.1f}%) - ETA: {eta}")
            
            # Continue updating until all files are processed
            if processed < total:
                root.after(100, update_progress)
            else:
                # Re-enable controls when done
                for widget in controls:
                    widget.config(state=tk.NORMAL)
        
        # Define worker function
        def worker():
            try:
                # Determine number of workers based on CPU count
                num_workers = os.cpu_count() or 4
                
                # Start the flattening process
                stats = parallel_flatten_directory(
                    source_path,
                    dest_path,
                    collision_strategy=strategy,
                    dry_run=dry_run,
                    num_workers=num_workers,
                    progress_tracker=progress,
                    status_callback=update_status
                )
                
                # Show summary when done
                summary = f"\n=== SUMMARY ===\n"
                summary += f"Total files: {stats['total_files']}\n"
                summary += f"Files moved: {stats['moved_files']}\n"
                summary += f"Collisions: {stats['collision_count']}\n"
                summary += f"Files skipped: {stats['skipped_files']}\n"
                summary += f"Errors: {stats['error_count']}\n"
                
                update_status(summary)
                
                if dry_run:
                    messagebox.showinfo("Dry Run Complete", 
                                      "Dry run completed successfully. No files were actually moved.")
                else:
                    messagebox.showinfo("Flattening Complete", 
                                      f"Directory flattening completed. {stats['moved_files']} files moved.")
                
            except Exception as e:
                update_status(f"Error: {str(e)}")
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                
                # Re-enable controls on error
                for widget in controls:
                    widget.config(state=tk.NORMAL)
        
        # Start progress updates
        root.after(100, update_progress)
        
        # Start worker thread
        threading.Thread(target=worker, daemon=True).start()
    
    # Create the UI elements
    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Source directory selection
    ttk.Label(frame, text="Source Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
    source_entry = ttk.Entry(frame, textvariable=source_var, width=50)
    source_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
    source_button = ttk.Button(frame, text="Browse...", command=browse_source)
    source_button.grid(row=0, column=2, padx=5, pady=5)
    
    # Destination directory selection
    ttk.Label(frame, text="Destination Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
    dest_entry = ttk.Entry(frame, textvariable=dest_var, width=50)
    dest_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
    dest_button = ttk.Button(frame, text="Browse...", command=browse_dest)
    dest_button.grid(row=1, column=2, padx=5, pady=5)
    
    # Collision strategy selection
    ttk.Label(frame, text="Collision Strategy:").grid(row=2, column=0, sticky=tk.W, pady=5)
    strategy_combo = ttk.Combobox(frame, textvariable=strategy_var, state="readonly")
    strategy_combo['values'] = list(CollisionHandler.STRATEGIES.keys())
    strategy_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
    
    # Strategy description label
    strategy_desc_var = tk.StringVar()
    
    def update_strategy_desc(*args):
        strategy = strategy_var.get()
        if strategy in CollisionHandler.STRATEGIES:
            strategy_desc_var.set(CollisionHandler.STRATEGIES[strategy])
    
    strategy_var.trace("w", update_strategy_desc)
    update_strategy_desc()  # Set initial description
    
    strategy_desc = ttk.Label(frame, textvariable=strategy_desc_var, font=("", 9, "italic"))
    strategy_desc.grid(row=3, column=1, sticky=tk.W, pady=0)
    
    # Dry run option
    dry_run_check = ttk.Checkbutton(frame, text="Dry Run (preview only, don't move files)", 
                                  variable=dry_run_var)
    dry_run_check.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)
    
    # Action buttons
    button_frame = ttk.Frame(frame)
    button_frame.grid(row=5, column=0, columnspan=3, pady=10)
    
    start_button = ttk.Button(button_frame, text="Start Flattening", command=start_flatten)
    start_button.pack(side=tk.LEFT, padx=5)
    
    # Progress bar
    ttk.Label(frame, text="Progress:").grid(row=6, column=0, sticky=tk.W, pady=5)
    progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=500, mode='determinate')
    progress_bar.grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    # Status bar
    status_bar = ttk.Label(frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
    
    # Log text area
    log_frame = ttk.LabelFrame(frame, text="Log")
    log_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5)
    
    log_text = tk.Text(log_frame, wrap=tk.WORD, width=80, height=20)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    scrollbar = ttk.Scrollbar(log_frame, command=log_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.config(yscrollcommand=scrollbar.set)
    
    # Keep track of controls to enable/disable
    controls = [source_entry, source_button, dest_entry, dest_button, 
               strategy_combo, dry_run_check, start_button]
    
    # Configure grid expansion
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(8, weight=1)
    
    # Start the UI
    root.mainloop()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Flatten directory structure")
    parser.add_argument("--source", type=str, help="Source directory")
    parser.add_argument("--dest", type=str, help="Destination directory")
    parser.add_argument("--strategy", type=str, default="rename",
                      choices=list(CollisionHandler.STRATEGIES.keys()),
                      help="Collision handling strategy")
    parser.add_argument("--dry-run", action="store_true",
                      help="Perform a dry run without moving any files")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                      help=f"Number of worker threads (default: {os.cpu_count()})")
    parser.add_argument("--no-gui", action="store_true",
                      help="Run in command-line mode without GUI")
    return parser.parse_args()

def main() -> None:
    """Main function to run the flattening process."""
    args = parse_arguments()
    
    # Check if we should use GUI or CLI mode
    if args.no_gui:
        # Command-line mode
        if not args.source:
            print("Error: Source directory is required in command-line mode")
            sys.exit(1)
        
        if not args.dest:
            print("Error: Destination directory is required in command-line mode")
            sys.exit(1)
        
        source_path = Path(args.source)
        dest_path = Path(args.dest)
        
        if not source_path.exists() or not source_path.is_dir():
            print(f"Error: Source directory does not exist: {args.source}")
            sys.exit(1)
        
        # Create progress tracker
        progress = ProgressTracker()
        
        # Status callback function
        def print_status(message):
            print(message)
        
        # Start the flattening process
        print(f"Flattening directory {args.source} to {args.dest}")
        print(f"Collision strategy: {args.strategy}")
        if args.dry_run:
            print("Dry run mode: no files will be moved")
        
        # Define progress update function
        def update_progress():
            if progress.total_items == 0:
                return
            
            processed, total, percent = progress.get_progress()
            eta = progress.get_eta()
            
            print(f"\rProgress: {processed}/{total} files ({percent:.1f}%) - ETA: {eta}", end="")
            
            # Continue updating until all files are processed
            if processed < total:
                threading.Timer(1.0, update_progress).start()
        
        # Start progress updates
        update_progress()
        
        try:
            stats = parallel_flatten_directory(
                source_path,
                dest_path,
                collision_strategy=args.strategy,
                dry_run=args.dry_run,
                num_workers=args.threads,
                progress_tracker=progress,
                status_callback=print_status
            )
            
            # Print summary
            print("\n\n=== SUMMARY ===")
            print(f"Total files: {stats['total_files']}")
            print(f"Files moved: {stats['moved_files']}")
            print(f"Collisions: {stats['collision_count']}")
            print(f"Files skipped: {stats['skipped_files']}")
            print(f"Errors: {stats['error_count']}")
            
            if args.dry_run:
                print("\nDry run completed successfully. No files were actually moved.")
            else:
                print(f"\nDirectory flattening completed. {stats['moved_files']} files moved.")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            sys.exit(1)
    
    else:
        # GUI mode
        create_gui()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

