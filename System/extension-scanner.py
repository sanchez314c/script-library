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
# Script Name: systems-extension-scanner.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible parallel file extension analyzer that catalogs
#     all file types in directories with comprehensive statistics, visualization,
#     and export capabilities using multi-threaded processing.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded scanning using all available CPU cores
#     - Extension counting and intelligent categorization
#     - Native macOS GUI directory selection dialogs
#     - Recursive traversal with depth control and progress tracking
#     - Interactive visualization and CSV/JSON export
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - matplotlib, pandas (auto-installed)
#     - tkinter (standard library)
#
# Usage:
#     python systems-extension-scanner.py
#
####################################################################################

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import tkinter as tk
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Set, Tuple, Union, Any

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Define file type categories for classification
FILE_CATEGORIES = {
    'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.svg'},
    'video': {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'},
    'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff'},
    'document': {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.rtf', '.odt', '.ods', '.odp'},
    'archive': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.iso', '.dmg'},
    'code': {'.py', '.java', '.js', '.html', '.css', '.c', '.cpp', '.h', '.swift', '.go', '.rs', '.php', '.rb', '.sh'},
    'data': {'.json', '.xml', '.csv', '.sql', '.db', '.sqlite', '.yml', '.yaml', '.toml'}
}

class ExtensionStats:
    """Class to track file extension statistics."""
    
    def __init__(self):
        self.extension_count = Counter()
        self.extension_sizes = defaultdict(int)
        self.category_count = Counter()
        self.category_sizes = defaultdict(int)
        self.total_files = 0
        self.total_size = 0
        self.largest_files = []  # Will store (path, size, extension) tuples
        self.oldest_files = []   # Will store (path, timestamp, extension) tuples
        self.newest_files = []   # Will store (path, timestamp, extension) tuples
        self.processed_dirs = 0
        
    def update(self, other: 'ExtensionStats') -> None:
        """Merge another ExtensionStats object into this one."""
        self.extension_count.update(other.extension_count)
        for ext, size in other.extension_sizes.items():
            self.extension_sizes[ext] += size
        self.category_count.update(other.category_count)
        for cat, size in other.category_sizes.items():
            self.category_sizes[cat] += size
        self.total_files += other.total_files
        self.total_size += other.total_size
        self.largest_files.extend(other.largest_files)
        self.largest_files.sort(key=lambda x: x[1], reverse=True)
        self.largest_files = self.largest_files[:100]  # Keep top 100
        self.oldest_files.extend(other.oldest_files)
        self.oldest_files.sort(key=lambda x: x[1])
        self.oldest_files = self.oldest_files[:100]    # Keep top 100
        self.newest_files.extend(other.newest_files)
        self.newest_files.sort(key=lambda x: x[1], reverse=True)
        self.newest_files = self.newest_files[:100]    # Keep top 100
        self.processed_dirs += other.processed_dirs

class ProgressTracker:
    """Track and display progress for the scanning process."""
    
    def __init__(self, total_items: int = 0):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.lock = tk.BooleanVar(value=False)  # For thread safety
        
    def update(self, items: int = 1) -> None:
        """Update the processed items count."""
        locked = self.lock.get()
        while locked:
            time.sleep(0.01)
            locked = self.lock.get()
            
        self.lock.set(True)
        self.processed_items += items
        self.lock.set(False)
        
    def get_progress(self) -> Tuple[int, int, float]:
        """Return the current progress stats."""
        if self.total_items == 0:
            percent = 0
        else:
            percent = (self.processed_items / self.total_items) * 100
        return self.processed_items, self.total_items, percent
    
    def get_eta(self) -> str:
        """Estimate the time remaining based on progress so far."""
        if self.processed_items == 0:
            return "calculating..."
        
        elapsed = time.time() - self.start_time
        if self.total_items == 0:
            return "unknown"
        
        items_per_second = self.processed_items / elapsed
        if items_per_second == 0:
            return "unknown"
            
        remaining_items = self.total_items - self.processed_items
        seconds_remaining = remaining_items / items_per_second
        
        if seconds_remaining < 60:
            return f"{int(seconds_remaining)} seconds"
        elif seconds_remaining < 3600:
            minutes = seconds_remaining / 60
            return f"{int(minutes)} minutes"
        else:
            hours = seconds_remaining / 3600
            return f"{hours:.1f} hours"

def select_directory() -> Optional[str]:
    """
    Open a macOS native file dialog to select a directory.
    
    Returns:
        Selected directory path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Ensure dialog appears on top
    folder_selected = filedialog.askdirectory(title="Select Directory to Scan")
    root.destroy()
    return folder_selected

def get_file_category(extension: str) -> str:
    """Determine the category of a file based on its extension."""
    extension = extension.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    return 'other'

def scan_directory(dir_path: str, progress_tracker: ProgressTracker) -> ExtensionStats:
    """
    Scan a single directory for files and compile extension statistics.
    
    Args:
        dir_path: Path to the directory to scan
        progress_tracker: Progress tracking object
        
    Returns:
        ExtensionStats object with the results
    """
    stats = ExtensionStats()
    stats.processed_dirs = 1
    
    try:
        # Get all files in this directory (non-recursive)
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file():
                    file_path = entry.path
                    file_stats = entry.stat()
                    file_size = file_stats.st_size
                    file_path_obj = Path(file_path)
                    extension = file_path_obj.suffix.lower()
                    
                    # Update extension stats
                    stats.extension_count[extension] += 1
                    stats.extension_sizes[extension] += file_size
                    
                    # Update category stats
                    category = get_file_category(extension)
                    stats.category_count[category] += 1
                    stats.category_sizes[category] += file_size
                    
                    # Track total counts
                    stats.total_files += 1
                    stats.total_size += file_size
                    
                    # Track largest files
                    stats.largest_files.append((file_path, file_size, extension))
                    if len(stats.largest_files) > 100:
                        stats.largest_files.sort(key=lambda x: x[1], reverse=True)
                        stats.largest_files = stats.largest_files[:100]
                    
                    # Track oldest and newest files
                    mod_time = file_stats.st_mtime
                    stats.oldest_files.append((file_path, mod_time, extension))
                    stats.newest_files.append((file_path, mod_time, extension))
                    if len(stats.oldest_files) > 100:
                        stats.oldest_files.sort(key=lambda x: x[1])
                        stats.oldest_files = stats.oldest_files[:100]
                    if len(stats.newest_files) > 100:
                        stats.newest_files.sort(key=lambda x: x[1], reverse=True)
                        stats.newest_files = stats.newest_files[:100]
        
        progress_tracker.update()
        return stats
    
    except (PermissionError, OSError) as e:
        # Just skip this directory if we can't access it
        progress_tracker.update()
        return stats

def collect_directories(start_path: str, max_depth: int = -1) -> List[str]:
    """
    Collect all directories under the start path up to max_depth.
    
    Args:
        start_path: Starting directory path
        max_depth: Maximum recursion depth (-1 for unlimited)
        
    Returns:
        List of directory paths
    """
    dirs_to_scan = []
    
    # Check if start_path is valid
    if not os.path.isdir(start_path):
        return dirs_to_scan
    
    # Add the start path itself
    dirs_to_scan.append(start_path)
    
    # If max_depth is 0, only scan the start path
    if max_depth == 0:
        return dirs_to_scan
    
    # Recursive function to collect directories
    def collect_dirs(current_path: str, current_depth: int) -> None:
        try:
            with os.scandir(current_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        dir_path = entry.path
                        # Skip hidden directories (starting with .)
                        if os.path.basename(dir_path).startswith('.'):
                            continue
                        dirs_to_scan.append(dir_path)
                        
                        # Recurse if we haven't reached max depth
                        if max_depth == -1 or current_depth < max_depth:
                            collect_dirs(dir_path, current_depth + 1)
        except (PermissionError, OSError):
            # Skip directories we can't access
            pass
    
    # Start recursive collection
    collect_dirs(start_path, 1)
    return dirs_to_scan

def format_size(size_bytes: int) -> str:
    """Format file size in a human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_timestamp(timestamp: float) -> str:
    """Format a timestamp in a human-readable format."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def export_results(stats: ExtensionStats, format_type: str, output_path: Optional[str] = None) -> str:
    """
    Export the scan results to a file.
    
    Args:
        stats: The compiled statistics
        format_type: 'csv' or 'json'
        output_path: Optional path for the output file
        
    Returns:
        Path to the created file
    """
    if output_path is None:
        # Create default filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format_type.lower() == 'csv':
            output_path = f"extension_scan_{timestamp}.csv"
        else:
            output_path = f"extension_scan_{timestamp}.json"
    
    if format_type.lower() == 'csv':
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Extension', 'Count', 'Total Size (bytes)', 'Average Size', 'Category'])
            
            for ext, count in sorted(stats.extension_count.items(), key=lambda x: x[1], reverse=True):
                total_size = stats.extension_sizes[ext]
                avg_size = total_size / count if count > 0 else 0
                category = get_file_category(ext)
                writer.writerow([ext, count, total_size, avg_size, category])
    
    else:  # json
        result = {
            'scan_time': datetime.now().isoformat(),
            'total_files': stats.total_files,
            'total_size': stats.total_size,
            'extensions': {
                ext: {
                    'count': count,
                    'total_size': stats.extension_sizes[ext],
                    'category': get_file_category(ext)
                }
                for ext, count in stats.extension_count.items()
            },
            'categories': {
                cat: {
                    'count': count,
                    'total_size': stats.category_sizes[cat]
                }
                for cat, count in stats.category_count.items()
            },
            'largest_files': [
                {
                    'path': path,
                    'size': size,
                    'extension': ext
                }
                for path, size, ext in stats.largest_files[:20]  # Top 20 for JSON
            ],
            'oldest_files': [
                {
                    'path': path,
                    'timestamp': timestamp,
                    'formatted_time': format_timestamp(timestamp),
                    'extension': ext
                }
                for path, timestamp, ext in stats.oldest_files[:20]  # Top 20 for JSON
            ],
            'newest_files': [
                {
                    'path': path,
                    'timestamp': timestamp,
                    'formatted_time': format_timestamp(timestamp),
                    'extension': ext
                }
                for path, timestamp, ext in stats.newest_files[:20]  # Top 20 for JSON
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    return output_path

def visualize_results(stats: ExtensionStats) -> None:
    """Create visualizations of the scan results."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Install with: pip install matplotlib")
        return
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # 1. Top 10 extensions by count
    plt.subplot(2, 2, 1)
    top_extensions = dict(stats.extension_count.most_common(10))
    plt.bar(top_extensions.keys(), top_extensions.values())
    plt.title('Top 10 Extensions by Count')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 2. Top 10 extensions by size
    plt.subplot(2, 2, 2)
    ext_sizes = {ext: stats.extension_sizes[ext] / (1024 * 1024 * 1024) 
                for ext, _ in stats.extension_count.most_common(10)}
    plt.bar(ext_sizes.keys(), ext_sizes.values())
    plt.title('Top 10 Extensions by Size')
    plt.xticks(rotation=45)
    plt.ylabel('Size (GB)')
    
    # 3. Categories by count
    plt.subplot(2, 2, 3)
    plt.pie([count for _, count in stats.category_count.most_common()], 
            labels=[cat for cat, _ in stats.category_count.most_common()],
            autopct='%1.1f%%')
    plt.title('File Categories by Count')
    
    # 4. Categories by size
    plt.subplot(2, 2, 4)
    cat_sizes = {cat: stats.category_sizes[cat] for cat, _ in stats.category_count.most_common()}
    plt.pie([size for _, size in sorted(cat_sizes.items(), key=lambda x: x[1], reverse=True)],
            labels=[cat for cat, _ in sorted(cat_sizes.items(), key=lambda x: x[1], reverse=True)],
            autopct='%1.1f%%')
    plt.title('File Categories by Size')
    
    plt.tight_layout()
    plt.show()

def create_gui(scan_dir: Optional[str] = None) -> None:
    """
    Create and run the GUI for the extension scanner.
    
    Args:
        scan_dir: Optional directory to scan immediately
    """
    # Create the main window
    root = tk.Tk()
    root.title("Extension Scanner")
    root.geometry("800x600")
    
    # Variables
    scan_path_var = tk.StringVar(value=scan_dir if scan_dir else "")
    max_depth_var = tk.IntVar(value=-1)
    export_format_var = tk.StringVar(value="json")
    status_var = tk.StringVar(value="Ready")
    
    # Compiled stats
    stats = [None]  # Use a list to make it mutable within closures
    
    # Function to start the scan
    def start_scan():
        path = scan_path_var.get()
        if not path:
            path = select_directory()
            if not path:
                return
            scan_path_var.set(path)
        
        if not os.path.isdir(path):
            messagebox.showerror("Error", f"Invalid directory: {path}")
            return
        
        # Collect directories to scan
        max_depth = max_depth_var.get()
        status_var.set("Collecting directories...")
        
        # Disable controls during scan
        start_button.config(state=tk.DISABLED)
        path_entry.config(state=tk.DISABLED)
        depth_entry.config(state=tk.DISABLED)
        browse_button.config(state=tk.DISABLED)
        export_button.config(state=tk.DISABLED)
        visualize_button.config(state=tk.DISABLED)
        
        # Reset progress bar
        progress_bar['value'] = 0
        root.update()
        
        # Start directory collection in a separate thread
        def collect_and_scan():
            try:
                dirs_to_scan = collect_directories(path, max_depth)
                total_dirs = len(dirs_to_scan)
                
                # Create progress tracker
                progress = ProgressTracker(total_dirs)
                
                # Update status
                status_var.set(f"Scanning {total_dirs} directories...")
                
                # Create thread pool for parallel scanning
                num_workers = os.cpu_count() or 4  # Default to 4 if can't detect
                
                merged_stats = ExtensionStats()
                
                # Function to update progress bar
                def update_progress():
                    if progress.total_items == 0:
                        return
                    
                    processed, total, percent = progress.get_progress()
                    eta = progress.get_eta()
                    
                    progress_bar['value'] = percent
                    status_var.set(f"Processed {processed}/{total} directories ({percent:.1f}%) - ETA: {eta}")
                    
                    # Continue updating until all directories are processed
                    if processed < total:
                        root.after(100, update_progress)
                    else:
                        status_var.set(f"Scan complete! Found {merged_stats.total_files} files in {total} directories.")
                        result_text.delete(1.0, tk.END)
                        show_results(merged_stats)
                        
                        # Re-enable controls
                        start_button.config(state=tk.NORMAL)
                        path_entry.config(state=tk.NORMAL)
                        depth_entry.config(state=tk.NORMAL)
                        browse_button.config(state=tk.NORMAL)
                        export_button.config(state=tk.NORMAL)
                        visualize_button.config(state=tk.NORMAL)
                
                # Start progress updates
                root.after(100, update_progress)
                
                # Process directories in parallel
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all directories for processing
                    future_to_dir = {
                        executor.submit(scan_directory, directory, progress): directory
                        for directory in dirs_to_scan
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_dir):
                        dir_stats = future.result()
                        merged_stats.update(dir_stats)
                
                # Save final stats
                stats[0] = merged_stats
                
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during scanning: {str(e)}")
                # Re-enable controls
                start_button.config(state=tk.NORMAL)
                path_entry.config(state=tk.NORMAL)
                depth_entry.config(state=tk.NORMAL)
                browse_button.config(state=tk.NORMAL)
                export_button.config(state=tk.NORMAL)
                visualize_button.config(state=tk.NORMAL)
        
        # Start the scanning thread
        scan_thread = threading.Thread(target=collect_and_scan)
        scan_thread.daemon = True
        scan_thread.start()
    
    # Function to show results in the text area
    def show_results(stats: ExtensionStats) -> None:
        result_text.delete(1.0, tk.END)
        
        # Display summary
        result_text.insert(tk.END, "=== EXTENSION SCAN SUMMARY ===\n\n")
        result_text.insert(tk.END, f"Total files scanned: {stats.total_files:,}\n")
        result_text.insert(tk.END, f"Total size: {format_size(stats.total_size)}\n")
        result_text.insert(tk.END, f"Total directories: {stats.processed_dirs:,}\n")
        result_text.insert(tk.END, f"Unique extensions: {len(stats.extension_count):,}\n\n")
        
        # Display top extensions by count
        result_text.insert(tk.END, "=== TOP 10 EXTENSIONS BY COUNT ===\n\n")
        for ext, count in stats.extension_count.most_common(10):
            size = stats.extension_sizes[ext]
            avg_size = size / count if count > 0 else 0
            result_text.insert(tk.END, f"{ext or '(no extension)'}: {count:,} files, "
                              f"{format_size(size)} total, {format_size(avg_size)} avg\n")
        
        result_text.insert(tk.END, "\n")
        
        # Display top extensions by size
        result_text.insert(tk.END, "=== TOP 10 EXTENSIONS BY SIZE ===\n\n")
        by_size = sorted(stats.extension_sizes.items(), key=lambda x: x[1], reverse=True)
        for ext, size in by_size[:10]:
            count = stats.extension_count[ext]
            result_text.insert(tk.END, f"{ext or '(no extension)'}: {format_size(size)}, "
                              f"{count:,} files, {(size / stats.total_size) * 100:.1f}% of total\n")
        
        result_text.insert(tk.END, "\n")
        
        # Display categories
        result_text.insert(tk.END, "=== FILE CATEGORIES ===\n\n")
        for cat, count in sorted(stats.category_count.items(), key=lambda x: x[1], reverse=True):
            size = stats.category_sizes[cat]
            result_text.insert(tk.END, f"{cat}: {count:,} files, "
                              f"{format_size(size)}, {(size / stats.total_size) * 100:.1f}% of total\n")
        
        result_text.insert(tk.END, "\n")
        
        # Display largest files
        result_text.insert(tk.END, "=== 5 LARGEST FILES ===\n\n")
        for path, size, ext in stats.largest_files[:5]:
            result_text.insert(tk.END, f"{os.path.basename(path)}: {format_size(size)}, {ext} file\n")
            result_text.insert(tk.END, f"  Location: {path}\n")
        
        result_text.insert(tk.END, "\n")
        
        # Display oldest and newest files
        result_text.insert(tk.END, "=== 5 OLDEST FILES ===\n\n")
        for path, timestamp, ext in stats.oldest_files[:5]:
            result_text.insert(tk.END, f"{os.path.basename(path)}: {format_timestamp(timestamp)}, {ext} file\n")
            result_text.insert(tk.END, f"  Location: {path}\n")
        
        result_text.insert(tk.END, "\n")
        
        result_text.insert(tk.END, "=== 5 NEWEST FILES ===\n\n")
        for path, timestamp, ext in stats.newest_files[:5]:
            result_text.insert(tk.END, f"{os.path.basename(path)}: {format_timestamp(timestamp)}, {ext} file\n")
            result_text.insert(tk.END, f"  Location: {path}\n")
    
    # Function to handle export
    def export_results_handler():
        if not stats[0]:
            messagebox.showerror("Error", "No scan results to export")
            return
        
        format_type = export_format_var.get()
        default_ext = ".json" if format_type == "json" else ".csv"
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=default_ext,
            filetypes=[("JSON Files", "*.json"), ("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Save Scan Results"
        )
        
        if not output_path:
            return
        
        try:
            file_path = export_results(stats[0], format_type, output_path)
            messagebox.showinfo("Export Complete", f"Results exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    # Function to handle visualization
    def visualize_results_handler():
        if not stats[0]:
            messagebox.showerror("Error", "No scan results to visualize")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("Missing Dependency", 
                                 "Matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        # Create visualizations in a separate thread to keep UI responsive
        def viz_thread():
            try:
                visualize_results(stats[0])
            except Exception as e:
                messagebox.showerror("Visualization Error", f"Failed to create visualizations: {str(e)}")
        
        thread = threading.Thread(target=viz_thread)
        thread.daemon = True
        thread.start()
    
    # Create the UI elements
    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Directory selection
    ttk.Label(frame, text="Directory to scan:").grid(row=0, column=0, sticky=tk.W, pady=5)
    path_entry = ttk.Entry(frame, textvariable=scan_path_var, width=50)
    path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
    
    # Browse button
    def browse_dir():
        dir_path = select_directory()
        if dir_path:
            scan_path_var.set(dir_path)
    
    browse_button = ttk.Button(frame, text="Browse...", command=browse_dir)
    browse_button.grid(row=0, column=2, pady=5)
    
    # Max depth option
    ttk.Label(frame, text="Max recursion depth:").grid(row=1, column=0, sticky=tk.W, pady=5)
    depth_entry = ttk.Entry(frame, textvariable=max_depth_var, width=10)
    depth_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
    ttk.Label(frame, text="(-1 for unlimited)").grid(row=1, column=2, sticky=tk.W, pady=5)
    
    # Export format
    ttk.Label(frame, text="Export format:").grid(row=2, column=0, sticky=tk.W, pady=5)
    ttk.Radiobutton(frame, text="JSON", variable=export_format_var, value="json").grid(row=2, column=1, sticky=tk.W, pady=5)
    ttk.Radiobutton(frame, text="CSV", variable=export_format_var, value="csv").grid(row=2, column=1, sticky=tk.E, pady=5)
    
    # Control buttons
    button_frame = ttk.Frame(frame)
    button_frame.grid(row=3, column=0, columnspan=3, pady=10)
    
    start_button = ttk.Button(button_frame, text="Start Scan", command=start_scan)
    start_button.pack(side=tk.LEFT, padx=5)
    
    export_button = ttk.Button(button_frame, text="Export Results", command=export_results_handler)
    export_button.pack(side=tk.LEFT, padx=5)
    
    visualize_button = ttk.Button(button_frame, text="Visualize", command=visualize_results_handler)
    visualize_button.pack(side=tk.LEFT, padx=5)
    
    # Progress bar
    ttk.Label(frame, text="Progress:").grid(row=4, column=0, sticky=tk.W, pady=5)
    progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
    progress_bar.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    # Status bar
    status_bar = ttk.Label(frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
    
    # Results area
    result_frame = ttk.LabelFrame(frame, text="Scan Results")
    result_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5)
    
    result_text = tk.Text(result_frame, wrap=tk.WORD, width=80, height=20)
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Scrollbar for results
    scrollbar = ttk.Scrollbar(result_frame, command=result_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.config(yscrollcommand=scrollbar.set)
    
    # Configure grid expansion
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(6, weight=1)
    
    # If a scan directory was provided, start scan automatically
    if scan_dir:
        root.after(100, start_scan)
    
    # Start the UI
    root.mainloop()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parallel file extension scanner")
    parser.add_argument("--path", type=str, help="Directory to scan")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                      help=f"Number of threads to use (default: {os.cpu_count()})")
    parser.add_argument("--depth", type=int, default=-1,
                      help="Maximum recursion depth (-1 for unlimited)")
    parser.add_argument("--export", type=str, choices=["json", "csv"], 
                      help="Export results to specified format")
    parser.add_argument("--no-gui", action="store_true",
                      help="Run in command-line mode without GUI")
    parser.add_argument("--visualize", action="store_true",
                      help="Show visualizations of results (requires matplotlib)")
    return parser.parse_args()

def main() -> None:
    """Main function to run the extension scanner."""
    args = parse_arguments()
    
    # Check if we should use GUI or CLI mode
    if args.no_gui:
        # Command-line mode
        if not args.path:
            print("Error: In command-line mode, --path must be specified")
            sys.exit(1)
        
        print(f"Starting scan of {args.path} with {args.threads} threads...")
        
        # Collect directories to scan
        print("Collecting directories...")
        dirs_to_scan = collect_directories(args.path, args.depth)
        print(f"Found {len(dirs_to_scan)} directories to scan")
        
        # Create progress tracker
        progress = ProgressTracker(len(dirs_to_scan))
        
        # Create merged stats
        merged_stats = ExtensionStats()
        
        # Process directories in parallel
        print(f"Scanning with {args.threads} threads...")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            # Submit all directories for processing
            future_to_dir = {
                executor.submit(scan_directory, directory, progress): directory
                for directory in dirs_to_scan
            }
            
            # Process results as they complete
            completed = 0
            total = len(dirs_to_scan)
            
            for future in concurrent.futures.as_completed(future_to_dir):
                dir_stats = future.result()
                merged_stats.update(dir_stats)
                
                # Update progress
                completed += 1
                _, _, percent = progress.get_progress()
                eta = progress.get_eta()
                
                print(f"\rProgress: {completed}/{total} directories ({percent:.1f}%) - ETA: {eta}", end="")
        
        print("\nScan complete!")
        
        # Print summary
        print("\n=== EXTENSION SCAN SUMMARY ===")
        print(f"Total files scanned: {merged_stats.total_files:,}")
        print(f"Total size: {format_size(merged_stats.total_size)}")
        print(f"Total directories: {merged_stats.processed_dirs:,}")
        print(f"Unique extensions: {len(merged_stats.extension_count):,}")
        
        print("\n=== TOP 10 EXTENSIONS BY COUNT ===")
        for ext, count in merged_stats.extension_count.most_common(10):
            size = merged_stats.extension_sizes[ext]
            avg_size = size / count if count > 0 else 0
            print(f"{ext or '(no extension)'}: {count:,} files, "
                 f"{format_size(size)} total, {format_size(avg_size)} avg")
        
        print("\n=== TOP 10 EXTENSIONS BY SIZE ===")
        by_size = sorted(merged_stats.extension_sizes.items(), key=lambda x: x[1], reverse=True)
        for ext, size in by_size[:10]:
            count = merged_stats.extension_count[ext]
            print(f"{ext or '(no extension)'}: {format_size(size)}, "
                 f"{count:,} files, {(size / merged_stats.total_size) * 100:.1f}% of total")
        
        # Export results if requested
        if args.export:
            file_path = export_results(merged_stats, args.export)
            print(f"\nResults exported to: {file_path}")
        
        # Show visualizations if requested
        if args.visualize:
            if MATPLOTLIB_AVAILABLE:
                print("\nGenerating visualizations...")
                visualize_results(merged_stats)
            else:
                print("\nVisualization requires matplotlib. Install with: pip install matplotlib")
    
    else:
        # GUI mode
        create_gui(args.path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)
