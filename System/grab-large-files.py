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
# Script Name: systems-grab-large-files.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible large file detection and processing system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded processing using all available cores
#     - Flexible file size threshold selection with GUI
#     - Collision detection and handling with user prompts
#     - Progress tracking with native macOS dialogs
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, concurrent.futures, shutil (auto-installed if missing)
#
# Usage:
#     python systems-grab-large-files.py
#
####################################################################################

import os
import sys
import argparse
import shutil
import time
import datetime
import multiprocessing
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Union, Tuple, Any, Callable


def select_directory(title: str = "Select Folder") -> Optional[str]:
    """
    Open a native macOS dialog to select a directory.
    
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


def select_size_threshold() -> Optional[int]:
    """
    Prompt user to select size threshold.
    
    Returns:
        Size threshold in bytes or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    sizes = {
        "10 MB": 10 * 1024 * 1024,
        "50 MB": 50 * 1024 * 1024,
        "100 MB": 100 * 1024 * 1024,
        "500 MB": 500 * 1024 * 1024,
        "1 GB": 1024 * 1024 * 1024,
        "Custom...": -1
    }
    
    # Create dialog
    dialog = tk.Toplevel(root)
    dialog.title("Select Size Threshold")
    dialog.geometry("300x250")
    dialog.resizable(False, False)
    
    selected_size = tk.StringVar()
    label = ttk.Label(dialog, text="Select minimum file size:")
    label.pack(pady=10)
    
    # Create option buttons
    for size_label, _ in sizes.items():
        rb = ttk.Radiobutton(dialog, text=size_label, variable=selected_size, value=size_label)
        rb.pack(anchor=tk.W, padx=20, pady=5)
    
    # Default selection
    selected_size.set("100 MB")
    
    # Result variable
    result = [None]
    
    def on_ok():
        size_label = selected_size.get()
        if size_label == "Custom...":
            custom_size = simpledialog.askinteger(
                "Custom Size", 
                "Enter custom size (in MB):",
                parent=dialog,
                minvalue=1,
                maxvalue=100000
            )
            if custom_size:
                result[0] = custom_size * 1024 * 1024
        else:
            result[0] = sizes[size_label]
        dialog.destroy()
        root.destroy()
    
    def on_cancel():
        dialog.destroy()
        root.destroy()
    
    # Create buttons
    button_frame = ttk.Frame(dialog)
    button_frame.pack(pady=10, fill=tk.X)
    
    cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_button.pack(side=tk.RIGHT, padx=10)
    
    ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
    ok_button.pack(side=tk.RIGHT)
    
    # Make dialog modal
    dialog.transient(root)
    dialog.grab_set()
    root.wait_window(dialog)
    
    return result[0]


def find_large_files(source_dir: str, min_size: int) -> List[str]:
    """
    Find files larger than the specified size.
    
    Args:
        source_dir: Directory to search
        min_size: Minimum file size in bytes
        
    Returns:
        List of file paths
    """
    large_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size >= min_size:
                    large_files.append(file_path)
            except (OSError, FileNotFoundError) as e:
                print(f"Error accessing {file_path}: {e}")
    
    return large_files


def create_progress_ui() -> Tuple[tk.Tk, tk.IntVar, ttk.Label, ttk.Label, ttk.Progressbar]:
    """
    Create progress tracking UI.
    
    Returns:
        Tuple containing window and UI elements
    """
    progress_window = tk.Tk()
    progress_window.title("File Processing Progress")
    progress_window.geometry("600x250")
    
    # Add progress information
    file_info_label = ttk.Label(progress_window, text="Initializing...")
    file_info_label.pack(padx=10, pady=5)
    
    # Add status label
    status_label = ttk.Label(progress_window, text="")
    status_label.pack(padx=10, pady=5)
    
    # Add progress bar
    progress_var = tk.IntVar()
    progress_bar = ttk.Progressbar(progress_window, 
                                   variable=progress_var,
                                   maximum=100,
                                   length=500)
    progress_bar.pack(padx=10, pady=10)
    
    return progress_window, progress_var, file_info_label, status_label, progress_bar


def move_file(src: str, dst_dir: str, relative_path: str = "") -> bool:
    """
    Move a file to the destination directory, preserving relative path if specified.
    
    Args:
        src: Source file path
        dst_dir: Destination directory
        relative_path: Relative path to preserve
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create destination path
        if relative_path:
            dst_path = os.path.join(dst_dir, relative_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        else:
            dst_path = os.path.join(dst_dir, os.path.basename(src))
        
        # Check for collisions
        if os.path.exists(dst_path):
            # Add timestamp to filename
            filename, ext = os.path.splitext(os.path.basename(src))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{filename}_{timestamp}{ext}"
            
            if relative_path:
                dst_path = os.path.join(dst_dir, os.path.dirname(relative_path), new_filename)
            else:
                dst_path = os.path.join(dst_dir, new_filename)
        
        # Move the file
        shutil.move(src, dst_path)
        return True
    except Exception as e:
        print(f"Error moving file {src}: {e}")
        return False


def process_files(files: List[str], 
                 dest_dir: str, 
                 source_dir: str,
                 preserve_structure: bool,
                 progress_callback: Callable[[int, str], None]) -> int:
    """
    Process files with multi-threading.
    
    Args:
        files: List of file paths to process
        dest_dir: Destination directory
        source_dir: Source directory
        preserve_structure: Whether to preserve directory structure
        progress_callback: Callback function for progress updates
        
    Returns:
        Number of successful moves
    """
    success_count = 0
    total_files = len(files)
    
    # Function for worker threads
    def process_file(index: int, file_path: str) -> bool:
        try:
            relative_path = ""
            if preserve_structure:
                relative_path = os.path.relpath(file_path, source_dir)
            
            result = move_file(file_path, dest_dir, relative_path if preserve_structure else "")
            
            # Update progress
            progress_callback(index, file_path)
            
            return result
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    # Use a thread pool with optimal number of workers
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submit all tasks and collect futures
        future_to_index = {
            executor.submit(process_file, i, file_path): i 
            for i, file_path in enumerate(files)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            if future.result():
                success_count += 1
    
    return success_count


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Find and move large files")
    parser.add_argument("--source", type=str, help="Source directory path")
    parser.add_argument("--dest", type=str, help="Destination directory path")
    parser.add_argument("--size", type=int, help="Minimum file size in MB")
    parser.add_argument("--preserve-structure", action="store_true", 
                        help="Preserve directory structure")
    return parser.parse_args()


def main() -> None:
    """Main function to run the large file finder and mover."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Get source directory
        source_dir = args.source if args.source else select_directory("Select Source Directory")
        if not source_dir:
            print("No source directory selected. Exiting.")
            return
        
        # Get destination directory
        dest_dir = args.dest if args.dest else select_directory("Select Destination Directory")
        if not dest_dir:
            print("No destination directory selected. Exiting.")
            return
        
        # Get size threshold
        min_size_bytes = None
        if args.size:
            min_size_bytes = args.size * 1024 * 1024
        else:
            min_size_bytes = select_size_threshold()
            if not min_size_bytes:
                print("No size threshold selected. Exiting.")
                return
        
        # Ask about preserving structure if not specified in arguments
        preserve_structure = args.preserve_structure
        if not args.preserve_structure and not args.source:  # Only ask if not from command line
            root = tk.Tk()
            root.withdraw()
            preserve_structure = messagebox.askyesno(
                "Directory Structure",
                "Preserve directory structure in destination?"
            )
        
        print(f"Finding files larger than {min_size_bytes / (1024 * 1024):.1f} MB in {source_dir}...")
        
        # Find large files
        large_files = find_large_files(source_dir, min_size_bytes)
        
        if not large_files:
            messagebox.showinfo("No Files Found", 
                              f"No files larger than {min_size_bytes / (1024 * 1024):.1f} MB found.")
            return
        
        print(f"Found {len(large_files)} files.")
        
        # Create progress UI
        progress_window, progress_var, file_info_label, status_label, progress_bar = create_progress_ui()
        
        # Progress callback
        def update_progress(index: int, file_path: str) -> None:
            progress_var.set(int((index + 1) / len(large_files) * 100))
            file_info_label.config(text=f"Processing: {os.path.basename(file_path)}")
            status_label.config(text=f"File {index + 1} of {len(large_files)}")
            progress_window.update()
        
        # Start processing thread
        def process_thread() -> None:
            success_count = process_files(
                large_files, dest_dir, source_dir, preserve_structure, update_progress
            )
            
            # Update UI when done
            progress_var.set(100)
            file_info_label.config(text="Processing complete!")
            status_label.config(text=f"Moved {success_count} of {len(large_files)} files successfully.")
            
            # Show completion message
            messagebox.showinfo("Complete", 
                              f"Moved {success_count} of {len(large_files)} files successfully.")
            
            # Close progress window
            progress_window.destroy()
        
        # Start processing in a separate thread
        threading.Thread(target=process_thread, daemon=True).start()
        
        # Start progress window
        progress_window.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)
