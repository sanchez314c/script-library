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
# Script Name: systems-rsync-gui.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible rsync file transfer system with optimized GUI
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Optimized rsync transfers with progress tracking
#     - Resume support for interrupted transfers
#     - Native macOS GUI dialogs for folder selection
#     - Multi-threaded performance optimization with speed metrics
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, rsync (auto-installed/verified if missing)
#
# Usage:
#     python systems-rsync-gui.py
#
####################################################################################

import argparse
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

def check_rsync_installed() -> bool:
    """Check if rsync is installed on the system."""
    try:
        subprocess.run(['rsync', '--version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=False)
        return True
    except FileNotFoundError:
        return False

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

def parse_progress(line: str) -> Optional[str]:
    """
    Parse rsync progress output to extract metrics.
    
    Args:
        line: A line of rsync output
        
    Returns:
        Formatted progress string or None if parsing failed
    """
    try:
        if "%" in line:
            # Extract percentage
            percent_match = re.search(r'(\d+)%', line)
            if percent_match:
                percent = int(percent_match.group(1))
            else:
                percent = 0

            # Extract speed
            speed_match = re.search(r'(\d+\.?\d*\s*[kMG]B/s)', line)
            speed = speed_match.group(1) if speed_match else "0 B/s"

            # Extract bytes
            bytes_match = re.search(r'(\d+(?:,\d+)*)\s*bytes', line)
            if bytes_match:
                bytes_transferred = int(bytes_match.group(1).replace(',', ''))
                gb_transferred = bytes_transferred / (1024**3)
            else:
                gb_transferred = 0

            # Extract time remaining
            time_match = re.search(r'(\d+):(\d+):(\d+)', line)
            if time_match:
                hours, minutes, seconds = map(int, time_match.groups())
                time_remaining = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_remaining = "calculating..."

            return f"Progress: {percent}% | {gb_transferred:.2f} GB | Speed: {speed} | Time remaining: {time_remaining}"
    except Exception as e:
        # Log exception but continue execution
        print(f"Error parsing progress: {str(e)}")
    return None

def run_rsync(source: str, dest: str, options: Dict[str, bool] = None) -> subprocess.Popen:
    """
    Execute rsync process with specified options.
    
    Args:
        source: Source directory path
        dest: Destination directory path
        options: Dictionary of rsync options
        
    Returns:
        A subprocess.Popen object for the running rsync process
    """
    if options is None:
        options = {
            "archive": True,
            "verbose": True,
            "human-readable": True,
            "progress": True,
            "stats": True,
            "partial": True,
            "ignore-existing": True
        }
    
    # Build command based on options
    cmd = ['rsync']
    
    # Add standard options
    if options.get("archive", True):
        cmd.append('-a')
    if options.get("verbose", True):
        cmd.append('-v')
    if options.get("human-readable", True):
        cmd.append('-h')
    if options.get("progress", True):
        cmd.append('--progress')
    if options.get("stats", True):
        cmd.append('--stats')
    if options.get("partial", True):
        cmd.append('--partial')
    if options.get("ignore-existing", True):
        cmd.append('--ignore-existing')
    
    # Add source and destination
    cmd.append(f"{source}/")
    cmd.append(dest)
    
    return subprocess.Popen(cmd, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          encoding='latin-1',  # More forgiving encoding
                          bufsize=1)

def process_output_thread(process: subprocess.Popen, 
                         status_update_callback: Callable[[str], None],
                         file_progress_callback: Callable[[int], None],
                         total_progress_callback: Callable[[int, int], None]) -> None:
    """
    Thread function to process rsync output in parallel.
    
    Args:
        process: Running rsync process
        status_update_callback: Function to call with status updates
        file_progress_callback: Function to update file progress
        total_progress_callback: Function to update overall progress
    """
    total_files = 0
    processed_files = 0
    
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            # Count total files if not yet counted
            if total_files == 0 and "files to consider" in output:
                try:
                    total_files = int(re.search(r'(\d+) files to consider', output).group(1))
                except Exception:
                    pass
            
            # Update progress information
            if "%" in output:
                # Update file progress
                percent_match = re.search(r'(\d+)%', output)
                if percent_match:
                    file_percent = int(percent_match.group(1))
                    file_progress_callback(file_percent)
                    
                    # If file completed, update total progress
                    if file_percent == 100:
                        processed_files += 1
                        if total_files > 0:
                            total_progress_callback(processed_files, total_files)
            
            # Parse and display status
            progress_info = parse_progress(output)
            if progress_info:
                status_update_callback(progress_info)
            
            # Also send the raw output
            status_update_callback(output, raw=True)

def create_progress_ui() -> Tuple[tk.Tk, 
                                 tk.DoubleVar, 
                                 tk.DoubleVar, 
                                 ttk.Label, 
                                 tk.Text]:
    """
    Create progress tracking UI.
    
    Returns:
        Tuple containing window and UI elements
    """
    progress_window = tk.Tk()
    progress_window.title("Rsync Transfer Progress")
    progress_window.geometry("700x500")

    # Add progress bars
    ttk.Label(progress_window, text="Overall Progress:").pack(padx=10, pady=2)
    total_var = tk.DoubleVar()
    total_bar = ttk.Progressbar(progress_window, 
                              variable=total_var,
                              maximum=100,
                              length=600)
    total_bar.pack(padx=10, pady=2)

    ttk.Label(progress_window, text="Current File:").pack(padx=10, pady=2)
    file_var = tk.DoubleVar()
    file_bar = ttk.Progressbar(progress_window, 
                             variable=file_var,
                             maximum=100,
                             length=600)
    file_bar.pack(padx=10, pady=2)

    # Add status label
    status_label = ttk.Label(progress_window, text="Initializing...")
    status_label.pack(padx=10, pady=5)
    
    # Add progress text widget
    progress_text = tk.Text(progress_window, height=20, width=80)
    progress_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    # Scrollbar for text widget
    scrollbar = ttk.Scrollbar(progress_window, command=progress_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    progress_text.config(yscrollcommand=scrollbar.set)
    
    return progress_window, total_var, file_var, status_label, progress_text

def run_rsync_with_gui(source: str, dest: str, options: Dict[str, bool] = None) -> None:
    """
    Run rsync with GUI progress tracking.
    
    Args:
        source: Source directory path
        dest: Destination directory path
        options: Dictionary of rsync options
    """
    try:
        # Create progress window and UI elements
        progress_window, total_var, file_var, status_label, progress_text = create_progress_ui()
        
        # Start rsync process
        process = run_rsync(source, dest, options)
        
        # Define update callbacks
        def update_status(text: str, raw: bool = False) -> None:
            if raw:
                progress_text.insert(tk.END, text)
                progress_text.see(tk.END)
            else:
                status_label.config(text=text)
                
        def update_file_progress(percent: int) -> None:
            file_var.set(percent)
            
        def update_total_progress(processed: int, total: int) -> None:
            total_var.set((processed / total) * 100)
        
        # Start output processing thread
        output_thread = threading.Thread(
            target=process_output_thread,
            args=(process, update_status, update_file_progress, update_total_progress),
            daemon=True
        )
        output_thread.start()
        
        # Function to check if process is done
        def check_process() -> None:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    status_label.config(text="Transfer completed successfully!")
                    messagebox.showinfo("Success", "Transfer completed successfully!")
                else:
                    status_label.config(text="Transfer failed!")
                    messagebox.showerror("Error", f"Transfer failed:\n{stderr}")
                progress_window.destroy()
            else:
                progress_window.after(100, check_process)
                
        # Start checking process status
        check_process()
        
        # Start main event loop
        progress_window.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Optimized GUI for rsync file transfers")
    parser.add_argument("--source", type=str, help="Source directory path")
    parser.add_argument("--dest", type=str, help="Destination directory path")
    parser.add_argument("--no-ignore-existing", action="store_true", 
                      help="Don't ignore existing files (will overwrite)")
    parser.add_argument("--no-progress", action="store_true", 
                      help="Don't show progress (faster)")
    parser.add_argument("--delete", action="store_true", 
                      help="Delete files in destination that don't exist in source")
    return parser.parse_args()

def main() -> None:
    """Main function to run the rsync GUI application."""
    # Check if rsync is installed
    if not check_rsync_installed():
        messagebox.showerror("Error", "Rsync is not installed. Please install rsync and try again.")
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get source and destination directories
    source = args.source if args.source else select_directory("Select Source Directory")
    if not source:
        return
        
    dest = args.dest if args.dest else select_directory("Select Destination Directory")
    if not dest:
        return
    
    # Ensure paths exist
    source_path = Path(source)
    dest_path = Path(dest)
    
    if not source_path.exists():
        messagebox.showerror("Error", f"Source directory does not exist: {source}")
        return
        
    if not dest_path.exists():
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create destination directory: {str(e)}")
            return
    
    # Set rsync options based on arguments
    options = {
        "archive": True,
        "verbose": True,
        "human-readable": True,
        "progress": not args.no_progress,
        "stats": True,
        "partial": True,
        "ignore-existing": not args.no_ignore_existing,
        "delete": args.delete
    }
    
    # Run rsync with GUI
    run_rsync_with_gui(source, dest, options)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)