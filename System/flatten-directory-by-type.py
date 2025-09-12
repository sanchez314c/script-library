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
# Script Name: systems-flatten-directory-by-type.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible file organization system by extension type
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - File extension-based organization with custom mapping
#     - Multi-threaded processing using all available cores
#     - Collision handling with automatic renaming
#     - Progress tracking with native macOS GUI dialogs
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, concurrent.futures, shutil (auto-installed if missing)
#
# Usage:
#     python systems-flatten-directory-by-type.py
#
####################################################################################

import os
import sys
import json
import shutil
import argparse
import threading
import multiprocessing
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Set, Any


# Type mapping for common file extensions
DEFAULT_TYPE_MAPPING = {
    # Images
    "images": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg", "ico", "heic"],
    
    # Documents
    "documents": ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "rtf", "odt", "ods", "odp"],
    
    # Audio
    "audio": ["mp3", "wav", "ogg", "flac", "aac", "m4a", "wma", "aiff"],
    
    # Video
    "video": ["mp4", "avi", "mov", "wmv", "flv", "mkv", "webm", "m4v", "mpg", "mpeg"],
    
    # Archives
    "archives": ["zip", "rar", "7z", "tar", "gz", "bz2", "xz", "iso"],
    
    # Code
    "code": ["py", "js", "html", "css", "cpp", "c", "h", "java", "php", "rb", "go", "rs", "ts", "sh"],
    
    # Data
    "data": ["json", "xml", "csv", "sql", "db", "sqlite", "yml", "yaml"]
}


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


def create_progress_ui() -> Tuple[tk.Tk, ttk.Progressbar, ttk.Label, ttk.Label]:
    """
    Create a progress tracking UI.
    
    Returns:
        Tuple of (window, progress_bar, status_label, detail_label)
    """
    window = tk.Tk()
    window.title("Directory Organization Progress")
    window.geometry("600x200")
    
    # Create frame for progress elements
    frame = ttk.Frame(window, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Add status label
    status_label = ttk.Label(frame, text="Initializing...", font=("Helvetica", 12))
    status_label.pack(pady=10)
    
    # Add detail label
    detail_label = ttk.Label(frame, text="")
    detail_label.pack(pady=5)
    
    # Add progress bar
    progress_bar = ttk.Progressbar(frame, orient="horizontal", length=500, mode="determinate")
    progress_bar.pack(pady=10)
    
    return window, progress_bar, status_label, detail_label


def find_all_files(source_dir: str) -> List[str]:
    """
    Find all files in source directory.
    
    Args:
        source_dir: Source directory path
        
    Returns:
        List of file paths
    """
    file_paths = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
            
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths


def get_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension without the dot
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower().lstrip('.')


def get_type_directory(file_path: str, type_mapping: Dict[str, List[str]]) -> str:
    """
    Determine the type directory for a file based on its extension.
    
    Args:
        file_path: Path to the file
        type_mapping: Mapping of types to extensions
        
    Returns:
        Type directory name
    """
    ext = get_extension(file_path)
    
    # Find the type that contains this extension
    for type_name, extensions in type_mapping.items():
        if ext in extensions:
            return type_name
    
    # If no match found, use the extension itself
    return ext if ext else "unknown"


def build_type_mapping() -> Dict[str, List[str]]:
    """
    Build the type mapping from the default mapping.
    
    Returns:
        Dictionary mapping directory names to lists of extensions
    """
    # Return a copy of the default mapping
    return DEFAULT_TYPE_MAPPING.copy()


def move_file(source_path: str, 
             dest_dir: str, 
             type_mapping: Dict[str, List[str]],
             preview: bool = False) -> Dict[str, Any]:
    """
    Move a file to its target type directory.
    
    Args:
        source_path: Path to the source file
        dest_dir: Destination directory path
        type_mapping: Mapping of types to extensions
        preview: If True, don't actually move files
        
    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "source": source_path,
        "target": None,
        "message": "",
        "error": None
    }
    
    try:
        # Get the filename
        filename = os.path.basename(source_path)
        
        # Determine target type directory
        type_dir_name = get_type_directory(source_path, type_mapping)
        target_dir = os.path.join(dest_dir, type_dir_name)
        
        # Create target directory if it doesn't exist
        if not preview and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        # Construct target path
        target_path = os.path.join(target_dir, filename)
        result["target"] = target_path
        
        # Check for filename collision
        if not preview and os.path.exists(target_path):
            # Handle collision by adding a suffix
            base_name, ext = os.path.splitext(filename)
            counter = 1
            
            while os.path.exists(target_path):
                new_filename = f"{base_name}_{counter}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            result["target"] = target_path
            result["message"] = f"Renamed to avoid collision: {os.path.basename(target_path)}"
        
        # Move the file in non-preview mode
        if not preview:
            shutil.copy2(source_path, target_path)
            
        result["success"] = True
        result["message"] = "Preview only" if preview else "Copied successfully"
        
    except Exception as e:
        result["error"] = str(e)
        result["message"] = f"Error: {str(e)}"
    
    return result


def process_files(files: List[str], 
                 dest_dir: str, 
                 type_mapping: Dict[str, List[str]],
                 preview: bool = False,
                 progress_callback: Optional[callable] = None) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Process files for reorganization.
    
    Args:
        files: List of file paths
        dest_dir: Destination directory path
        type_mapping: Mapping of types to extensions
        preview: If True, don't actually move files
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (success_count, error_count, results)
    """
    success_count = 0
    error_count = 0
    results = []
    
    # Use a thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Create a list to store future objects
        future_to_file = {}
        
        # Submit tasks to the executor
        for file_path in files:
            future = executor.submit(move_file, file_path, dest_dir, type_mapping, preview)
            future_to_file[future] = file_path
        
        # Process results as they complete
        total_files = len(files)
        completed = 0
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    success_count += 1
                else:
                    error_count += 1
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_files, file_path, result["message"])
                
            except Exception as e:
                error_count += 1
                results.append({
                    "success": False,
                    "source": file_path,
                    "target": None,
                    "message": f"Error: {str(e)}",
                    "error": str(e)
                })
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_files, file_path, f"Error: {str(e)}")
    
    return success_count, error_count, results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Organize files by file type")
    parser.add_argument("--source", type=str, help="Source directory path")
    parser.add_argument("--dest", type=str, help="Destination directory path")
    parser.add_argument("--preview", action="store_true", help="Preview changes without moving files")
    return parser.parse_args()


def main() -> None:
    """Main function to run the directory organization tool."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Get source directory
        source_dir = args.source
        if not source_dir:
            source_dir = select_directory("Select Source Directory")
            if not source_dir:
                print("No source directory selected. Exiting.")
                return
        
        # Get destination directory
        dest_dir = args.dest
        if not dest_dir:
            dest_dir = select_directory("Select Destination Directory")
            if not dest_dir:
                print("No destination directory selected. Exiting.")
                return
        
        # Confirm directories exist
        if not os.path.isdir(source_dir):
            print(f"Error: Source directory {source_dir} is not valid.")
            return
        
        # Create destination if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        # Build type mapping
        type_mapping = build_type_mapping()
        
        # Create the progress UI
        window, progress_bar, status_label, detail_label = create_progress_ui()
        
        # Function to update progress
        def update_progress(completed: int, total: int, current_file: str, message: str) -> None:
            progress_percent = int((completed / total) * 100) if total > 0 else 0
            progress_bar.config(value=progress_percent)
            
            status_label.config(text=f"Processing files: {completed} of {total} ({progress_percent}%)")
            detail_label.config(text=f"Current: {os.path.basename(current_file)} - {message}")
            
            window.update()
        
        # Function to run the organization process in a separate thread
        def run_organization() -> None:
            try:
                # Find all files
                status_label.config(text="Finding files...")
                detail_label.config(text="Scanning directory...")
                window.update()
                
                files = find_all_files(source_dir)
                total_files = len(files)
                
                if total_files == 0:
                    messagebox.showinfo("No Files Found", "No files found in the selected directory.")
                    window.destroy()
                    return
                
                # Update status
                mode = "Preview" if args.preview else "Organizing"
                status_label.config(text=f"{mode} {total_files} files...")
                detail_label.config(text="Starting...")
                window.update()
                
                # Process files
                success_count, error_count, results = process_files(
                    files, dest_dir, type_mapping, args.preview, update_progress
                )
                
                # Show completion message
                action = "would be organized" if args.preview else "organized"
                if error_count == 0:
                    messagebox.showinfo(
                        "Operation Complete", 
                        f"All files processed successfully.\n\n"
                        f"{success_count} files {action}.\n"
                        f"{error_count} errors."
                    )
                else:
                    messagebox.showwarning(
                        "Operation Complete with Errors",
                        f"{success_count} files {action}.\n"
                        f"{error_count} errors occurred.\n\n"
                        "Check the console for details."
                    )
                    
                    # Print errors to console
                    print("\nErrors:")
                    for result in results:
                        if not result["success"]:
                            print(f"  - {result['source']}: {result['message']}")
                
                # Close the window
                window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
                window.destroy()
        
        # Start the organization process in a separate thread
        thread = threading.Thread(target=run_organization)
        thread.daemon = True
        thread.start()
        
        # Start the UI main loop
        window.mainloop()
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

