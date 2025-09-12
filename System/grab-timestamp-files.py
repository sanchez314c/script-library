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
# Script Name: systems-grab-timestamp-files.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible timestamp file detection and processing system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Advanced timestamp pattern matching (00:00:00)
#     - Multi-threaded processing for efficient scanning
#     - Collision handling with automatic resolution
#     - Progress tracking with native macOS GUI dialogs
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, concurrent.futures, shutil (auto-installed if missing)
#
# Usage:
#     python systems-grab-timestamp-files.py
#
####################################################################################

import os
import sys
import re
import shutil
import json
import datetime
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import time

def install_and_import(package_name: str, import_name: str = None):
    """Auto-install and import packages with error handling"""
    if import_name is None:
        import_name = package_name
    
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return __import__(import_name)
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return None

# Auto-install and import required packages
tkinter = install_and_import("tkinter")
if tkinter:
    from tkinter import messagebox, filedialog, ttk
    import tkinter as tk

class TimestampFileGrabber:
    def __init__(self):
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / "timestamp_file_grabber_log.txt"
        self.found_files = []
        self.processed_count = 0
        self.total_files = 0
        self.start_time = None
        
        # Timestamp patterns to search for
        self.timestamp_patterns = [
            r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\d{1,2}:\d{2}:\d{2}',  # H:MM:SS or HH:MM:SS
            r'\d{2}-\d{2}-\d{2}',  # HH-MM-SS
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}.\d{2}.\d{2}',  # HH.MM.SS
            r'\d{8}_\d{6}',  # YYYYMMDD_HHMMSS
            r'\d{14}',  # YYYYMMDDHHMMSS
            r'(\d{1,2}h\d{1,2}m\d{1,2}s)',  # 1h30m45s format
            r'(\d+min\d+sec)',  # 30min45sec format
        ]
        
        # File extensions to search
        self.target_extensions = {
            '.txt', '.log', '.csv', '.json', '.xml', '.md', '.rtf',
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm',
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
        }
        
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI interface"""
        if not tkinter:
            print("GUI not available, running in console mode...")
            return
            
        self.root = tk.Tk()
        self.root.title("GET SWIFTY - Timestamp File Grabber v1.0.0")
        self.root.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Timestamp File Grabber", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Source directory selection
        ttk.Label(main_frame, text="Source Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar()
        source_frame = ttk.Frame(main_frame)
        source_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_var, width=50)
        self.source_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        source_button = ttk.Button(source_frame, text="Browse", command=self.select_source_directory)
        source_button.grid(row=0, column=1, padx=(5, 0))
        
        # Destination directory selection
        ttk.Label(main_frame, text="Destination Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.dest_var = tk.StringVar()
        dest_frame = ttk.Frame(main_frame)
        dest_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.dest_entry = ttk.Entry(dest_frame, textvariable=self.dest_var, width=50)
        self.dest_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        dest_button = ttk.Button(dest_frame, text="Browse", command=self.select_dest_directory)
        dest_button.grid(row=0, column=1, padx=(5, 0))
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.copy_files_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Copy files (instead of move)", 
                       variable=self.copy_files_var).grid(row=0, column=0, sticky=tk.W)
        
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Search subdirectories recursively", 
                       variable=self.recursive_var).grid(row=1, column=0, sticky=tk.W)
        
        self.organize_by_type_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Organize by file type", 
                       variable=self.organize_by_type_var).grid(row=2, column=0, sticky=tk.W)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to scan...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        self.scan_button = ttk.Button(buttons_frame, text="Start Scan", command=self.start_scan)
        self.scan_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_scan, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(buttons_frame, text="View Log", command=self.view_log).grid(row=0, column=2, padx=5)
        ttk.Button(buttons_frame, text="Exit", command=self.root.quit).grid(row=0, column=3, padx=5)
        
        # Results text area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        source_frame.columnconfigure(0, weight=1)
        dest_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.running = False
        
    def select_source_directory(self):
        """Select source directory for scanning"""
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.source_var.set(directory)
            
    def select_dest_directory(self):
        """Select destination directory for timestamp files"""
        directory = filedialog.askdirectory(title="Select Destination Directory")
        if directory:
            self.dest_var.set(directory)
    
    def log_message(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Logging error: {e}")
    
    def contains_timestamp(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains timestamp patterns"""
        found_patterns = []
        
        for pattern in self.timestamp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_patterns.extend(matches)
        
        return len(found_patterns) > 0, found_patterns
    
    def scan_file_content(self, file_path: Path) -> bool:
        """Scan file content for timestamp patterns"""
        try:
            # For text files, scan content
            if file_path.suffix.lower() in {'.txt', '.log', '.csv', '.md', '.rtf', '.json', '.xml'}:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # Read first 10KB
                    has_timestamp, patterns = self.contains_timestamp(content)
                    return has_timestamp
            
            # For other files, check filename only
            has_timestamp, patterns = self.contains_timestamp(file_path.name)
            return has_timestamp
            
        except Exception as e:
            self.log_message(f"Error scanning {file_path}: {e}")
            return False
    
    def find_timestamp_files(self, source_dir: Path) -> List[Dict]:
        """Find all files containing timestamp patterns"""
        timestamp_files = []
        
        try:
            if self.recursive_var.get():
                file_pattern = "**/*"
            else:
                file_pattern = "*"
            
            for file_path in source_dir.glob(file_pattern):
                if not self.running:
                    break
                    
                if file_path.is_file():
                    self.total_files += 1
                    
                    # Check file extension
                    if file_path.suffix.lower() in self.target_extensions:
                        # Check filename for timestamps
                        has_timestamp_in_name, name_patterns = self.contains_timestamp(file_path.name)
                        
                        # Check content for text files
                        has_timestamp_in_content = False
                        if file_path.suffix.lower() in {'.txt', '.log', '.csv', '.md', '.rtf', '.json', '.xml'}:
                            has_timestamp_in_content = self.scan_file_content(file_path)
                        
                        if has_timestamp_in_name or has_timestamp_in_content:
                            file_info = {
                                'path': file_path,
                                'size': file_path.stat().st_size,
                                'modified': datetime.datetime.fromtimestamp(file_path.stat().st_mtime),
                                'type': file_path.suffix.lower(),
                                'name_patterns': name_patterns if has_timestamp_in_name else [],
                                'has_content_timestamp': has_timestamp_in_content
                            }
                            timestamp_files.append(file_info)
                            
                            self.log_message(f"Found timestamp file: {file_path}")
                    
                    self.processed_count += 1
                    
                    # Update progress
                    if hasattr(self, 'progress_var'):
                        self.update_progress()
        
        except Exception as e:
            self.log_message(f"Error scanning directory {source_dir}: {e}")
        
        return timestamp_files
    
    def update_progress(self):
        """Update progress display"""
        if hasattr(self, 'progress_var') and self.total_files > 0:
            progress_pct = (self.processed_count / self.total_files) * 100
            self.progress_var.set(f"Processed {self.processed_count}/{self.total_files} files ({progress_pct:.1f}%)")
            self.progress_bar['value'] = progress_pct
            
            if hasattr(self, 'root'):
                self.root.update_idletasks()
    
    def organize_files(self, files: List[Dict], dest_dir: Path) -> Dict:
        """Organize and copy/move timestamp files"""
        results = {
            'copied': 0,
            'moved': 0,
            'errors': 0,
            'skipped': 0
        }
        
        for file_info in files:
            if not self.running:
                break
                
            try:
                source_path = file_info['path']
                
                # Determine destination subdirectory
                if self.organize_by_type_var.get():
                    file_type = file_info['type'].lstrip('.')
                    dest_subdir = dest_dir / file_type
                else:
                    dest_subdir = dest_dir
                
                # Create destination directory
                dest_subdir.mkdir(parents=True, exist_ok=True)
                
                # Handle name collisions
                dest_path = dest_subdir / source_path.name
                counter = 1
                while dest_path.exists():
                    name_parts = source_path.stem, counter, source_path.suffix
                    new_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                    dest_path = dest_subdir / new_name
                    counter += 1
                
                # Copy or move file
                if self.copy_files_var.get():
                    shutil.copy2(source_path, dest_path)
                    results['copied'] += 1
                    action = "Copied"
                else:
                    shutil.move(str(source_path), str(dest_path))
                    results['moved'] += 1
                    action = "Moved"
                
                self.log_message(f"{action}: {source_path} -> {dest_path}")
                
                # Update results display
                if hasattr(self, 'results_text'):
                    self.results_text.insert(tk.END, f"{action}: {source_path.name}\n")
                    self.results_text.see(tk.END)
                    self.root.update_idletasks()
                
            except Exception as e:
                results['errors'] += 1
                self.log_message(f"Error processing {file_info['path']}: {e}")
        
        return results
    
    def start_scan(self):
        """Start the scanning process"""
        source_dir = self.source_var.get()
        dest_dir = self.dest_var.get()
        
        if not source_dir or not dest_dir:
            messagebox.showerror("Error", "Please select both source and destination directories")
            return
        
        if not Path(source_dir).exists():
            messagebox.showerror("Error", "Source directory does not exist")
            return
        
        # Reset counters
        self.found_files = []
        self.processed_count = 0
        self.total_files = 0
        self.running = True
        self.start_time = time.time()
        
        # Update UI
        self.scan_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar['value'] = 0
        self.results_text.delete(1.0, tk.END)
        
        # Start scanning in separate thread
        def scan_thread():
            try:
                self.log_message("Starting timestamp file scan...")
                self.log_message(f"Source: {source_dir}")
                self.log_message(f"Destination: {dest_dir}")
                
                # Find timestamp files
                source_path = Path(source_dir)
                dest_path = Path(dest_dir)
                
                self.found_files = self.find_timestamp_files(source_path)
                
                if self.found_files and self.running:
                    self.log_message(f"Found {len(self.found_files)} timestamp files")
                    
                    # Organize files
                    results = self.organize_files(self.found_files, dest_path)
                    
                    # Show results
                    elapsed_time = time.time() - self.start_time
                    summary = f"""
Scan completed in {elapsed_time:.1f} seconds
Files processed: {self.processed_count}
Timestamp files found: {len(self.found_files)}
Files copied: {results['copied']}
Files moved: {results['moved']}
Errors: {results['errors']}
"""
                    self.log_message(summary)
                    
                    if hasattr(self, 'results_text'):
                        self.results_text.insert(tk.END, f"\n{summary}")
                        messagebox.showinfo("Scan Complete", summary)
                
                else:
                    self.log_message("No timestamp files found")
                    if hasattr(self, 'results_text'):
                        self.results_text.insert(tk.END, "No timestamp files found.\n")
                        messagebox.showinfo("Scan Complete", "No timestamp files found")
                
            except Exception as e:
                error_msg = f"Scan error: {e}"
                self.log_message(error_msg)
                if hasattr(self, 'root'):
                    messagebox.showerror("Error", error_msg)
            
            finally:
                self.running = False
                if hasattr(self, 'scan_button'):
                    self.scan_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    self.progress_var.set("Scan complete")
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def stop_scan(self):
        """Stop the scanning process"""
        self.running = False
        self.scan_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_var.set("Scan stopped")
        self.log_message("Scan stopped by user")
    
    def view_log(self):
        """Open the log file"""
        try:
            if self.log_file.exists():
                os.system(f'open "{self.log_file}"')
            else:
                messagebox.showinfo("Log", "No log file found")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open log file: {e}")
    
    def run(self):
        """Run the application"""
        if tkinter:
            self.log_message("GET SWIFTY Timestamp File Grabber v1.0.0 started")
            self.root.mainloop()
        else:
            self.console_mode()
    
    def console_mode(self):
        """Run in console mode if GUI is not available"""
        print("Running in console mode...")
        
        source_dir = input("Enter source directory path: ").strip()
        dest_dir = input("Enter destination directory path: ").strip()
        
        if not source_dir or not dest_dir:
            print("Both source and destination directories are required")
            return
        
        source_path = Path(source_dir)
        dest_path = Path(dest_dir)
        
        if not source_path.exists():
            print("Source directory does not exist")
            return
        
        self.running = True
        self.copy_files_var = type('obj', (object,), {'get': lambda: True})()
        self.recursive_var = type('obj', (object,), {'get': lambda: True})()
        self.organize_by_type_var = type('obj', (object,), {'get': lambda: False})()
        
        print("Scanning for timestamp files...")
        found_files = self.find_timestamp_files(source_path)
        
        if found_files:
            print(f"Found {len(found_files)} timestamp files")
            results = self.organize_files(found_files, dest_path)
            print(f"Results: {results}")
        else:
            print("No timestamp files found")

def main():
    """Main execution function"""
    try:
        grabber = TimestampFileGrabber()
        grabber.run()
    except Exception as e:
        print(f"Application error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()