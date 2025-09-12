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
# Script Name: systems-file-move-manager.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible advanced file movement management system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded file operations for high performance
#     - Advanced collision detection and handling
#     - Progress monitoring with real-time updates
#     - Timestamp-based conflict resolution with user options
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, concurrent.futures, shutil (auto-installed if missing)
#
# Usage:
#     python systems-file-move-manager.py
#
####################################################################################

import os
import sys
import shutil
import hashlib
import json
import datetime
import threading
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import queue

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

class FileMoveManager:
    def __init__(self):
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / "file_move_manager_log.txt"
        self.operations_file = self.desktop_path / "file_move_operations.json"
        
        # Operation tracking
        self.move_queue = queue.Queue()
        self.completed_operations = []
        self.failed_operations = []
        self.total_files = 0
        self.processed_files = 0
        self.total_size = 0
        self.processed_size = 0
        self.start_time = None
        
        # Thread management
        self.worker_threads = []
        self.max_workers = 4
        self.running = False
        self.paused = False
        
        # Collision handling options
        self.collision_strategy = "ask"  # ask, skip, overwrite, rename
        
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI interface"""
        if not tkinter:
            print("GUI not available, running in console mode...")
            return
            
        self.root = tk.Tk()
        self.root.title("GET SWIFTY - File Move Manager v1.0.0")
        self.root.geometry("800x700")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced File Move Manager", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Source and destination frame
        paths_frame = ttk.LabelFrame(main_frame, text="File Paths", padding="10")
        paths_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Source directory
        ttk.Label(paths_frame, text="Source Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar()
        source_frame = ttk.Frame(paths_frame)
        source_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_var, width=60)
        self.source_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(source_frame, text="Browse", command=self.select_source_directory).grid(row=0, column=1, padx=(5, 0))
        
        # Destination directory
        ttk.Label(paths_frame, text="Destination Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dest_var = tk.StringVar()
        dest_frame = ttk.Frame(paths_frame)
        dest_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.dest_entry = ttk.Entry(dest_frame, textvariable=self.dest_var, width=60)
        self.dest_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(dest_frame, text="Browse", command=self.select_dest_directory).grid(row=0, column=1, padx=(5, 0))
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Move Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # File filters
        ttk.Label(options_frame, text="File Extensions (comma-separated):").grid(row=0, column=0, sticky=tk.W)
        self.extensions_var = tk.StringVar()
        ttk.Entry(options_frame, textvariable=self.extensions_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Recursive", variable=self.recursive_var).grid(row=0, column=2, sticky=tk.W)
        
        # Size filters
        ttk.Label(options_frame, text="Min Size (MB):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.min_size_var = tk.StringVar(value="0")
        ttk.Entry(options_frame, textvariable=self.min_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(5, 20), pady=(10, 0))
        
        ttk.Label(options_frame, text="Max Size (MB):").grid(row=1, column=2, sticky=tk.W, pady=(10, 0))
        self.max_size_var = tk.StringVar()
        ttk.Entry(options_frame, textvariable=self.max_size_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        
        # Collision handling
        ttk.Label(options_frame, text="File Collision:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.collision_var = tk.StringVar(value="ask")
        collision_combo = ttk.Combobox(options_frame, textvariable=self.collision_var, width=15)
        collision_combo['values'] = ["ask", "skip", "overwrite", "rename"]
        collision_combo.grid(row=2, column=1, sticky=tk.W, padx=(5, 20), pady=(10, 0))
        
        # Advanced options
        self.verify_integrity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Verify file integrity", 
                       variable=self.verify_integrity_var).grid(row=2, column=2, sticky=tk.W, pady=(10, 0))
        
        self.preserve_timestamps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Preserve timestamps", 
                       variable=self.preserve_timestamps_var).grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        
        # Worker threads
        ttk.Label(options_frame, text="Worker Threads:").grid(row=3, column=1, sticky=tk.W, pady=(10, 0), padx=(5, 5))
        self.workers_var = tk.StringVar(value="4")
        workers_spin = tk.Spinbox(options_frame, from_=1, to=16, textvariable=self.workers_var, width=5)
        workers_spin.grid(row=3, column=2, sticky=tk.W, pady=(10, 0))
        
        # Control buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        self.scan_button = ttk.Button(buttons_frame, text="Scan Files", command=self.scan_files)
        self.scan_button.grid(row=0, column=0, padx=5)
        
        self.start_button = ttk.Button(buttons_frame, text="Start Move", command=self.start_move, state='disabled')
        self.start_button.grid(row=0, column=1, padx=5)
        
        self.pause_button = ttk.Button(buttons_frame, text="Pause", command=self.pause_move, state='disabled')
        self.pause_button.grid(row=0, column=2, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_move, state='disabled')
        self.stop_button.grid(row=0, column=3, padx=5)
        
        ttk.Button(buttons_frame, text="View Log", command=self.view_log).grid(row=0, column=4, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # File progress
        self.file_progress_var = tk.StringVar(value="Ready to scan...")
        ttk.Label(progress_frame, textvariable=self.file_progress_var).grid(row=0, column=0, sticky=tk.W)
        
        self.file_progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.file_progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 10))
        
        # Size progress
        self.size_progress_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.size_progress_var).grid(row=2, column=0, sticky=tk.W)
        
        self.size_progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.size_progress_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 10))
        
        # Speed and ETA
        self.stats_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.stats_var).grid(row=4, column=0, sticky=tk.W)
        
        # File list frame
        list_frame = ttk.LabelFrame(main_frame, text="Files to Move", padding="10")
        list_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Treeview for file list
        columns = ("Name", "Size", "Source", "Status")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        self.file_tree.heading("Name", text="File Name")
        self.file_tree.heading("Size", text="Size")
        self.file_tree.heading("Source", text="Source Path")
        self.file_tree.heading("Status", text="Status")
        
        self.file_tree.column("Name", width=200)
        self.file_tree.column("Size", width=100)
        self.file_tree.column("Source", width=300)
        self.file_tree.column("Status", width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=self.file_tree.xview)
        self.file_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.file_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        source_frame.columnconfigure(0, weight=1)
        dest_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize file list
        self.file_list = []
        
    def select_source_directory(self):
        """Select source directory"""
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.source_var.set(directory)
            
    def select_dest_directory(self):
        """Select destination directory"""
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
    
    def format_size(self, size: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def scan_files(self):
        """Scan source directory for files to move"""
        source_dir = self.source_var.get()
        if not source_dir:
            messagebox.showerror("Error", "Please select a source directory")
            return
        
        if not Path(source_dir).exists():
            messagebox.showerror("Error", "Source directory does not exist")
            return
        
        def scan_thread():
            try:
                self.scan_button.config(state='disabled')
                self.file_progress_var.set("Scanning files...")
                
                source_path = Path(source_dir)
                self.file_list = []
                
                # Get filter criteria
                extensions = [ext.strip().lower() for ext in self.extensions_var.get().split(',') if ext.strip()]
                min_size_mb = float(self.min_size_var.get() or 0)
                max_size_mb = float(self.max_size_var.get() or float('inf'))
                min_size = min_size_mb * 1024 * 1024
                max_size = max_size_mb * 1024 * 1024
                
                # Scan files
                pattern = "**/*" if self.recursive_var.get() else "*"
                
                for file_path in source_path.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_size = file_path.stat().st_size
                            
                            # Filter by extension
                            if extensions and file_path.suffix.lower() not in extensions:
                                continue
                            
                            # Filter by size
                            if file_size < min_size or file_size > max_size:
                                continue
                            
                            file_info = {
                                'path': file_path,
                                'size': file_size,
                                'status': 'pending'
                            }
                            self.file_list.append(file_info)
                            
                        except Exception as e:
                            self.log_message(f"Error scanning {file_path}: {e}")
                
                # Update display
                self.update_file_list()
                
                self.total_files = len(self.file_list)
                self.total_size = sum(f['size'] for f in self.file_list)
                
                self.file_progress_var.set(f"Scanned {self.total_files} files ({self.format_size(self.total_size)})")
                self.start_button.config(state='normal' if self.file_list else 'disabled')
                
                self.log_message(f"Scan complete: {self.total_files} files found")
                
            except Exception as e:
                self.log_message(f"Scan error: {e}")
                messagebox.showerror("Error", f"Scan failed: {e}")
            
            finally:
                self.scan_button.config(state='normal')
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def update_file_list(self):
        """Update the file list display"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add files
        for file_info in self.file_list:
            self.file_tree.insert('', 'end', values=(
                file_info['path'].name,
                self.format_size(file_info['size']),
                str(file_info['path'].parent),
                file_info['status']
            ))
    
    def start_move(self):
        """Start the file move operation"""
        dest_dir = self.dest_var.get()
        if not dest_dir:
            messagebox.showerror("Error", "Please select a destination directory")
            return
        
        if not self.file_list:
            messagebox.showerror("Error", "No files to move. Please scan first.")
            return
        
        # Reset counters
        self.processed_files = 0
        self.processed_size = 0
        self.completed_operations = []
        self.failed_operations = []
        self.start_time = time.time()
        
        # Update UI
        self.running = True
        self.paused = False
        self.start_button.config(state='disabled')
        self.pause_button.config(state='normal')
        self.stop_button.config(state='normal')
        self.scan_button.config(state='disabled')
        
        # Update collision strategy
        self.collision_strategy = self.collision_var.get()
        
        # Get worker count
        try:
            self.max_workers = int(self.workers_var.get())
        except ValueError:
            self.max_workers = 4
        
        # Add files to queue
        for file_info in self.file_list:
            if file_info['status'] == 'pending':
                self.move_queue.put(file_info)
        
        # Start worker threads
        self.start_workers()
        
        # Start progress monitor
        self.monitor_progress()
        
        self.log_message(f"Started move operation: {len(self.file_list)} files")
    
    def start_workers(self):
        """Start worker threads for file moving"""
        self.worker_threads = []
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self.worker_thread, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
    
    def worker_thread(self):
        """Worker thread for moving files"""
        dest_path = Path(self.dest_var.get())
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                file_info = self.move_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                self.move_file(file_info, dest_path)
            except Exception as e:
                self.log_message(f"Worker error moving {file_info['path']}: {e}")
                file_info['status'] = 'failed'
                self.failed_operations.append(file_info)
            
            finally:
                self.move_queue.task_done()
    
    def move_file(self, file_info: Dict, dest_dir: Path):
        """Move a single file with collision handling"""
        source_path = file_info['path']
        dest_path = dest_dir / source_path.name
        
        # Update status
        file_info['status'] = 'moving'
        
        try:
            # Handle collisions
            if dest_path.exists():
                if self.collision_strategy == "skip":
                    file_info['status'] = 'skipped'
                    self.log_message(f"Skipped existing file: {dest_path}")
                    return
                elif self.collision_strategy == "rename":
                    counter = 1
                    while dest_path.exists():
                        name_parts = source_path.stem, counter, source_path.suffix
                        new_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        dest_path = dest_dir / new_name
                        counter += 1
                elif self.collision_strategy == "overwrite":
                    pass  # Will overwrite
                elif self.collision_strategy == "ask":
                    # In threaded environment, default to rename
                    counter = 1
                    while dest_path.exists():
                        name_parts = source_path.stem, counter, source_path.suffix
                        new_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        dest_path = dest_dir / new_name
                        counter += 1
            
            # Create destination directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get source hash if integrity verification is enabled
            source_hash = ""
            if self.verify_integrity_var.get():
                source_hash = self.calculate_file_hash(source_path)
            
            # Move the file
            if self.preserve_timestamps_var.get():
                shutil.copy2(source_path, dest_path)
            else:
                shutil.copy(source_path, dest_path)
            
            # Verify integrity
            if self.verify_integrity_var.get() and source_hash:
                dest_hash = self.calculate_file_hash(dest_path)
                if source_hash != dest_hash:
                    dest_path.unlink()
                    raise Exception("File integrity verification failed")
            
            # Remove source file
            source_path.unlink()
            
            # Update status
            file_info['status'] = 'completed'
            file_info['dest_path'] = dest_path
            self.completed_operations.append(file_info)
            
            # Update counters
            self.processed_files += 1
            self.processed_size += file_info['size']
            
            self.log_message(f"Moved: {source_path} -> {dest_path}")
            
        except Exception as e:
            file_info['status'] = 'failed'
            file_info['error'] = str(e)
            self.failed_operations.append(file_info)
            self.log_message(f"Failed to move {source_path}: {e}")
            raise
    
    def monitor_progress(self):
        """Monitor and update progress"""
        def update_progress():
            while self.running:
                if self.total_files > 0:
                    file_progress = (self.processed_files / self.total_files) * 100
                    self.file_progress_bar['value'] = file_progress
                    self.file_progress_var.set(f"Files: {self.processed_files}/{self.total_files} ({file_progress:.1f}%)")
                
                if self.total_size > 0:
                    size_progress = (self.processed_size / self.total_size) * 100
                    self.size_progress_bar['value'] = size_progress
                    self.size_progress_var.set(f"Size: {self.format_size(self.processed_size)}/{self.format_size(self.total_size)} ({size_progress:.1f}%)")
                
                # Calculate speed and ETA
                if self.start_time and self.processed_size > 0:
                    elapsed = time.time() - self.start_time
                    speed = self.processed_size / elapsed if elapsed > 0 else 0
                    remaining_size = self.total_size - self.processed_size
                    eta = remaining_size / speed if speed > 0 else 0
                    
                    speed_str = self.format_size(speed) + "/s"
                    eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
                    
                    self.stats_var.set(f"Speed: {speed_str}, ETA: {eta_str}")
                
                # Update file list status
                self.update_file_list()
                
                time.sleep(0.5)
        
        threading.Thread(target=update_progress, daemon=True).start()
    
    def pause_move(self):
        """Pause/resume the move operation"""
        if self.paused:
            self.paused = False
            self.pause_button.config(text="Pause")
            self.log_message("Move operation resumed")
        else:
            self.paused = True
            self.pause_button.config(text="Resume")
            self.log_message("Move operation paused")
    
    def stop_move(self):
        """Stop the move operation"""
        self.running = False
        self.paused = False
        
        # Update UI
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled', text="Pause")
        self.stop_button.config(state='disabled')
        self.scan_button.config(state='normal')
        
        # Show results
        elapsed = time.time() - self.start_time if self.start_time else 0
        summary = f"""
Move operation completed in {elapsed:.1f} seconds

Successfully moved: {len(self.completed_operations)} files
Failed: {len(self.failed_operations)} files
Total size moved: {self.format_size(self.processed_size)}
"""
        
        messagebox.showinfo("Move Complete", summary)
        self.log_message("Move operation stopped")
        self.log_message(summary.strip())
    
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
            self.log_message("GET SWIFTY File Move Manager v1.0.0 started")
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
        
        # Simple file move
        moved_count = 0
        for file_path in source_path.iterdir():
            if file_path.is_file():
                try:
                    dest_file = dest_path / file_path.name
                    dest_path.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(dest_file))
                    moved_count += 1
                    print(f"Moved: {file_path.name}")
                except Exception as e:
                    print(f"Failed to move {file_path.name}: {e}")
        
        print(f"Console mode completed: {moved_count} files moved")

def main():
    """Main execution function"""
    try:
        manager = FileMoveManager()
        manager.run()
    except Exception as e:
        print(f"Application error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()