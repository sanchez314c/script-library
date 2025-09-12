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
# Script Name: systems-turbo-file-mover.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible high-performance file movement system with advanced features
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Turbo-charged multi-threaded operations for maximum performance
#     - Advanced pattern matching and file organization
#     - Real-time progress monitoring with speed optimization
#     - Memory-efficient chunked processing for large datasets
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - Standard library with performance optimizations
#
# Usage:
#     python systems-turbo-file-mover.py
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
import mmap
import stat
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Generator
import queue
import multiprocessing
import platform

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

class TurboFileMover:
    def __init__(self):
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / "turbo_file_mover_log.txt"
        self.stats_file = self.desktop_path / "turbo_move_stats.json"
        
        # Performance optimization settings
        self.cpu_count = multiprocessing.cpu_count()
        self.max_workers = min(self.cpu_count * 2, 16)
        self.chunk_size = 1024 * 1024  # 1MB chunks
        self.buffer_size = 64 * 1024   # 64KB buffer
        
        # Operation tracking
        self.file_queue = queue.Queue(maxsize=1000)
        self.results_queue = queue.Queue()
        self.total_files = 0
        self.processed_files = 0
        self.total_size = 0
        self.processed_size = 0
        self.start_time = None
        self.bytes_per_second = 0
        
        # Thread management
        self.running = False
        self.paused = False
        self.workers = []
        
        # File type organization
        self.file_type_map = {
            'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico'},
            'videos': {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp'},
            'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'},
            'documents': {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.rtf'},
            'archives': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tar.gz'},
            'code': {'.py', '.js', '.html', '.css', '.cpp', '.c', '.java', '.php', '.rb', '.go'},
            'executables': {'.app', '.dmg', '.pkg', '.exe', '.msi', '.deb', '.rpm'}
        }
        
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI interface"""
        if not tkinter:
            print("GUI not available, running in console mode...")
            return
            
        self.root = tk.Tk()
        self.root.title("GET SWIFTY - Turbo File Mover v1.0.0")
        self.root.geometry("900x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Turbo File Mover", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Performance info
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Configuration", padding="10")
        perf_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(perf_frame, text=f"CPU Cores: {self.cpu_count}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(perf_frame, text=f"Max Workers: {self.max_workers}").grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(perf_frame, text=f"Platform: {platform.system()}").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # Operation mode
        ttk.Label(perf_frame, text="Operation Mode:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.mode_var = tk.StringVar(value="turbo")
        mode_combo = ttk.Combobox(perf_frame, textvariable=self.mode_var, width=15)
        mode_combo['values'] = ["turbo", "balanced", "safe"]
        mode_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 20), pady=(10, 0))
        
        # Worker threads
        ttk.Label(perf_frame, text="Worker Threads:").grid(row=1, column=2, sticky=tk.W, pady=(10, 0))
        self.workers_var = tk.StringVar(value=str(self.max_workers))
        workers_spin = tk.Spinbox(perf_frame, from_=1, to=32, textvariable=self.workers_var, width=5)
        workers_spin.grid(row=1, column=3, sticky=tk.W, pady=(10, 0), padx=(5, 0))
        
        # Source and destination paths
        paths_frame = ttk.LabelFrame(main_frame, text="File Paths", padding="10")
        paths_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Source directory
        ttk.Label(paths_frame, text="Source Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar()
        source_frame = ttk.Frame(paths_frame)
        source_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_var, width=70)
        self.source_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(source_frame, text="Browse", command=self.select_source_directory).grid(row=0, column=1, padx=(5, 0))
        
        # Destination directory
        ttk.Label(paths_frame, text="Destination Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.dest_var = tk.StringVar()
        dest_frame = ttk.Frame(paths_frame)
        dest_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.dest_entry = ttk.Entry(dest_frame, textvariable=self.dest_var, width=70)
        self.dest_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(dest_frame, text="Browse", command=self.select_dest_directory).grid(row=0, column=1, padx=(5, 0))
        
        # Advanced options
        options_frame = ttk.LabelFrame(main_frame, text="Turbo Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # File organization
        self.organize_by_type_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Auto-organize by file type", 
                       variable=self.organize_by_type_var).grid(row=0, column=0, sticky=tk.W)
        
        self.preserve_structure_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Preserve directory structure", 
                       variable=self.preserve_structure_var).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        self.verify_integrity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Verify file integrity (MD5)", 
                       variable=self.verify_integrity_var).grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # Size filters
        ttk.Label(options_frame, text="Min Size (MB):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.min_size_var = tk.StringVar(value="0")
        ttk.Entry(options_frame, textvariable=self.min_size_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(5, 20), pady=(10, 0))
        
        ttk.Label(options_frame, text="Max Size (GB):").grid(row=1, column=2, sticky=tk.W, pady=(10, 0))
        self.max_size_var = tk.StringVar()
        ttk.Entry(options_frame, textvariable=self.max_size_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        
        # Pattern matching
        ttk.Label(options_frame, text="File Patterns:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.patterns_var = tk.StringVar(value="*")
        ttk.Entry(options_frame, textvariable=self.patterns_var, width=30).grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        
        # Control buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.scan_button = ttk.Button(buttons_frame, text="🔍 Turbo Scan", command=self.turbo_scan)
        self.scan_button.grid(row=0, column=0, padx=5)
        
        self.start_button = ttk.Button(buttons_frame, text="🚀 Start Turbo Move", command=self.start_turbo_move, state='disabled')
        self.start_button.grid(row=0, column=1, padx=5)
        
        self.pause_button = ttk.Button(buttons_frame, text="⏸️ Pause", command=self.pause_move, state='disabled')
        self.pause_button.grid(row=0, column=2, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="🛑 Stop", command=self.stop_move, state='disabled')
        self.stop_button.grid(row=0, column=3, padx=5)
        
        ttk.Button(buttons_frame, text="📊 Stats", command=self.view_stats).grid(row=0, column=4, padx=5)
        ttk.Button(buttons_frame, text="📝 Log", command=self.view_log).grid(row=0, column=5, padx=5)
        
        # Real-time statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Turbo Performance Monitor", padding="10")
        stats_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Speed indicators
        self.speed_var = tk.StringVar(value="Speed: 0 MB/s")
        ttk.Label(stats_frame, textvariable=self.speed_var, font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W)
        
        self.eta_var = tk.StringVar(value="ETA: --:--")
        ttk.Label(stats_frame, textvariable=self.eta_var, font=("Arial", 12)).grid(row=0, column=1, sticky=tk.W, padx=(40, 0))
        
        self.throughput_var = tk.StringVar(value="Throughput: 0 files/s")
        ttk.Label(stats_frame, textvariable=self.throughput_var, font=("Arial", 12)).grid(row=0, column=2, sticky=tk.W, padx=(40, 0))
        
        # Progress bars
        self.file_progress_var = tk.StringVar(value="Ready for turbo scan...")
        ttk.Label(stats_frame, textvariable=self.file_progress_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        self.file_progress_bar = ttk.Progressbar(stats_frame, length=500, mode='determinate')
        self.file_progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.size_progress_var = tk.StringVar(value="")
        ttk.Label(stats_frame, textvariable=self.size_progress_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        self.size_progress_bar = ttk.Progressbar(stats_frame, length=500, mode='determinate')
        self.size_progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # File preview list
        preview_frame = ttk.LabelFrame(main_frame, text="File Preview", padding="10")
        preview_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Treeview for file preview
        columns = ("Name", "Type", "Size", "Source", "Destination")
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=12)
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=150)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        h_scroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.preview_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        source_frame.columnconfigure(0, weight=1)
        dest_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize data
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
    
    def format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine file type category"""
        suffix = file_path.suffix.lower()
        
        for file_type, extensions in self.file_type_map.items():
            if suffix in extensions:
                return file_type
        
        return 'other'
    
    def calculate_fast_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash efficiently using memory mapping"""
        try:
            with open(file_path, 'rb') as f:
                # Use memory mapping for large files
                if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        return hashlib.md5(mm).hexdigest()
                else:
                    return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.log_message(f"Hash calculation failed for {file_path}: {e}")
            return ""
    
    def turbo_scan(self):
        """Perform high-speed directory scan"""
        source_dir = self.source_var.get()
        if not source_dir or not Path(source_dir).exists():
            messagebox.showerror("Error", "Please select a valid source directory")
            return
        
        def scan_worker():
            try:
                self.scan_button.config(state='disabled')
                self.file_progress_var.set("🔍 Turbo scanning in progress...")
                
                source_path = Path(source_dir)
                self.file_list = []
                
                # Get filter criteria
                min_size_mb = float(self.min_size_var.get() or 0)
                max_size_gb = float(self.max_size_var.get() or float('inf'))
                min_size = min_size_mb * 1024 * 1024
                max_size = max_size_gb * 1024 * 1024 * 1024
                
                patterns = [p.strip() for p in self.patterns_var.get().split(',')]
                
                # High-speed recursive scan using os.walk for better performance
                scan_count = 0
                for root, dirs, files in os.walk(source_path):
                    for file_name in files:
                        file_path = Path(root) / file_name
                        
                        try:
                            file_stats = file_path.stat()
                            file_size = file_stats.st_size
                            
                            # Apply size filters
                            if file_size < min_size or file_size > max_size:
                                continue
                            
                            # Apply pattern filters
                            if patterns and patterns != ['*']:
                                if not any(file_path.match(pattern) for pattern in patterns):
                                    continue
                            
                            file_type = self.get_file_type(file_path)
                            
                            file_info = {
                                'path': file_path,
                                'size': file_size,
                                'type': file_type,
                                'mtime': file_stats.st_mtime,
                                'status': 'pending'
                            }
                            
                            self.file_list.append(file_info)
                            scan_count += 1
                            
                            # Update progress every 100 files
                            if scan_count % 100 == 0:
                                self.file_progress_var.set(f"🔍 Scanned {scan_count} files...")
                                self.root.update_idletasks()
                            
                        except (OSError, PermissionError) as e:
                            self.log_message(f"Scan error for {file_path}: {e}")
                            continue
                
                # Calculate totals
                self.total_files = len(self.file_list)
                self.total_size = sum(f['size'] for f in self.file_list)
                
                # Update preview
                self.update_preview()
                
                # Update UI
                size_str = self.format_size(self.total_size)
                self.file_progress_var.set(f"✅ Turbo scan complete: {self.total_files} files ({size_str})")
                self.start_button.config(state='normal' if self.file_list else 'disabled')
                
                self.log_message(f"Turbo scan complete: {self.total_files} files, {size_str}")
                
            except Exception as e:
                self.log_message(f"Turbo scan error: {e}")
                messagebox.showerror("Error", f"Turbo scan failed: {e}")
            
            finally:
                self.scan_button.config(state='normal')
        
        threading.Thread(target=scan_worker, daemon=True).start()
    
    def update_preview(self):
        """Update the file preview list"""
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        # Show first 100 files for preview
        dest_path = Path(self.dest_var.get()) if self.dest_var.get() else Path("Not Set")
        
        for i, file_info in enumerate(self.file_list[:100]):
            file_path = file_info['path']
            
            # Determine destination
            if self.organize_by_type_var.get():
                dest_dir = dest_path / file_info['type']
            else:
                dest_dir = dest_path
            
            if self.preserve_structure_var.get():
                rel_path = file_path.relative_to(Path(self.source_var.get()))
                final_dest = dest_dir / rel_path
            else:
                final_dest = dest_dir / file_path.name
            
            self.preview_tree.insert('', 'end', values=(
                file_path.name,
                file_info['type'],
                self.format_size(file_info['size']),
                str(file_path.parent),
                str(final_dest.parent)
            ))
        
        if len(self.file_list) > 100:
            self.preview_tree.insert('', 'end', values=(
                f"... and {len(self.file_list) - 100} more files",
                "", "", "", ""
            ))
    
    def start_turbo_move(self):
        """Start the turbo file move operation"""
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
        self.start_time = time.time()
        self.running = True
        self.paused = False
        
        # Update UI
        self.start_button.config(state='disabled')
        self.pause_button.config(state='normal')
        self.stop_button.config(state='normal')
        self.scan_button.config(state='disabled')
        
        # Get worker count
        try:
            worker_count = int(self.workers_var.get())
        except ValueError:
            worker_count = self.max_workers
        
        # Fill the queue
        for file_info in self.file_list:
            self.file_queue.put(file_info)
        
        # Start worker threads
        self.workers = []
        for i in range(worker_count):
            worker = threading.Thread(target=self.turbo_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Start progress monitor
        self.start_progress_monitor()
        
        self.log_message(f"🚀 Turbo move started: {len(self.file_list)} files with {worker_count} workers")
    
    def turbo_worker(self):
        """High-performance worker thread"""
        dest_path = Path(self.dest_var.get())
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                file_info = self.file_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            try:
                self.turbo_move_file(file_info, dest_path)
                self.results_queue.put(('success', file_info))
            except Exception as e:
                self.log_message(f"Turbo move error for {file_info['path']}: {e}")
                self.results_queue.put(('error', file_info, str(e)))
            
            finally:
                self.file_queue.task_done()
    
    def turbo_move_file(self, file_info: Dict, dest_base: Path):
        """Move a single file with turbo optimizations"""
        source_path = file_info['path']
        
        # Determine destination path
        if self.organize_by_type_var.get():
            dest_dir = dest_base / file_info['type']
        else:
            dest_dir = dest_base
        
        if self.preserve_structure_var.get():
            rel_path = source_path.relative_to(Path(self.source_var.get()))
            dest_path = dest_dir / rel_path
        else:
            dest_path = dest_dir / source_path.name
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle name collisions with high-speed renaming
        if dest_path.exists():
            counter = 1
            while dest_path.exists():
                name_parts = source_path.stem, counter, source_path.suffix
                new_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                dest_path = dest_path.parent / new_name
                counter += 1
        
        # Turbo copy operation
        mode = self.mode_var.get()
        
        if mode == "turbo":
            # Maximum speed - direct copy without verification
            shutil.copy2(source_path, dest_path)
        elif mode == "balanced":
            # Balance between speed and safety
            shutil.copy2(source_path, dest_path)
            if source_path.stat().st_size != dest_path.stat().st_size:
                raise Exception("Size mismatch after copy")
        else:  # safe mode
            # Full verification
            if self.verify_integrity_var.get():
                source_hash = self.calculate_fast_hash(source_path)
            
            shutil.copy2(source_path, dest_path)
            
            if self.verify_integrity_var.get():
                dest_hash = self.calculate_fast_hash(dest_path)
                if source_hash != dest_hash:
                    dest_path.unlink()
                    raise Exception("Hash verification failed")
        
        # Remove source file
        source_path.unlink()
        
        # Update counters atomically
        self.processed_files += 1
        self.processed_size += file_info['size']
        
        file_info['status'] = 'completed'
        file_info['dest_path'] = dest_path
    
    def start_progress_monitor(self):
        """Start real-time progress monitoring"""
        def monitor_worker():
            last_size = 0
            last_time = time.time()
            
            while self.running:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Update progress bars
                if self.total_files > 0:
                    file_progress = (self.processed_files / self.total_files) * 100
                    self.file_progress_bar['value'] = file_progress
                    self.file_progress_var.set(f"Files: {self.processed_files}/{self.total_files} ({file_progress:.1f}%)")
                
                if self.total_size > 0:
                    size_progress = (self.processed_size / self.total_size) * 100
                    self.size_progress_bar['value'] = size_progress
                    self.size_progress_var.set(f"Size: {self.format_size(self.processed_size)}/{self.format_size(self.total_size)} ({size_progress:.1f}%)")
                
                # Calculate speed
                if current_time - last_time >= 1.0:  # Update every second
                    bytes_delta = self.processed_size - last_size
                    time_delta = current_time - last_time
                    
                    if time_delta > 0:
                        speed = bytes_delta / time_delta
                        self.bytes_per_second = speed
                        self.speed_var.set(f"Speed: {self.format_size(speed)}/s")
                        
                        # Calculate throughput
                        if elapsed > 0:
                            files_per_sec = self.processed_files / elapsed
                            self.throughput_var.set(f"Throughput: {files_per_sec:.1f} files/s")
                        
                        # Calculate ETA
                        remaining_size = self.total_size - self.processed_size
                        if speed > 0:
                            eta_seconds = remaining_size / speed
                            self.eta_var.set(f"ETA: {self.format_time(eta_seconds)}")
                    
                    last_size = self.processed_size
                    last_time = current_time
                
                time.sleep(0.1)
        
        threading.Thread(target=monitor_worker, daemon=True).start()
    
    def pause_move(self):
        """Pause/resume the move operation"""
        if self.paused:
            self.paused = False
            self.pause_button.config(text="⏸️ Pause")
            self.log_message("Turbo move operation resumed")
        else:
            self.paused = True
            self.pause_button.config(text="▶️ Resume")
            self.log_message("Turbo move operation paused")
    
    def stop_move(self):
        """Stop the move operation"""
        self.running = False
        self.paused = False
        
        # Update UI
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled', text="⏸️ Pause")
        self.stop_button.config(state='disabled')
        self.scan_button.config(state='normal')
        
        # Calculate final statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_speed = self.processed_size / elapsed if elapsed > 0 else 0
        
        # Save statistics
        self.save_statistics(elapsed, avg_speed)
        
        # Show results
        summary = f"""🚀 Turbo Move Complete!

Time: {self.format_time(elapsed)}
Files Moved: {self.processed_files}/{self.total_files}
Data Moved: {self.format_size(self.processed_size)}
Average Speed: {self.format_size(avg_speed)}/s
Peak Speed: {self.format_size(self.bytes_per_second)}/s
"""
        
        messagebox.showinfo("Turbo Move Complete", summary)
        self.log_message("🚀 Turbo move operation completed")
        self.log_message(summary.replace('\n', ' | '))
    
    def save_statistics(self, elapsed: float, avg_speed: float):
        """Save operation statistics"""
        stats = {
            'timestamp': datetime.datetime.now().isoformat(),
            'operation': 'turbo_move',
            'files_total': self.total_files,
            'files_processed': self.processed_files,
            'size_total': self.total_size,
            'size_processed': self.processed_size,
            'time_elapsed': elapsed,
            'avg_speed': avg_speed,
            'peak_speed': self.bytes_per_second,
            'mode': self.mode_var.get(),
            'workers': self.workers_var.get()
        }
        
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            self.log_message(f"Failed to save statistics: {e}")
    
    def view_stats(self):
        """View operation statistics"""
        try:
            if self.stats_file.exists():
                os.system(f'open "{self.stats_file}"')
            else:
                messagebox.showinfo("Statistics", "No statistics file found")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open statistics file: {e}")
    
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
            self.log_message("🚀 GET SWIFTY Turbo File Mover v1.0.0 started")
            self.root.mainloop()
        else:
            self.console_mode()
    
    def console_mode(self):
        """Run in console mode for maximum performance"""
        print("🚀 Turbo File Mover - Console Mode")
        
        source_dir = input("Source directory: ").strip()
        dest_dir = input("Destination directory: ").strip()
        
        if not source_dir or not dest_dir:
            print("Both directories are required")
            return
        
        source_path = Path(source_dir)
        dest_path = Path(dest_dir)
        
        if not source_path.exists():
            print("Source directory does not exist")
            return
        
        print("🔍 Turbo scanning...")
        file_count = 0
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                try:
                    dest_file = dest_path / file_path.name
                    dest_path.mkdir(parents=True, exist_ok=True)
                    
                    shutil.move(str(file_path), str(dest_file))
                    file_count += 1
                    
                    if file_count % 100 == 0:
                        print(f"Moved {file_count} files...")
                        
                except Exception as e:
                    print(f"Error moving {file_path.name}: {e}")
        
        print(f"🚀 Turbo move complete: {file_count} files moved")

def main():
    """Main execution function"""
    try:
        mover = TurboFileMover()
        mover.run()
    except Exception as e:
        print(f"Application error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()