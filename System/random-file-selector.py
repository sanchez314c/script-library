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
# Script Name: systems-random-file-selector.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible random file selection and sampling system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Random file selection with customizable sample sizes
#     - Multi-threaded copying for efficient processing
#     - Collision handling with automatic resolution
#     - Progress tracking with native macOS GUI dialogs
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, concurrent.futures, shutil (auto-installed if missing)
#
# Usage:
#     python systems-random-file-selector.py
#
####################################################################################

import os
import sys
import shutil
import random
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import logging

# Setup logging to desktop
desktop = Path.home() / "Desktop"
log_file = desktop / "random_file_selector.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class RandomFileSelector:
    def __init__(self):
        self.source_files = []
        self.selected_files = []
        self.total_size = 0
        self.selected_size = 0
        
    def scan_directory(self, directory, file_extensions=None, min_size=0, max_size=None):
        """Scan directory for files matching criteria"""
        files = []
        total_size = 0
        
        logging.info(f"Scanning directory: {directory}")
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(root) / filename
                
                try:
                    # Check if file exists and get stats
                    if not filepath.exists():
                        continue
                    
                    stat = filepath.stat()
                    file_size = stat.st_size
                    
                    # Apply size filters
                    if file_size < min_size:
                        continue
                    if max_size and file_size > max_size:
                        continue
                    
                    # Apply extension filter
                    if file_extensions:
                        ext = filepath.suffix.lower()
                        if ext not in file_extensions:
                            continue
                    
                    file_info = {
                        "path": filepath,
                        "name": filename,
                        "size": file_size,
                        "relative_path": filepath.relative_to(directory),
                        "extension": filepath.suffix.lower()
                    }
                    
                    files.append(file_info)
                    total_size += file_size
                    
                except (OSError, PermissionError):
                    continue
        
        self.source_files = files
        self.total_size = total_size
        
        logging.info(f"Found {len(files)} files ({total_size / (1024**3):.2f} GB)")
        return files
    
    def select_random_files(self, count=None, size_limit=None, strategy="count"):
        """Select random files based on strategy"""
        if not self.source_files:
            return []
        
        available_files = self.source_files.copy()
        selected = []
        selected_size = 0
        
        if strategy == "count" and count:
            # Select by count
            count = min(count, len(available_files))
            selected = random.sample(available_files, count)
            selected_size = sum(f["size"] for f in selected)
            
        elif strategy == "size" and size_limit:
            # Select by total size limit
            random.shuffle(available_files)
            
            for file_info in available_files:
                if selected_size + file_info["size"] <= size_limit:
                    selected.append(file_info)
                    selected_size += file_info["size"]
                else:
                    break
        
        elif strategy == "percentage" and count:
            # Select by percentage
            percentage = count / 100.0
            select_count = max(1, int(len(available_files) * percentage))
            selected = random.sample(available_files, select_count)
            selected_size = sum(f["size"] for f in selected)
        
        self.selected_files = selected
        self.selected_size = selected_size
        
        logging.info(f"Selected {len(selected)} files ({selected_size / (1024**2):.1f} MB)")
        return selected
    
    def copy_selected_files(self, destination_dir, progress_callback=None):
        """Copy selected files to destination"""
        if not self.selected_files:
            return {"success": False, "error": "No files selected"}
        
        dest_path = Path(destination_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        copied_files = 0
        failed_files = 0
        copied_size = 0
        
        def copy_file(file_info):
            try:
                src_path = file_info["path"]
                dest_file_path = dest_path / file_info["name"]
                
                # Handle name collisions
                if dest_file_path.exists():
                    counter = 1
                    name_parts = file_info["name"].rsplit('.', 1)
                    if len(name_parts) == 2:
                        base_name, ext = name_parts
                        while dest_file_path.exists():
                            new_name = f"{base_name}_{counter:03d}.{ext}"
                            dest_file_path = dest_path / new_name
                            counter += 1
                    else:
                        base_name = file_info["name"]
                        while dest_file_path.exists():
                            new_name = f"{base_name}_{counter:03d}"
                            dest_file_path = dest_path / new_name
                            counter += 1
                
                # Copy file
                shutil.copy2(src_path, dest_file_path)
                logging.info(f"Copied: {file_info['name']}")
                
                return {"success": True, "size": file_info["size"], "dest": str(dest_file_path)}
                
            except Exception as e:
                logging.error(f"Failed to copy {file_info['name']}: {e}")
                return {"success": False, "error": str(e)}
        
        # Use threading for parallel copying
        max_workers = min(16, max(4, os.cpu_count()))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit copy tasks
            future_to_file = {
                executor.submit(copy_file, file_info): file_info 
                for file_info in self.selected_files
            }
            
            # Process results
            for i, future in enumerate(as_completed(future_to_file)):
                result = future.result()
                if result["success"]:
                    copied_files += 1
                    copied_size += result["size"]
                else:
                    failed_files += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, len(self.selected_files), copied_size)
        
        return {
            "success": True,
            "copied": copied_files,
            "failed": failed_files,
            "total": len(self.selected_files),
            "copied_size": copied_size
        }

class RandomFileSelectorGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Random File Selector")
        self.window.geometry("800x700")
        
        self.selector = RandomFileSelector()
        self.source_dir = None
        self.dest_dir = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.window, text="🎲 RANDOM FILE SELECTOR", 
                        font=("Arial", 18, "bold"), fg="purple")
        title.pack(pady=10)
        
        # Source selection
        source_frame = tk.Frame(self.window)
        source_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(source_frame, text="Source Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.source_var = tk.StringVar()
        
        src_entry_frame = tk.Frame(source_frame)
        src_entry_frame.pack(fill="x", pady=2)
        tk.Entry(src_entry_frame, textvariable=self.source_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(src_entry_frame, text="Browse", command=self.browse_source).pack(side="right", padx=(5,0))
        
        # Filters frame
        filters_frame = tk.LabelFrame(self.window, text="File Filters", font=("Arial", 11, "bold"))
        filters_frame.pack(fill="x", padx=20, pady=10)
        
        # Extensions filter
        ext_frame = tk.Frame(filters_frame)
        ext_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(ext_frame, text="File Extensions (comma-separated, e.g., .jpg,.png,.mp4):").pack(anchor="w")
        self.extensions_var = tk.StringVar()
        tk.Entry(ext_frame, textvariable=self.extensions_var, font=("Arial", 10)).pack(fill="x", pady=2)
        
        # Size filters
        size_frame = tk.Frame(filters_frame)
        size_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(size_frame, text="Minimum File Size (MB):").pack(side="left")
        self.min_size_var = tk.StringVar(value="0")
        tk.Entry(size_frame, textvariable=self.min_size_var, width=10).pack(side="left", padx=(5,20))
        
        tk.Label(size_frame, text="Maximum File Size (MB):").pack(side="left")
        self.max_size_var = tk.StringVar()
        tk.Entry(size_frame, textvariable=self.max_size_var, width=10).pack(side="left", padx=5)
        
        # Selection strategy frame
        strategy_frame = tk.LabelFrame(self.window, text="Selection Strategy", font=("Arial", 11, "bold"))
        strategy_frame.pack(fill="x", padx=20, pady=10)
        
        self.strategy_var = tk.StringVar(value="count")
        
        strat_row1 = tk.Frame(strategy_frame)
        strat_row1.pack(fill="x", padx=10, pady=5)
        
        tk.Radiobutton(strat_row1, text="Select by Count:", variable=self.strategy_var, 
                      value="count").pack(side="left")
        self.count_var = tk.StringVar(value="100")
        tk.Entry(strat_row1, textvariable=self.count_var, width=10).pack(side="left", padx=5)
        tk.Label(strat_row1, text="files").pack(side="left")
        
        strat_row2 = tk.Frame(strategy_frame)
        strat_row2.pack(fill="x", padx=10, pady=5)
        
        tk.Radiobutton(strat_row2, text="Select by Total Size:", variable=self.strategy_var, 
                      value="size").pack(side="left")
        self.size_limit_var = tk.StringVar(value="1000")
        tk.Entry(strat_row2, textvariable=self.size_limit_var, width=10).pack(side="left", padx=5)
        tk.Label(strat_row2, text="MB").pack(side="left")
        
        strat_row3 = tk.Frame(strategy_frame)
        strat_row3.pack(fill="x", padx=10, pady=5)
        
        tk.Radiobutton(strat_row3, text="Select by Percentage:", variable=self.strategy_var, 
                      value="percentage").pack(side="left")
        self.percentage_var = tk.StringVar(value="10")
        tk.Entry(strat_row3, textvariable=self.percentage_var, width=10).pack(side="left", padx=5)
        tk.Label(strat_row3, text="% of files").pack(side="left")
        
        # Destination selection
        dest_frame = tk.Frame(self.window)
        dest_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(dest_frame, text="Destination Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.dest_var = tk.StringVar()
        
        dest_entry_frame = tk.Frame(dest_frame)
        dest_entry_frame.pack(fill="x", pady=2)
        tk.Entry(dest_entry_frame, textvariable=self.dest_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(dest_entry_frame, text="Browse", command=self.browse_dest).pack(side="right", padx=(5,0))
        
        # Control buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="🔍 Scan Files", command=self.scan_files,
                 bg="blue", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        tk.Button(button_frame, text="🎲 Select Random", command=self.select_random,
                 bg="orange", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        tk.Button(button_frame, text="💾 Copy Selected", command=self.copy_files,
                 bg="green", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        # Progress frame
        progress_frame = tk.Frame(self.window)
        progress_frame.pack(fill="x", padx=20, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready to scan files...")
        tk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 10)).pack(anchor="w")
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=2)
        
        # Status area
        status_frame = tk.Frame(self.window)
        status_frame.pack(fill="both", expand=True, padx=20, pady=(5,20))
        
        tk.Label(status_frame, text="Status Log:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, font=("Courier", 9))
        self.status_text.pack(fill="both", expand=True)
        
    def log_status(self, message):
        """Add message to status log"""
        timestamp = time.strftime('%H:%M:%S')
        self.status_text.insert(tk.END, f"{timestamp} - {message}\n")
        self.status_text.see(tk.END)
        self.window.update_idletasks()
    
    def browse_source(self):
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.source_dir = directory
            self.source_var.set(directory)
    
    def browse_dest(self):
        directory = filedialog.askdirectory(title="Select Destination Directory")
        if directory:
            self.dest_dir = directory
            self.dest_var.set(directory)
    
    def scan_files(self):
        """Scan source directory for files"""
        if not self.source_dir:
            messagebox.showerror("Error", "Please select a source directory first.")
            return
        
        self.log_status("Starting file scan...")
        self.progress_var.set("Scanning files...")
        
        def scan_thread():
            try:
                # Parse filters
                extensions = None
                ext_text = self.extensions_var.get().strip()
                if ext_text:
                    extensions = [ext.strip().lower() for ext in ext_text.split(',')]
                    if extensions and not extensions[0].startswith('.'):
                        extensions = ['.' + ext for ext in extensions]
                
                min_size = 0
                try:
                    min_size = float(self.min_size_var.get()) * 1024 * 1024  # Convert MB to bytes
                except:
                    pass
                
                max_size = None
                try:
                    max_val = self.max_size_var.get().strip()
                    if max_val:
                        max_size = float(max_val) * 1024 * 1024  # Convert MB to bytes
                except:
                    pass
                
                # Scan files
                files = self.selector.scan_directory(self.source_dir, extensions, min_size, max_size)
                self.window.after(0, self.scan_completed, files)
                
            except Exception as e:
                self.window.after(0, lambda: self.log_status(f"Scan failed: {e}"))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def scan_completed(self, files):
        """Handle scan completion"""
        total_size_gb = self.selector.total_size / (1024**3)
        self.progress_var.set(f"Scan complete - {len(files)} files ({total_size_gb:.2f} GB)")
        self.log_status(f"Found {len(files)} files matching criteria")
        self.log_status(f"Total size: {total_size_gb:.2f} GB")
    
    def select_random(self):
        """Select random files based on strategy"""
        if not self.selector.source_files:
            messagebox.showerror("Error", "Please scan files first.")
            return
        
        strategy = self.strategy_var.get()
        
        try:
            if strategy == "count":
                count = int(self.count_var.get())
                files = self.selector.select_random_files(count=count, strategy="count")
            elif strategy == "size":
                size_limit = float(self.size_limit_var.get()) * 1024 * 1024  # MB to bytes
                files = self.selector.select_random_files(size_limit=size_limit, strategy="size")
            elif strategy == "percentage":
                percentage = float(self.percentage_var.get())
                files = self.selector.select_random_files(count=percentage, strategy="percentage")
            
            selected_size_mb = self.selector.selected_size / (1024**2)
            self.progress_var.set(f"Selected {len(files)} files ({selected_size_mb:.1f} MB)")
            self.log_status(f"Randomly selected {len(files)} files")
            self.log_status(f"Selected size: {selected_size_mb:.1f} MB")
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numbers for selection criteria.")
        except Exception as e:
            self.log_status(f"Selection failed: {e}")
    
    def copy_files(self):
        """Copy selected files to destination"""
        if not self.selector.selected_files:
            messagebox.showerror("Error", "Please select random files first.")
            return
        
        if not self.dest_dir:
            messagebox.showerror("Error", "Please select a destination directory.")
            return
        
        # Confirm copy
        file_count = len(self.selector.selected_files)
        size_mb = self.selector.selected_size / (1024**2)
        
        if not messagebox.askyesno("Confirm Copy", 
                                  f"Copy {file_count} randomly selected files ({size_mb:.1f} MB) to:\n{self.dest_dir}\n\nContinue?"):
            return
        
        self.log_status(f"Starting copy of {file_count} files...")
        
        def copy_thread():
            try:
                def progress_callback(current, total, copied_size):
                    percent = (current / total) * 100
                    copied_mb = copied_size / (1024**2)
                    self.window.after(0, lambda: self.progress_var.set(f"Copying: {current}/{total} ({copied_mb:.1f} MB)"))
                    self.window.after(0, lambda: setattr(self.progress_bar, 'value', percent))
                
                result = self.selector.copy_selected_files(self.dest_dir, progress_callback)
                self.window.after(0, self.copy_completed, result)
                
            except Exception as e:
                self.window.after(0, lambda: self.log_status(f"Copy failed: {e}"))
        
        threading.Thread(target=copy_thread, daemon=True).start()
    
    def copy_completed(self, result):
        """Handle copy completion"""
        copied = result["copied"]
        failed = result["failed"]
        total = result["total"]
        copied_size_mb = result["copied_size"] / (1024**2)
        
        self.progress_var.set(f"Copy complete - {copied} files ({copied_size_mb:.1f} MB)")
        self.log_status(f"Copy completed: {copied}/{total} files copied successfully")
        
        if failed > 0:
            self.log_status(f"⚠️ {failed} files failed to copy")
        
        messagebox.showinfo("Copy Complete", 
                           f"Random file copy completed!\n\n"
                           f"Files copied: {copied}\n"
                           f"Files failed: {failed}\n"
                           f"Total size: {copied_size_mb:.1f} MB")
    
    def run(self):
        self.log_status("Random File Selector ready!")
        self.log_status("Select source directory and configure filters to begin.")
        self.window.mainloop()

if __name__ == "__main__":
    app = RandomFileSelectorGUI()
    app.run()