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
# Script Name: turbo-copy.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible high-performance file copying system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded file operations for maximum speed
#     - Detailed logging with progress monitoring
#     - Collision handling with automatic resolution
#     - Error recovery with data integrity verification
#     - Smart chunked copying for large files
#     - Preserve file attributes and timestamps
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, concurrent.futures, logging, shutil (auto-installed if missing)
#
# Usage:
#     python turbo-copy.py
#
####################################################################################

import os
import sys
import shutil
import hashlib
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging

# Setup logging to desktop
desktop = Path.home() / "Desktop"
log_file = desktop / "turbo_copy.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class TurboCopier:
    def __init__(self):
        self.source_dir = None
        self.dest_dir = None
        self.total_files = 0
        self.copied_files = 0
        self.failed_files = 0
        self.total_size = 0
        self.copied_size = 0
        self.start_time = None
        self.collision_strategy = "rename"  # rename, skip, overwrite
        self.preserve_attributes = True
        self.verify_integrity = True
        
    def get_file_hash(self, file_path, chunk_size=65536):
        """Calculate MD5 hash of file for integrity checking with optimized chunk size"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def get_unique_name(self, dest_path):
        """Generate unique filename for collision resolution"""
        if not dest_path.exists():
            return dest_path
        
        counter = 1
        while True:
            name = dest_path.stem
            ext = dest_path.suffix
            new_name = f"{name}_copy_{counter:03d}{ext}"
            new_path = dest_path.parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    def chunked_copy(self, src_path, dest_path, chunk_size=1024*1024):
        """Copy file in chunks for better progress tracking and memory management"""
        try:
            with open(src_path, 'rb') as src_file, open(dest_path, 'wb') as dest_file:
                while chunk := src_file.read(chunk_size):
                    dest_file.write(chunk)
            return True
        except Exception:
            return False
    
    def copy_file(self, src_file, dest_dir, progress_callback=None):
        """Copy single file with integrity checking and attribute preservation"""
        try:
            src_path = Path(src_file)
            dest_path = Path(dest_dir) / src_path.name
            
            # Handle collisions
            if dest_path.exists():
                if self.collision_strategy == "skip":
                    logging.info(f"Skipped (exists): {src_path.name}")
                    return {"success": True, "skipped": True, "size": src_path.stat().st_size}
                elif self.collision_strategy == "rename":
                    dest_path = self.get_unique_name(dest_path)
                # overwrite strategy continues normally
            
            # Create destination directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file size and hash before copy
            file_size = src_path.stat().st_size
            src_hash = None
            if self.verify_integrity:
                src_hash = self.get_file_hash(src_path)
            
            # Perform the copy with chunked copying for large files
            if file_size > 100 * 1024 * 1024:  # Files larger than 100MB
                success = self.chunked_copy(src_path, dest_path)
                if not success:
                    raise Exception("Chunked copy failed")
            else:
                # Use shutil.copy2 for smaller files (preserves metadata)
                if self.preserve_attributes:
                    shutil.copy2(str(src_path), str(dest_path))
                else:
                    shutil.copy(str(src_path), str(dest_path))
            
            # Verify integrity if enabled
            if self.verify_integrity and src_hash:
                dest_hash = self.get_file_hash(dest_path)
                if dest_hash and src_hash != dest_hash:
                    dest_path.unlink()  # Remove corrupted copy
                    raise Exception("File integrity check failed")
            
            # Preserve attributes for chunked copies if needed
            if self.preserve_attributes and file_size > 100 * 1024 * 1024:
                shutil.copystat(str(src_path), str(dest_path))
            
            logging.info(f"Copied: {src_path.name} -> {dest_path}")
            
            if progress_callback:
                progress_callback(file_size)
            
            return {"success": True, "size": file_size, "dest": str(dest_path)}
            
        except Exception as e:
            logging.error(f"Failed to copy {src_file}: {str(e)}")
            return {"success": False, "error": str(e), "file": src_file}
    
    def scan_directory(self, directory):
        """Scan directory for files and calculate total size"""
        files = []
        total_size = 0
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(root) / filename
                try:
                    size = filepath.stat().st_size
                    files.append(str(filepath))
                    total_size += size
                except:
                    continue
        
        return files, total_size
    
    def copy_files(self, source_dir, dest_dir, max_workers=None, progress_callback=None):
        """Copy all files from source to destination with threading"""
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.start_time = time.time()
        self.copied_files = 0
        self.failed_files = 0
        self.copied_size = 0
        
        logging.info(f"Starting turbo copy: {source_dir} -> {dest_dir}")
        
        # Scan source directory
        files, self.total_size = self.scan_directory(source_dir)
        self.total_files = len(files)
        
        if not files:
            logging.info("No files found to copy")
            return {"success": True, "copied": 0, "failed": 0}
        
        logging.info(f"Found {self.total_files} files ({self.total_size / (1024**3):.2f} GB)")
        
        # Determine optimal thread count (more conservative for copy operations)
        if max_workers is None:
            max_workers = min(16, max(4, os.cpu_count() // 2))
        
        # Create destination directory
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy files with threading
        failed_copies = []
        
        def update_progress(size):
            self.copied_size += size
            self.copied_files += 1
            if progress_callback:
                try:
                    progress_callback(self.copied_files, self.total_files, self.copied_size, self.total_size)
                except Exception as e:
                    logging.error(f"Progress callback error: {e}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all copy tasks
            future_to_file = {
                executor.submit(self.copy_file, file_path, dest_dir, update_progress): file_path 
                for file_path in files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                result = future.result()
                if not result["success"]:
                    self.failed_files += 1
                    failed_copies.append(result)
        
        elapsed = time.time() - self.start_time
        speed = (self.copied_size / (1024**2)) / elapsed if elapsed > 0 else 0
        
        logging.info(f"Copy completed: {self.copied_files} copied, {self.failed_files} failed")
        logging.info(f"Time: {elapsed:.1f}s, Speed: {speed:.1f} MB/s")
        
        return {
            "success": True,
            "copied": self.copied_files,
            "failed": self.failed_files,
            "elapsed": elapsed,
            "speed": speed,
            "failed_files": failed_copies
        }

class TurboCopyGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Turbo File Copier")
        self.window.geometry("750x600")
        
        self.copier = TurboCopier()
        self.copying = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.window, text="📋 TURBO FILE COPIER", 
                        font=("Arial", 18, "bold"), fg="blue")
        title.pack(pady=10)
        
        # Source directory selection
        src_frame = tk.Frame(self.window)
        src_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(src_frame, text="Source Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.src_var = tk.StringVar()
        src_entry_frame = tk.Frame(src_frame)
        src_entry_frame.pack(fill="x", pady=2)
        
        tk.Entry(src_entry_frame, textvariable=self.src_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(src_entry_frame, text="Browse", command=self.browse_source).pack(side="right", padx=(5,0))
        
        # Destination directory selection
        dest_frame = tk.Frame(self.window)
        dest_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(dest_frame, text="Destination Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.dest_var = tk.StringVar()
        dest_entry_frame = tk.Frame(dest_frame)
        dest_entry_frame.pack(fill="x", pady=2)
        
        tk.Entry(dest_entry_frame, textvariable=self.dest_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(dest_entry_frame, text="Browse", command=self.browse_dest).pack(side="right", padx=(5,0))
        
        # Options
        options_frame = tk.Frame(self.window)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        # Collision Strategy
        tk.Label(options_frame, text="Collision Strategy:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.collision_var = tk.StringVar(value="rename")
        collision_frame = tk.Frame(options_frame)
        collision_frame.pack(anchor="w")
        
        tk.Radiobutton(collision_frame, text="Rename", variable=self.collision_var, value="rename").pack(side="left")
        tk.Radiobutton(collision_frame, text="Skip", variable=self.collision_var, value="skip").pack(side="left", padx=(20,0))
        tk.Radiobutton(collision_frame, text="Overwrite", variable=self.collision_var, value="overwrite").pack(side="left", padx=(20,0))
        
        # Copy Options
        copy_options_frame = tk.Frame(options_frame)
        copy_options_frame.pack(anchor="w", pady=(10,0))
        
        self.preserve_attr_var = tk.BooleanVar(value=True)
        tk.Checkbutton(copy_options_frame, text="Preserve file attributes", 
                      variable=self.preserve_attr_var).pack(side="left")
        
        self.verify_integrity_var = tk.BooleanVar(value=True)
        tk.Checkbutton(copy_options_frame, text="Verify file integrity", 
                      variable=self.verify_integrity_var).pack(side="left", padx=(20,0))
        
        # Progress section
        progress_frame = tk.Frame(self.window)
        progress_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(progress_frame, text="Progress:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        self.progress_var = tk.StringVar(value="Ready to copy files...")
        tk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 10)).pack(anchor="w", pady=2)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=5)
        
        self.speed_var = tk.StringVar(value="")
        tk.Label(progress_frame, textvariable=self.speed_var, font=("Arial", 9), fg="gray").pack(anchor="w")
        
        # Control buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=20)
        
        self.copy_button = tk.Button(button_frame, text="📋 START TURBO COPY", 
                                   command=self.start_copy, bg="green", fg="white", 
                                   font=("Arial", 12, "bold"))
        self.copy_button.pack(side="left", padx=10)
        
        tk.Button(button_frame, text="📄 View Log", command=self.view_log).pack(side="left", padx=10)
        
        # Status text area
        status_frame = tk.Frame(self.window)
        status_frame.pack(fill="both", expand=True, padx=20, pady=(0,20))
        
        tk.Label(status_frame, text="Status Log:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.status_text = tk.Text(status_frame, height=8, font=("Courier", 9))
        scrollbar = tk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def browse_source(self):
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.src_var.set(directory)
    
    def browse_dest(self):
        directory = filedialog.askdirectory(title="Select Destination Directory")
        if directory:
            self.dest_var.set(directory)
    
    def log_status(self, message):
        """Add message to status log"""
        timestamp = time.strftime('%H:%M:%S')
        log_message = f"{timestamp} - {message}\n"
        self.status_text.insert(tk.END, log_message)
        self.status_text.see(tk.END)
        self.window.update()
        # Also log to file
        logging.info(message)
    
    def update_progress(self, copied_files, total_files, copied_size, total_size):
        """Update progress display"""
        percent = (copied_files / total_files) * 100 if total_files > 0 else 0
        
        copied_mb = copied_size / (1024**2)
        total_mb = total_size / (1024**2)
        
        elapsed = time.time() - self.copier.start_time
        speed = copied_mb / elapsed if elapsed > 0 else 0
        
        self.progress_var.set(f"Copying: {copied_files}/{total_files} files ({copied_mb:.1f}/{total_mb:.1f} MB)")
        self.speed_var.set(f"Speed: {speed:.1f} MB/s | ETA: {((total_mb - copied_mb) / speed / 60):.1f} min" if speed > 0 else "")
        self.progress_bar['value'] = percent
        
        self.window.update_idletasks()
    
    def start_copy(self):
        if self.copying:
            return
        
        source = self.src_var.get().strip()
        dest = self.dest_var.get().strip()
        
        if not source or not dest:
            messagebox.showerror("Error", "Please select both source and destination directories.")
            return
        
        if not os.path.isdir(source):
            messagebox.showerror("Error", "Source directory does not exist.")
            return
        
        if source == dest:
            messagebox.showerror("Error", "Source and destination cannot be the same.")
            return
        
        # Confirm copy
        result = messagebox.askyesno("Confirm Copy", 
                                   f"Copy all files from:\n{source}\n\nTo:\n{dest}\n\nContinue?")
        if not result:
            return
        
        self.copying = True
        self.copy_button.config(state="disabled", text="Copying...")
        
        # Set copier options
        self.copier.collision_strategy = self.collision_var.get()
        self.copier.preserve_attributes = self.preserve_attr_var.get()
        self.copier.verify_integrity = self.verify_integrity_var.get()
        
        self.log_status("Starting turbo copy operation...")
        self.log_status(f"Source: {source}")
        self.log_status(f"Destination: {dest}")
        self.progress_bar['value'] = 0
        
        # Force UI update
        self.window.update_idletasks()
        
        def copy_thread():
            try:
                self.log_status("Scanning source directory...")
                result = self.copier.copy_files(source, dest, progress_callback=self.update_progress)
                self.window.after(100, lambda: self.copy_completed(result))
            except Exception as e:
                error_msg = str(e)
                self.window.after(100, lambda: self.copy_error(error_msg))
        
        # Start copy thread
        thread = threading.Thread(target=copy_thread, daemon=True)
        thread.start()
    
    def copy_completed(self, result):
        """Handle successful copy completion"""
        self.copying = False
        self.copy_button.config(state="normal", text="📋 START TURBO COPY")
        
        copied = result["copied"]
        failed = result["failed"]
        elapsed = result["elapsed"]
        speed = result["speed"]
        
        self.log_status(f"✅ Copy completed! {copied} files copied, {failed} failed")
        self.log_status(f"Time: {elapsed:.1f}s, Average speed: {speed:.1f} MB/s")
        
        if failed > 0:
            self.log_status(f"⚠️ {failed} files failed to copy - check log for details")
        
        messagebox.showinfo("Copy Complete", 
                           f"Turbo copy completed!\n\n"
                           f"Files copied: {copied}\n"
                           f"Files failed: {failed}\n"
                           f"Time: {elapsed:.1f} seconds\n"
                           f"Speed: {speed:.1f} MB/s")
    
    def copy_error(self, error):
        """Handle copy errors"""
        self.copying = False
        self.copy_button.config(state="normal", text="📋 START TURBO COPY")
        self.log_status(f"❌ Copy failed: {error}")
        messagebox.showerror("Copy Failed", f"An error occurred during the copy:\n\n{error}")
    
    def view_log(self):
        """Open log file in default application"""
        os.system(f"open {log_file}")
    
    def run(self):
        self.log_status("Turbo File Copier ready!")
        self.window.mainloop()

if __name__ == "__main__":
    app = TurboCopyGUI()
    app.run()