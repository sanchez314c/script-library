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
# Script Name: systems-sort-videos-by-dimensions.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible video sorting system by resolution dimensions
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Video dimension analysis with ffprobe integration
#     - Automatic sorting by resolution categories (4K, 1080p, 720p, etc.)
#     - Progress tracking with native macOS GUI dialogs
#     - Multi-threaded video processing for efficiency
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - ffprobe (auto-installed via ffmpeg if missing)
#     - Standard library with video processing capabilities
#
# Usage:
#     python systems-sort-videos-by-dimensions.py
#
####################################################################################

import os
import sys
import subprocess
import shutil
import json
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import logging

# Auto-install ffmpeg if needed
def install_ffmpeg():
    try:
        # Check if brew is available
        subprocess.run(['brew', '--version'], capture_output=True, check=True)
        print("Installing ffmpeg via Homebrew...")
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        return True
    except:
        print("Homebrew not available or ffmpeg installation failed")
        return False

# Check for ffprobe availability
def check_ffprobe():
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except FileNotFoundError:
        print("ffprobe not found. Installing ffmpeg...")
        return install_ffmpeg()

# Setup logging to desktop
desktop = Path.home() / "Desktop"
log_file = desktop / "video_sorter.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class VideoSorter:
    def __init__(self):
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp'}
        self.categories = {
            '4K': {'min_width': 3840, 'min_height': 2160},
            '1440p': {'min_width': 2560, 'min_height': 1440},
            '1080p': {'min_width': 1920, 'min_height': 1080},
            '720p': {'min_width': 1280, 'min_height': 720},
            '480p': {'min_width': 640, 'min_height': 480},
            'Other': {'min_width': 0, 'min_height': 0}
        }
        self.total_videos = 0
        self.processed_videos = 0
        self.failed_videos = 0
        
    def get_video_dimensions(self, video_path):
        """Get video dimensions using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'v:0', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [])
                
                if streams:
                    video_stream = streams[0]
                    width = video_stream.get('width')
                    height = video_stream.get('height')
                    
                    if width and height:
                        return int(width), int(height)
            
            return None, None
            
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to get dimensions for {video_path}: {e}")
            return None, None
    
    def categorize_video(self, width, height):
        """Categorize video based on dimensions"""
        if not width or not height:
            return 'Other'
        
        # Check categories from highest to lowest resolution
        for category, specs in self.categories.items():
            if category == 'Other':
                continue
            if width >= specs['min_width'] and height >= specs['min_height']:
                return category
        
        return 'Other'
    
    def scan_videos(self, source_dir, progress_callback=None):
        """Scan directory for video files"""
        videos = []
        
        logging.info(f"Scanning for videos in: {source_dir}")
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.video_extensions:
                    videos.append(file_path)
                    
                    if progress_callback:
                        progress_callback(f"Found: {file}")
        
        self.total_videos = len(videos)
        logging.info(f"Found {len(videos)} video files")
        return videos
    
    def sort_videos(self, videos, destination_dir, move_files=False, progress_callback=None):
        """Sort videos by dimensions"""
        if not videos:
            return {"success": False, "error": "No videos to sort"}
        
        dest_path = Path(destination_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Create category directories
        category_dirs = {}
        for category in self.categories.keys():
            category_dir = dest_path / category
            category_dir.mkdir(exist_ok=True)
            category_dirs[category] = category_dir
        
        sorted_count = {category: 0 for category in self.categories.keys()}
        self.processed_videos = 0
        self.failed_videos = 0
        
        def process_video(video_path):
            try:
                # Get video dimensions
                width, height = self.get_video_dimensions(video_path)
                
                # Categorize video
                category = self.categorize_video(width, height)
                
                # Determine destination
                dest_file = category_dirs[category] / video_path.name
                
                # Handle name collisions
                if dest_file.exists():
                    counter = 1
                    stem = video_path.stem
                    suffix = video_path.suffix
                    while dest_file.exists():
                        new_name = f"{stem}_{counter:03d}{suffix}"
                        dest_file = category_dirs[category] / new_name
                        counter += 1
                
                # Move or copy file
                if move_files:
                    shutil.move(str(video_path), str(dest_file))
                    action = "Moved"
                else:
                    shutil.copy2(str(video_path), str(dest_file))
                    action = "Copied"
                
                dimensions_str = f"{width}x{height}" if width and height else "Unknown"
                logging.info(f"{action}: {video_path.name} -> {category} ({dimensions_str})")
                
                return {
                    "success": True,
                    "category": category,
                    "dimensions": (width, height),
                    "dest": str(dest_file)
                }
                
            except Exception as e:
                logging.error(f"Failed to process {video_path}: {e}")
                return {"success": False, "error": str(e), "file": str(video_path)}
        
        # Process videos with threading
        max_workers = min(8, max(2, os.cpu_count() // 2))  # Conservative for video processing
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(process_video, video): video 
                for video in videos
            }
            
            # Process results
            for future in as_completed(future_to_video):
                result = future.result()
                
                if result["success"]:
                    category = result["category"]
                    sorted_count[category] += 1
                    self.processed_videos += 1
                else:
                    self.failed_videos += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(self.processed_videos, self.total_videos, sorted_count)
        
        return {
            "success": True,
            "processed": self.processed_videos,
            "failed": self.failed_videos,
            "total": self.total_videos,
            "categories": sorted_count
        }

class VideoSorterGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Video Sorter by Dimensions")
        self.window.geometry("700x600")
        
        self.sorter = VideoSorter()
        self.source_dir = None
        self.dest_dir = None
        self.videos = []
        
        # Check ffprobe availability
        if not check_ffprobe():
            messagebox.showerror("Missing Dependency", 
                               "ffprobe is required but not installed. Please install ffmpeg and try again.")
            sys.exit(1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.window, text="🎬 VIDEO SORTER BY DIMENSIONS", 
                        font=("Arial", 16, "bold"), fg="navy")
        title.pack(pady=10)
        
        # Source directory
        source_frame = tk.Frame(self.window)
        source_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(source_frame, text="Source Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.source_var = tk.StringVar()
        
        src_entry_frame = tk.Frame(source_frame)
        src_entry_frame.pack(fill="x", pady=2)
        tk.Entry(src_entry_frame, textvariable=self.source_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(src_entry_frame, text="Browse", command=self.browse_source).pack(side="right", padx=(5,0))
        
        # Destination directory
        dest_frame = tk.Frame(self.window)
        dest_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(dest_frame, text="Destination Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.dest_var = tk.StringVar()
        
        dest_entry_frame = tk.Frame(dest_frame)
        dest_entry_frame.pack(fill="x", pady=2)
        tk.Entry(dest_entry_frame, textvariable=self.dest_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(dest_entry_frame, text="Browse", command=self.browse_dest).pack(side="right", padx=(5,0))
        
        # Options
        options_frame = tk.LabelFrame(self.window, text="Options", font=("Arial", 11, "bold"))
        options_frame.pack(fill="x", padx=20, pady=10)
        
        self.move_files_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Move files (instead of copy)", 
                      variable=self.move_files_var, font=("Arial", 10)).pack(anchor="w", padx=10, pady=5)
        
        # Resolution categories info
        info_frame = tk.LabelFrame(self.window, text="Resolution Categories", font=("Arial", 11, "bold"))
        info_frame.pack(fill="x", padx=20, pady=10)
        
        info_text = tk.Text(info_frame, height=4, font=("Arial", 9))
        info_text.pack(fill="x", padx=10, pady=5)
        info_text.insert(tk.END, "Videos will be sorted into these categories:\n")
        info_text.insert(tk.END, "• 4K: 3840x2160 and above\n")
        info_text.insert(tk.END, "• 1440p: 2560x1440 to 3839x2159\n")
        info_text.insert(tk.END, "• 1080p: 1920x1080 to 2559x1439\n")
        info_text.insert(tk.END, "• 720p: 1280x720 to 1919x1079\n")
        info_text.insert(tk.END, "• 480p: 640x480 to 1279x719\n")
        info_text.insert(tk.END, "• Other: Below 640x480 or unreadable files")
        info_text.config(state="disabled")
        
        # Control buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="🔍 Scan Videos", command=self.scan_videos,
                 bg="blue", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        tk.Button(button_frame, text="📁 Sort Videos", command=self.sort_videos,
                 bg="green", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        # Progress frame
        progress_frame = tk.Frame(self.window)
        progress_frame.pack(fill="x", padx=20, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready to scan videos...")
        tk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 10)).pack(anchor="w")
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=2)
        
        # Status log
        status_frame = tk.Frame(self.window)
        status_frame.pack(fill="both", expand=True, padx=20, pady=(5,20))
        
        tk.Label(status_frame, text="Processing Log:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, font=("Courier", 9))
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
    
    def scan_videos(self):
        """Scan source directory for videos"""
        if not self.source_dir:
            messagebox.showerror("Error", "Please select a source directory first.")
            return
        
        self.log_status("Starting video scan...")
        self.progress_var.set("Scanning for video files...")
        
        def scan_thread():
            try:
                def progress_callback(message):
                    self.window.after(0, lambda: self.progress_var.set(message))
                
                videos = self.sorter.scan_videos(self.source_dir, progress_callback)
                self.window.after(0, self.scan_completed, videos)
                
            except Exception as e:
                self.window.after(0, lambda: self.log_status(f"Scan failed: {e}"))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def scan_completed(self, videos):
        """Handle scan completion"""
        self.videos = videos
        self.progress_var.set(f"Scan complete - Found {len(videos)} video files")
        self.log_status(f"Found {len(videos)} video files for sorting")
    
    def sort_videos(self):
        """Sort videos by dimensions"""
        if not self.videos:
            messagebox.showerror("Error", "Please scan for videos first.")
            return
        
        if not self.dest_dir:
            messagebox.showerror("Error", "Please select a destination directory.")
            return
        
        # Confirm operation
        action = "move" if self.move_files_var.get() else "copy"
        if not messagebox.askyesno("Confirm Sort", 
                                  f"This will {action} {len(self.videos)} video files to categorized folders.\n\nContinue?"):
            return
        
        self.log_status(f"Starting video sorting ({action} mode)...")
        
        def sort_thread():
            try:
                def progress_callback(current, total, categories):
                    percent = (current / total) * 100 if total > 0 else 0
                    self.window.after(0, lambda: self.progress_var.set(f"Processing: {current}/{total} videos"))
                    self.window.after(0, lambda: setattr(self.progress_bar, 'value', percent))
                
                result = self.sorter.sort_videos(self.videos, self.dest_dir, 
                                               self.move_files_var.get(), progress_callback)
                self.window.after(0, self.sort_completed, result)
                
            except Exception as e:
                self.window.after(0, lambda: self.log_status(f"Sort failed: {e}"))
        
        threading.Thread(target=sort_thread, daemon=True).start()
    
    def sort_completed(self, result):
        """Handle sort completion"""
        processed = result["processed"]
        failed = result["failed"]
        total = result["total"]
        categories = result["categories"]
        
        self.progress_var.set(f"Sort complete - {processed} processed, {failed} failed")
        self.log_status(f"Video sorting completed: {processed}/{total} videos processed")
        
        # Log category breakdown
        for category, count in categories.items():
            if count > 0:
                self.log_status(f"  {category}: {count} videos")
        
        if failed > 0:
            self.log_status(f"⚠️ {failed} videos failed to process")
        
        messagebox.showinfo("Sort Complete", 
                           f"Video sorting completed!\n\n"
                           f"Videos processed: {processed}\n"
                           f"Videos failed: {failed}\n"
                           f"Total: {total}\n\n"
                           f"Check the destination folders for sorted videos.")
    
    def run(self):
        self.log_status("Video Sorter ready!")
        self.log_status("Select source and destination directories to begin.")
        self.window.mainloop()

if __name__ == "__main__":
    app = VideoSorterGUI()
    app.run()