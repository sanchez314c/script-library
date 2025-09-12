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
# Script Name: systems-folder-monitor.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible real-time directory monitoring system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Real-time file system monitoring with instant updates
#     - Size calculation and file count tracking
#     - Event logging with detailed change notifications
#     - Native macOS GUI folder selection dialogs
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - watchdog, tkinter, subprocess (auto-installed if missing)
#
# Usage:
#     python systems-folder-monitor.py
#
####################################################################################

import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import logging

# Auto-install dependencies
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# Try importing watchdog, install if missing
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog...")
    install_package("watchdog")
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

# Setup logging to desktop
desktop = Path.home() / "Desktop"
log_file = desktop / "folder_monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class FolderEventHandler(FileSystemEventHandler):
    """Handles file system events"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        super().__init__()
    
    def on_modified(self, event):
        if not event.is_directory:
            self.monitor.log_event("Modified", event.src_path)
    
    def on_created(self, event):
        event_type = "Directory Created" if event.is_directory else "File Created"
        self.monitor.log_event(event_type, event.src_path)
        self.monitor.update_stats()
    
    def on_deleted(self, event):
        event_type = "Directory Deleted" if event.is_directory else "File Deleted"
        self.monitor.log_event(event_type, event.src_path)
        self.monitor.update_stats()
    
    def on_moved(self, event):
        event_type = "Directory Moved" if event.is_directory else "File Moved"
        self.monitor.log_event(event_type, f"{event.src_path} → {event.dest_path}")
        self.monitor.update_stats()

class FolderMonitor:
    def __init__(self):
        self.observer = None
        self.monitoring = False
        self.monitored_path = None
        self.event_count = 0
        self.start_time = None
        self.stats_lock = threading.Lock()
        
    def calculate_folder_stats(self, path):
        """Calculate folder statistics"""
        try:
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(path):
                dir_count += len(dirs)
                for file in files:
                    try:
                        file_path = Path(root) / file
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except (OSError, PermissionError):
                        pass
            
            return {
                "total_size": total_size,
                "file_count": file_count,
                "dir_count": dir_count
            }
        except Exception as e:
            logging.error(f"Error calculating stats: {e}")
            return {"total_size": 0, "file_count": 0, "dir_count": 0}
    
    def start_monitoring(self, path, callback=None):
        """Start monitoring a directory"""
        try:
            if self.monitoring:
                self.stop_monitoring()
            
            self.monitored_path = Path(path)
            self.event_count = 0
            self.start_time = time.time()
            self.callback = callback
            
            # Create event handler and observer
            event_handler = FolderEventHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, str(path), recursive=True)
            
            # Start monitoring
            self.observer.start()
            self.monitoring = True
            
            logging.info(f"Started monitoring: {path}")
            
            # Initial stats calculation
            self.update_stats()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.observer and self.monitoring:
            self.observer.stop()
            self.observer.join()
            self.monitoring = False
            logging.info("Stopped monitoring")
            return True
        return False
    
    def log_event(self, event_type, path):
        """Log a file system event"""
        with self.stats_lock:
            self.event_count += 1
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[{timestamp}] {event_type}: {Path(path).name}"
        
        logging.info(message)
        
        if self.callback:
            self.callback("event", message)
    
    def update_stats(self):
        """Update folder statistics"""
        if not self.monitored_path or not self.monitoring:
            return
        
        def stats_thread():
            try:
                stats = self.calculate_folder_stats(self.monitored_path)
                if self.callback:
                    self.callback("stats", stats)
            except Exception as e:
                logging.error(f"Stats update failed: {e}")
        
        # Run stats calculation in background to avoid blocking
        threading.Thread(target=stats_thread, daemon=True).start()

class FolderMonitorGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Folder Monitor")
        self.window.geometry("800x600")
        
        self.monitor = FolderMonitor()
        self.monitored_path = None
        
        self.setup_ui()
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.window, text="📁 FOLDER MONITOR", 
                        font=("Arial", 18, "bold"), fg="green")
        title.pack(pady=10)
        
        # Path selection frame
        path_frame = tk.Frame(self.window)
        path_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(path_frame, text="Monitor Directory:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        path_entry_frame = tk.Frame(path_frame)
        path_entry_frame.pack(fill="x", pady=5)
        
        self.path_var = tk.StringVar()
        tk.Entry(path_entry_frame, textvariable=self.path_var, font=("Arial", 10)).pack(side="left", fill="x", expand=True)
        tk.Button(path_entry_frame, text="Browse", command=self.browse_directory).pack(side="right", padx=(5,0))
        
        # Control buttons
        control_frame = tk.Frame(self.window)
        control_frame.pack(pady=10)
        
        self.start_button = tk.Button(control_frame, text="▶️ Start Monitoring", 
                                     command=self.start_monitoring, bg="green", fg="white", 
                                     font=("Arial", 11, "bold"))
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = tk.Button(control_frame, text="⏹️ Stop Monitoring", 
                                    command=self.stop_monitoring, bg="red", fg="white", 
                                    font=("Arial", 11, "bold"), state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        tk.Button(control_frame, text="📄 View Log", command=self.view_log).pack(side="left", padx=5)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(self.window, text="Folder Statistics", font=("Arial", 11, "bold"))
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        stats_grid = tk.Frame(stats_frame)
        stats_grid.pack(fill="x", padx=10, pady=10)
        
        # Create stats labels
        tk.Label(stats_grid, text="Files:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,10))
        self.files_var = tk.StringVar(value="0")
        tk.Label(stats_grid, textvariable=self.files_var, font=("Arial", 10)).grid(row=0, column=1, sticky="w", padx=(0,30))
        
        tk.Label(stats_grid, text="Directories:", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky="w", padx=(0,10))
        self.dirs_var = tk.StringVar(value="0")
        tk.Label(stats_grid, textvariable=self.dirs_var, font=("Arial", 10)).grid(row=0, column=3, sticky="w", padx=(0,30))
        
        tk.Label(stats_grid, text="Total Size:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0,10))
        self.size_var = tk.StringVar(value="0 B")
        tk.Label(stats_grid, textvariable=self.size_var, font=("Arial", 10)).grid(row=1, column=1, sticky="w", padx=(0,30))
        
        tk.Label(stats_grid, text="Events:", font=("Arial", 10, "bold")).grid(row=1, column=2, sticky="w", padx=(0,10))
        self.events_var = tk.StringVar(value="0")
        tk.Label(stats_grid, textvariable=self.events_var, font=("Arial", 10)).grid(row=1, column=3, sticky="w")
        
        # Status frame
        status_frame = tk.Frame(self.window)
        status_frame.pack(fill="x", padx=20, pady=5)
        
        self.status_var = tk.StringVar(value="Ready to monitor...")
        tk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10), fg="blue").pack(anchor="w")
        
        # Events log
        events_frame = tk.Frame(self.window)
        events_frame.pack(fill="both", expand=True, padx=20, pady=(0,20))
        
        tk.Label(events_frame, text="Real-time Events:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.events_text = scrolledtext.ScrolledText(events_frame, height=15, font=("Courier", 9))
        self.events_text.pack(fill="both", expand=True)
        
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.2f} GB"
    
    def browse_directory(self):
        """Browse for directory to monitor"""
        directory = filedialog.askdirectory(title="Select Directory to Monitor")
        if directory:
            self.monitored_path = directory
            self.path_var.set(directory)
    
    def monitor_callback(self, event_type, data):
        """Handle callbacks from monitor"""
        if event_type == "event":
            # Add event to log
            self.events_text.insert(tk.END, f"{data}\n")
            self.events_text.see(tk.END)
            
            # Update event count
            self.events_var.set(str(self.monitor.event_count))
            
        elif event_type == "stats":
            # Update statistics
            self.files_var.set(f"{data['file_count']:,}")
            self.dirs_var.set(f"{data['dir_count']:,}")
            self.size_var.set(self.format_size(data['total_size']))
        
        self.window.update_idletasks()
    
    def start_monitoring(self):
        """Start monitoring selected directory"""
        if not self.monitored_path:
            messagebox.showerror("Error", "Please select a directory to monitor first.")
            return
        
        if not os.path.isdir(self.monitored_path):
            messagebox.showerror("Error", "Selected path is not a valid directory.")
            return
        
        # Clear events log
        self.events_text.delete(1.0, tk.END)
        
        # Start monitoring
        success = self.monitor.start_monitoring(self.monitored_path, self.monitor_callback)
        
        if success:
            self.status_var.set(f"Monitoring: {Path(self.monitored_path).name}")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Add initial message
            self.events_text.insert(tk.END, f"Started monitoring: {self.monitored_path}\n")
            self.events_text.insert(tk.END, "Watching for file system changes...\n\n")
        else:
            messagebox.showerror("Error", "Failed to start monitoring. Check the log for details.")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        success = self.monitor.stop_monitoring()
        
        if success:
            self.status_var.set("Monitoring stopped")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            # Add stop message
            elapsed = time.time() - self.monitor.start_time if self.monitor.start_time else 0
            self.events_text.insert(tk.END, f"\nMonitoring stopped after {elapsed:.1f} seconds\n")
            self.events_text.insert(tk.END, f"Total events captured: {self.monitor.event_count}\n")
    
    def view_log(self):
        """Open log file in default application"""
        os.system(f"open {log_file}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
        self.window.destroy()
    
    def run(self):
        """Start the GUI"""
        self.events_text.insert(tk.END, "Folder Monitor ready!\n")
        self.events_text.insert(tk.END, "Select a directory and click 'Start Monitoring' to begin.\n\n")
        self.window.mainloop()

if __name__ == "__main__":
    app = FolderMonitorGUI()
    app.run()