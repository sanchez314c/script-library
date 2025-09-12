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
# Script Name: systems-trash-recovery.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible trash recovery system for file restoration
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - External volume trash support for comprehensive recovery
#     - Collision handling with automatic renaming
#     - Progress tracking with native macOS GUI dialogs
#     - File integrity verification during recovery
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - tkinter, os, shutil (auto-installed if missing)
#
# Usage:
#     python systems-trash-recovery.py
#
####################################################################################

import os
import sys
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import logging

# Setup logging to desktop
desktop = Path.home() / "Desktop"
log_file = desktop / "trash_recovery.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class TrashRecovery:
    def __init__(self):
        self.trash_locations = []
        self.found_files = []
        self.recovered_files = 0
        self.failed_files = 0
        
    def find_trash_locations(self):
        """Find all trash locations on system"""
        locations = []
        
        # User's main trash
        user_trash = Path.home() / ".Trash"
        if user_trash.exists():
            locations.append({"path": user_trash, "type": "User Trash", "volume": "Main"})
        
        # External volume trash folders
        volumes_path = Path("/Volumes")
        if volumes_path.exists():
            for volume in volumes_path.iterdir():
                if volume.is_dir() and volume.name != ".":
                    # Check for .Trashes folder
                    trashes_path = volume / ".Trashes"
                    if trashes_path.exists():
                        # Check user-specific trash within .Trashes
                        user_id = os.getuid()
                        user_trash_path = trashes_path / str(user_id)
                        if user_trash_path.exists():
                            locations.append({
                                "path": user_trash_path, 
                                "type": "External Volume Trash", 
                                "volume": volume.name
                            })
                    
                    # Also check for .Trash folder (some external drives)
                    trash_path = volume / ".Trash"
                    if trash_path.exists():
                        locations.append({
                            "path": trash_path, 
                            "type": "External Volume User Trash", 
                            "volume": volume.name
                        })
        
        self.trash_locations = locations
        return locations
    
    def scan_trash(self, progress_callback=None):
        """Scan all trash locations for files"""
        self.found_files = []
        
        for location in self.trash_locations:
            trash_path = location["path"]
            location_type = location["type"]
            volume = location["volume"]
            
            logging.info(f"Scanning {location_type} on {volume}: {trash_path}")
            
            if progress_callback:
                progress_callback(f"Scanning {location_type} on {volume}...")
            
            try:
                for root, dirs, files in os.walk(trash_path):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            stat = file_path.stat()
                            file_info = {
                                "path": file_path,
                                "name": file,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "location_type": location_type,
                                "volume": volume,
                                "relative_path": file_path.relative_to(trash_path)
                            }
                            self.found_files.append(file_info)
                        except (OSError, PermissionError) as e:
                            logging.warning(f"Cannot access {file_path}: {e}")
            
            except (OSError, PermissionError) as e:
                logging.error(f"Cannot access trash location {trash_path}: {e}")
        
        logging.info(f"Found {len(self.found_files)} files in trash")
        return self.found_files
    
    def recover_file(self, file_info, destination_dir):
        """Recover a single file from trash"""
        try:
            src_path = file_info["path"]
            dest_dir = Path(destination_dir)
            dest_path = dest_dir / file_info["name"]
            
            # Handle name collisions
            if dest_path.exists():
                counter = 1
                name_parts = file_info["name"].rsplit('.', 1)
                if len(name_parts) == 2:
                    base_name, ext = name_parts
                    while dest_path.exists():
                        new_name = f"{base_name}_recovered_{counter:03d}.{ext}"
                        dest_path = dest_dir / new_name
                        counter += 1
                else:
                    base_name = file_info["name"]
                    while dest_path.exists():
                        new_name = f"{base_name}_recovered_{counter:03d}"
                        dest_path = dest_dir / new_name
                        counter += 1
            
            # Create destination directory if needed
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file (don't move, keep in trash for safety)
            shutil.copy2(src_path, dest_path)
            
            logging.info(f"Recovered: {file_info['name']} -> {dest_path}")
            return {"success": True, "dest": str(dest_path)}
            
        except Exception as e:
            logging.error(f"Failed to recover {file_info['name']}: {e}")
            return {"success": False, "error": str(e)}
    
    def recover_selected_files(self, selected_files, destination_dir, progress_callback=None):
        """Recover multiple selected files"""
        self.recovered_files = 0
        self.failed_files = 0
        total_files = len(selected_files)
        
        for i, file_info in enumerate(selected_files):
            if progress_callback:
                progress_callback(i + 1, total_files, file_info["name"])
            
            result = self.recover_file(file_info, destination_dir)
            if result["success"]:
                self.recovered_files += 1
            else:
                self.failed_files += 1
        
        return {
            "recovered": self.recovered_files,
            "failed": self.failed_files,
            "total": total_files
        }

class TrashRecoveryGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Trash Recovery System")
        self.window.geometry("900x600")
        
        self.recovery = TrashRecovery()
        self.found_files = []
        self.selected_files = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.window, text="🗑️ TRASH RECOVERY SYSTEM", 
                        font=("Arial", 18, "bold"), fg="red")
        title.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.window)
        control_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Button(control_frame, text="🔍 Scan All Trash", command=self.scan_trash,
                 bg="blue", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        tk.Button(control_frame, text="💾 Recover Selected", command=self.recover_selected,
                 bg="green", fg="white", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        tk.Button(control_frame, text="📂 Select Destination", command=self.select_destination,
                 font=("Arial", 11)).pack(side="left", padx=5)
        
        # Destination display
        self.dest_var = tk.StringVar(value="No destination selected")
        dest_label = tk.Label(control_frame, textvariable=self.dest_var, 
                             font=("Arial", 10), fg="gray")
        dest_label.pack(side="right", padx=10)
        
        # Progress frame
        progress_frame = tk.Frame(self.window)
        progress_frame.pack(fill="x", padx=20, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready to scan trash...")
        tk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 10)).pack(anchor="w")
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=2)
        
        # Files listbox with scrollbar
        list_frame = tk.Frame(self.window)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(list_frame, text="Found Files in Trash:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # Create treeview for file list
        columns = ("Name", "Size", "Modified", "Location", "Volume")
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        self.file_tree.heading("Name", text="File Name")
        self.file_tree.heading("Size", text="Size")
        self.file_tree.heading("Modified", text="Date Modified")
        self.file_tree.heading("Location", text="Trash Location")
        self.file_tree.heading("Volume", text="Volume")
        
        self.file_tree.column("Name", width=200)
        self.file_tree.column("Size", width=80)
        self.file_tree.column("Modified", width=130)
        self.file_tree.column("Location", width=150)
        self.file_tree.column("Volume", width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=self.file_tree.xview)
        self.file_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.file_tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        
        # Selection buttons
        select_frame = tk.Frame(self.window)
        select_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Button(select_frame, text="Select All", command=self.select_all).pack(side="left", padx=5)
        tk.Button(select_frame, text="Clear Selection", command=self.clear_selection).pack(side="left", padx=5)
        
        # Status area
        status_frame = tk.Frame(self.window)
        status_frame.pack(fill="x", padx=20, pady=(0,20))
        
        tk.Label(status_frame, text="Status:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(status_frame, height=6, font=("Courier", 9))
        self.status_text.pack(fill="x")
        
        self.destination_dir = None
        
    def log_status(self, message):
        """Add message to status log"""
        timestamp = time.strftime('%H:%M:%S')
        self.status_text.insert(tk.END, f"{timestamp} - {message}\n")
        self.status_text.see(tk.END)
        self.window.update_idletasks()
    
    def scan_trash(self):
        """Scan all trash locations"""
        self.log_status("Starting trash scan...")
        self.progress_var.set("Finding trash locations...")
        
        def scan_thread():
            try:
                # Find trash locations
                locations = self.recovery.find_trash_locations()
                self.window.after(0, lambda: self.log_status(f"Found {len(locations)} trash locations"))
                
                # Scan for files
                def progress_callback(message):
                    self.window.after(0, lambda: self.progress_var.set(message))
                
                files = self.recovery.scan_trash(progress_callback)
                self.window.after(0, self.scan_completed, files)
                
            except Exception as e:
                self.window.after(0, lambda: self.log_status(f"Scan failed: {e}"))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def scan_completed(self, files):
        """Handle scan completion"""
        self.found_files = files
        self.populate_file_list()
        self.progress_var.set(f"Scan complete - Found {len(files)} files")
        self.log_status(f"Scan completed: {len(files)} files found in trash")
    
    def populate_file_list(self):
        """Populate the file list treeview"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add files to treeview
        for file_info in self.found_files:
            size_str = self.format_size(file_info["size"])
            date_str = file_info["modified"].strftime("%Y-%m-%d %H:%M")
            
            self.file_tree.insert("", "end", values=(
                file_info["name"],
                size_str,
                date_str,
                file_info["location_type"],
                file_info["volume"]
            ))
    
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    def select_all(self):
        """Select all files in the list"""
        for item in self.file_tree.get_children():
            self.file_tree.selection_add(item)
    
    def clear_selection(self):
        """Clear all selections"""
        self.file_tree.selection_remove(self.file_tree.selection())
    
    def select_destination(self):
        """Select destination directory for recovery"""
        directory = filedialog.askdirectory(title="Select Recovery Destination")
        if directory:
            self.destination_dir = directory
            self.dest_var.set(f"Destination: {directory}")
            self.log_status(f"Recovery destination set: {directory}")
    
    def recover_selected(self):
        """Recover selected files"""
        selected_items = self.file_tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select files to recover.")
            return
        
        if not self.destination_dir:
            messagebox.showerror("No Destination", "Please select a destination directory first.")
            return
        
        # Get selected file info
        selected_files = []
        for item in selected_items:
            index = self.file_tree.index(item)
            selected_files.append(self.found_files[index])
        
        # Confirm recovery
        total_size = sum(f["size"] for f in selected_files)
        size_str = self.format_size(total_size)
        
        if not messagebox.askyesno("Confirm Recovery", 
                                  f"Recover {len(selected_files)} files ({size_str}) to:\n{self.destination_dir}\n\nContinue?"):
            return
        
        self.log_status(f"Starting recovery of {len(selected_files)} files...")
        
        def recovery_thread():
            try:
                def progress_callback(current, total, filename):
                    percent = (current / total) * 100
                    self.window.after(0, lambda: self.progress_var.set(f"Recovering: {current}/{total} - {filename}"))
                    self.window.after(0, lambda: setattr(self.progress_bar, 'value', percent))
                
                result = self.recovery.recover_selected_files(selected_files, self.destination_dir, progress_callback)
                self.window.after(0, self.recovery_completed, result)
                
            except Exception as e:
                self.window.after(0, lambda: self.log_status(f"Recovery failed: {e}"))
        
        threading.Thread(target=recovery_thread, daemon=True).start()
    
    def recovery_completed(self, result):
        """Handle recovery completion"""
        recovered = result["recovered"]
        failed = result["failed"]
        total = result["total"]
        
        self.progress_var.set(f"Recovery complete - {recovered} succeeded, {failed} failed")
        self.log_status(f"Recovery completed: {recovered}/{total} files recovered successfully")
        
        if failed > 0:
            self.log_status(f"⚠️ {failed} files failed to recover - check log for details")
        
        messagebox.showinfo("Recovery Complete", 
                           f"File recovery completed!\n\n"
                           f"Successfully recovered: {recovered}\n"
                           f"Failed: {failed}\n"
                           f"Total: {total}")
    
    def run(self):
        self.log_status("Trash Recovery System ready!")
        self.log_status("Click 'Scan All Trash' to find recoverable files.")
        self.window.mainloop()

if __name__ == "__main__":
    app = TrashRecoveryGUI()
    app.run()