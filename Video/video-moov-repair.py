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
# Script Name: video-moov-repair.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: MP4 MOOV atom repair tool that fixes corrupted video file indexes    
#              using FFmpeg. Repairs seeking issues and playback problems caused        
#              by misplaced or corrupted MOOV atoms with batch processing support.                                               
#
# Usage: python3 video-moov-repair.py
#
# Dependencies: ffmpeg, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Supports MOV, MP4, MPG, AVI, MKV, M4V formats, uses faststart movflags       
#        for streaming optimization, features error detection and ignore modes,   
#        and includes visual progress tracking with multi-threaded processing.                                                    
#                                                                                
####################################################################################

"""
Video MOOV Atom Repair Tool

MP4 MOOV atom repair tool that fixes corrupted video file indexes using FFmpeg
to resolve seeking issues and playback problems with professional-grade repair.
"""

import os
import sys
import subprocess
import time
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("video-moov-repair")

# Define supported video extensions
VIDEO_EXTENSIONS = ('.mov', '.mp4', '.mpg', '.avi', '.mkv', '.m4v')

def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def repair_video(video_info: Tuple[str, str, int, int]) -> Tuple[bool, str]:
    """
    Repair a video file by re-writing the MOOV atom.
    
    Args:
        video_info: Tuple containing (source_file, dest_file, index, total)
        
    Returns:
        Tuple of (success, error_message)
    """
    source_file, dest_file, index, total = video_info
    
    try:
        # Ensure destination directory exists
        Path(os.path.dirname(dest_file)).mkdir(parents=True, exist_ok=True)
        
        # Create filename without directory for display
        source_filename = os.path.basename(source_file)
        
        # Log progress
        logger.info(f"[{index}/{total}] Repairing: {source_filename}")
        
        # Run FFmpeg to repair the video
        cmd = [
            'ffmpeg',
            '-v', 'warning',
            '-stats',
            '-err_detect', 'ignore_err',
            '-i', source_file,
            '-c', 'copy',
            '-movflags', 'faststart',
            dest_file
        ]
        
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"Failed to repair {source_file}: {error_msg}")
        return False, error_msg

def create_progress_dialog(total: int) -> Tuple[tk.Toplevel, ttk.Progressbar, tk.Label, tk.Tk]:
    """Create and return a progress dialog."""
    root = tk.Tk()
    root.withdraw()
    
    window = tk.Toplevel(root)
    window.title("Repairing Videos")
    window.geometry("400x150")
    window.resizable(False, False)
    window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
    
    # Center window
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f"+{x}+{y}")
    
    # Progress elements
    frame = tk.Frame(window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    progress_label = tk.Label(frame, text="Starting video repair...", anchor=tk.W)
    progress_label.pack(fill=tk.X, pady=(0, 10))
    
    progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, 
                                  length=360, mode='determinate', maximum=total)
    progress_bar.pack(fill=tk.X)
    
    return window, progress_bar, progress_label, root

def process_videos_with_progress(video_info_list: List[Tuple[str, str, int, int]]) -> Tuple[int, int]:
    """
    Process videos with a progress dialog.
    
    Args:
        video_info_list: List of (source_file, dest_file, index, total) tuples
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    if not video_info_list:
        return 0, 0
    
    total = len(video_info_list)
    window, progress_bar, progress_label, root = create_progress_dialog(total)
    
    success_count = 0
    failure_count = 0
    
    # Process each video individually for better progress tracking
    for source_file, dest_file, index, total_count in video_info_list:
        # Update progress dialog
        progress_bar['value'] = index
        file_name = os.path.basename(source_file)
        progress_label.config(text=f"Repairing {index} of {total_count}: {file_name}")
        root.update()
        
        # Repair the video
        success, _ = repair_video((source_file, dest_file, index, total_count))
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Close progress dialog
    window.destroy()
    root.destroy()
    
    return success_count, failure_count

def main():
    """Main program execution."""
    # Check for FFmpeg
    if not check_ffmpeg():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "FFmpeg Not Found", 
            "FFmpeg is required but not found. Please install FFmpeg and try again."
        )
        root.destroy()
        sys.exit(1)
    
    # Create root window for dialogs
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Select source folder
        source_folder = filedialog.askdirectory(title='Select the folder containing corrupted video files')
        if not source_folder:
            logger.info("No source folder selected. Exiting.")
            return
        
        # Select destination folder
        dest_folder = filedialog.askdirectory(title='Select the destination folder for repaired videos')
        if not dest_folder:
            logger.info("No destination folder selected. Exiting.")
            return
        
        # Create destination folder if it doesn't exist
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        
        # Find all video files in the source folder
        video_files = []
        for file in os.listdir(source_folder):
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(source_folder, file))
        
        if not video_files:
            messagebox.showinfo("No Videos Found", "No video files found in the selected folder.")
            return
        
        # Prepare file paths for processing
        total = len(video_files)
        video_info_list = []
        
        for i, video_file in enumerate(video_files, 1):
            base_name = os.path.basename(video_file)
            dest_file = os.path.join(dest_folder, f"repaired_{base_name}")
            video_info_list.append((video_file, dest_file, i, total))
        
        # Process videos with progress dialog
        success_count, failure_count = process_videos_with_progress(video_info_list)
        
        # Show completion message
        messagebox.showinfo(
            "Processing Complete",
            f"Video repair complete:\n"
            f"- Total files: {total}\n"
            f"- Successfully repaired: {success_count}\n"
            f"- Failed: {failure_count}\n\n"
            f"Repaired videos saved to:\n{dest_folder}"
        )
        
    except Exception as e:
        logger.exception("An error occurred:")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        root.destroy()

if __name__ == '__main__':
    main()
