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
# Script Name: Video-Image-Sequence-Random.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Random frame extraction tool that creates image sequences from videos    
#              using FFmpeg. Extracts 10 random frames per video at high quality        
#              with timestamp-based naming and organized output directories.                                               
#
# Usage: python3 Video-Image-Sequence-Random.py
#
# Dependencies: ffmpeg, tkinter, random                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Supports all major video formats (MP4, AVI, MOV, MKV, M4V, MPG, MPEG, WMV),       
#        generates frames at 1920px width, creates organized subdirectories per video,   
#        uses precision timestamp naming, and includes video duration detection.                                                    
#                                                                                
####################################################################################

"""
Random Image Sequence Generator

Random frame extraction tool that creates high-quality image sequences from
videos using intelligent timestamp distribution and organized output structure.
"""

import os
import random
import subprocess
from tkinter import filedialog, Tk, messagebox
from datetime import datetime
from typing import List, Optional, Tuple

def setup_gui() -> Tk:
    """Initialize and hide the Tkinter root window."""
    root = Tk()
    root.withdraw()
    return root

def select_directory(title: str) -> str:
    """Present directory selection dialog."""
    directory = filedialog.askdirectory(title=title)
    if not directory:
        raise ValueError("No directory selected")
    return directory

def get_video_files(directory: str) -> List[str]:
    """Collect all video files from directory."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv')
    video_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files

def get_video_duration(file_path: str) -> Optional[float]:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None

def generate_random_timestamps(duration: float, count: int) -> List[float]:
    """Generate random timestamps within video duration."""
    if duration <= 0:
        return []
    
    timestamps = []
    for _ in range(count):
        timestamp = random.uniform(0, duration)
        timestamps.append(timestamp)
    
    return sorted(timestamps)

def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract single frame from video at specified timestamp."""
    try:
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            '-vf', 'scale=1920:-1',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
        
    except subprocess.CalledProcessError:
        return False

def process_video(video_path: str, output_dir: str, frame_count: int = 10) -> Tuple[int, int]:
    """Process single video file."""
    try:
        # Get video duration
        duration = get_video_duration(video_path)
        if not duration:
            return 0, 1
        
        # Generate random timestamps
        timestamps = generate_random_timestamps(duration, frame_count)
        
        # Create output subdirectory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sequence_dir = os.path.join(output_dir, video_name)
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Extract frames
        success_count = 0
        for i, timestamp in enumerate(timestamps):
            output_path = os.path.join(
                sequence_dir,
                f"frame_{i:03d}_{timestamp:.3f}.jpg"
            )
            
            if extract_frame(video_path, timestamp, output_path):
                success_count += 1
        
        return success_count, frame_count - success_count
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0, frame_count

def main():
    """Main execution flow."""
    try:
        root = setup_gui()
        
        # Select directories
        source_dir = select_directory("Select Video Source Directory")
        output_dir = select_directory("Select Output Directory for Sequences")
        
        # Get video files
        video_files = get_video_files(source_dir)
        if not video_files:
            messagebox.showwarning("No Files", "No video files found in selected directory.")
            return
        
        # Process files
        total_success = 0
        total_errors = 0
        total_files = len(video_files)
        
        for i, video_path in enumerate(video_files, 1):
            print(f"Processing {i}/{total_files}: {os.path.basename(video_path)}")
            success, errors = process_video(video_path, output_dir)
            total_success += success
            total_errors += errors
        
        # Show completion message
        message = (
            f"Processing complete!\n\n"
            f"Total videos: {total_files}\n"
            f"Frames extracted: {total_success}\n"
            f"Failed frames: {total_errors}"
        )
        messagebox.showinfo("Complete", message)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
