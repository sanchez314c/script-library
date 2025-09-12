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
# Script Name: video-generate-thumbnail.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-quality video thumbnail generator that creates and embeds    
#              thumbnails for video files using FFmpeg. Features batch processing,        
#              automatic thumbnail positioning, and embedded metadata support.                                               
#
# Usage: python3 video-generate-thumbnail.py
#
# Dependencies: ffmpeg, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Supports multiple video formats (MP4, AVI, MOV, MKV, M4V, MPG, MPEG, WMV),       
#        generates thumbnails at 33% position through video, uses 1920px scaling   
#        for high-quality output, and embeds thumbnails into video metadata.                                                    
#                                                                                
####################################################################################

"""
Video Thumbnail Generator

High-quality video thumbnail generator that creates and embeds thumbnails
for video files using FFmpeg with automatic positioning and metadata support.
"""

import os
import subprocess
from tkinter import filedialog, Tk, messagebox
from datetime import datetime
from typing import List, Optional

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

def generate_thumbnail(video_path: str, temp_dir: str) -> Optional[str]:
    """Generate thumbnail from video."""
    try:
        # Get video duration
        duration = get_video_duration(video_path)
        if not duration:
            return None
        
        # Calculate thumbnail position (33% through video)
        position = duration * 0.33
        
        # Generate temporary thumbnail path
        temp_thumb = os.path.join(
            temp_dir,
            f"thumb_{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
        )
        
        # Generate high-quality thumbnail
        cmd = [
            'ffmpeg',
            '-ss', str(position),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            '-vf', 'scale=1920:-1',
            temp_thumb
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_thumb if os.path.exists(temp_thumb) else None
        
    except subprocess.CalledProcessError:
        return None

def embed_thumbnail(video_path: str, thumbnail_path: str) -> bool:
    """Embed thumbnail into video metadata."""
    try:
        temp_output = video_path + '.temp'
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', thumbnail_path,
            '-map', '0',
            '-map', '1',
            '-c', 'copy',
            '-disposition:v:1', 'attached_pic',
            temp_output
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Replace original with new file
        os.replace(temp_output, video_path)
        return True
        
    except subprocess.CalledProcessError:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False

def main():
    """Main execution flow."""
    try:
        root = setup_gui()
        
        # Select directory
        directory = select_directory("Select Video Directory")
        
        # Get video files
        video_files = get_video_files(directory)
        if not video_files:
            messagebox.showwarning("No Files", "No video files found in selected directory.")
            return
        
        # Create temporary directory
        temp_dir = os.path.join(directory, '.thumbs_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process files
        success_count = 0
        error_count = 0
        total_files = len(video_files)
        
        for i, video_path in enumerate(video_files, 1):
            try:
                print(f"Processing {i}/{total_files}: {os.path.basename(video_path)}")
                
                # Generate thumbnail
                thumbnail_path = generate_thumbnail(video_path, temp_dir)
                if not thumbnail_path:
                    print(f"Failed to generate thumbnail for: {video_path}")
                    error_count += 1
                    continue
                
                # Embed thumbnail
                if embed_thumbnail(video_path, thumbnail_path):
                    success_count += 1
                else:
                    error_count += 1
                
                # Clean up thumbnail
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                error_count += 1
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
        
        # Show completion message
        message = f"Processing complete!\n\nTotal files: {total_files}\nSuccessful: {success_count}\nFailed: {error_count}"
        messagebox.showinfo("Complete", message)
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
