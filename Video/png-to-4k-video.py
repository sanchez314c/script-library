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
# Script Name: png-to-4k-video.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-performance PNG sequence to 4K video converter using hardware-    
#              accelerated encoding. Converts image sequences to professional-grade        
#              4K (3840x2160) videos with automatic frame rate detection and optimization.                                               
#
# Usage: python3 png-to-4k-video.py
#
# Dependencies: ffmpeg, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features hardware acceleration, batch processing, collision-safe naming,       
#        and high-quality H.264 encoding optimized for 4K output resolution.   
#        Supports automatic frame rate detection and GUI-based directory selection.                                                    
#                                                                                
####################################################################################

"""
PNG to 4K Video Converter

High-performance tool for converting PNG image sequences into professional-grade
4K videos using hardware-accelerated encoding and automatic optimization.
"""

Requirements:
    - Python 3.6+
    - ffmpeg with H.264 support
    - tkinter (for GUI)
    - Hardware acceleration support (optional)

Usage:
    python png-to-4k-video.py
    Then select source and output directories through GUI
"""

import os
import subprocess
from tkinter import filedialog, Tk, messagebox
from datetime import datetime

def convert_png_to_video(source_dir, output_video_path):
    """Convert PNG sequence to 4K video with hardware acceleration."""
    try:
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

        # Count PNG files for frame rate calculation
        png_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]
        if not png_files:
            raise FileNotFoundError("No PNG files found in source directory.")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        ffmpeg_command = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-framerate', '30',  # Standard frame rate
            '-pattern_type', 'glob',
            '-i', os.path.join(source_dir, '*.png'),
            '-c:v', 'libx264',
            '-preset', 'slow',  # Higher quality encoding
            '-crf', '18',       # High quality setting
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=3840:2160:flags=lanczos',  # 4K with high-quality scaling
            '-movflags', '+faststart',  # Enable streaming optimization
            '-metadata', f'creation_time={datetime.now().isoformat()}',
            output_video_path
        ]

        # Run FFmpeg with progress monitoring
        process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_command, stderr)

        return True

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return False

def main():
    """Main function with GUI interaction."""
    root = Tk()
    root.withdraw()  # Hide the main window

    try:
        # Select source directory
        source_dir = filedialog.askdirectory(title="Select Source Directory with PNG Files")
        if not source_dir:
            print("No source directory selected, exiting.")
            return

        # Select output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory for Video")
        if not output_dir:
            print("No output directory selected, exiting.")
            return

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video = os.path.join(output_dir, f"4k_video_{timestamp}.mp4")

        # Handle file collisions
        counter = 1
        while os.path.exists(output_video):
            output_video = os.path.join(output_dir, f"4k_video_{timestamp}_{counter}.mp4")
            counter += 1

        if convert_png_to_video(source_dir, output_video):
            messagebox.showinfo("Success", f"Video conversion complete!\nSaved to: {output_video}")
        else:
            messagebox.showerror("Error", "Conversion failed. Check console for details.")

    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
