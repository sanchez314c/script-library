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
# Script Name: png-to-dmt-4k.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Psychedelic 4K video generator that creates trippy visual effects    
#              from PNG sequences using frame interpolation and color manipulation.        
#              Features advanced minterpolate filtering and dynamic hue shifting.                                               
#
# Usage: python3 png-to-dmt-4k.py
#
# Dependencies: ffmpeg, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Uses 60 FPS minterpolate with bidirectional motion estimation,       
#        applies color balance and dynamic hue effects, outputs at 4K resolution   
#        with Lanczos scaling, and includes hardware acceleration support.                                                    
#                                                                                
####################################################################################

"""
PNG to Trippy 4K Video Converter

Psychedelic 4K video generator that creates trippy visual effects from PNG
sequences using advanced frame interpolation and dynamic color manipulation.
"""

import os
import subprocess
from tkinter import filedialog, Tk, messagebox
from datetime import datetime

def convert_png_to_video(source_dir, output_video_path):
    """Convert PNG sequence to trippy 4K video with effects."""
    try:
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

        # Count PNG files
        png_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]
        if not png_files:
            raise FileNotFoundError("No PNG files found in source directory.")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Complex filter for trippy effects
        complex_filter = (
            "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
            "colorbalance=rs=0.2:gs=0.2:bs=0.2:rm=0.2:gm=0.2:bm=0.2:rh=0.2:gh=0.2:bh=0.2,"
            "hue=h=0.1*PI*t:s=2,"
            "scale=3840:2160:flags=lanczos"
        )

        ffmpeg_command = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-framerate', '24',
            '-pattern_type', 'glob',
            '-i', os.path.join(source_dir, '*.png'),
            '-vf', complex_filter,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
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
        output_video = os.path.join(output_dir, f"trippy_4k_{timestamp}.mp4")

        # Handle file collisions
        counter = 1
        while os.path.exists(output_video):
            output_video = os.path.join(output_dir, f"trippy_4k_{timestamp}_{counter}.mp4")
            counter += 1

        if convert_png_to_video(source_dir, output_video):
            messagebox.showinfo("Success", f"Video conversion complete!\nSaved to: {output_video}")
        else:
            messagebox.showerror("Error", "Conversion failed. Check console for details.")

    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
