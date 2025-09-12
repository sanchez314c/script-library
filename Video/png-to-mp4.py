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
# Script Name: png-to-mp4.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-performance PNG to MP4 converter using hardware-accelerated    
#              encoding. Converts individual PNG images to 5-second MP4 videos        
#              with professional-grade H.264 encoding and collision-safe naming.                                               
#
# Usage: python3 png-to-mp4.py
#
# Dependencies: ffmpeg, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features hardware acceleration support, CRF 18 quality encoding,       
#        yuv420p pixel format for compatibility, faststart optimization for   
#        streaming, and automatic collision detection with incremental naming.                                                    
#                                                                                
####################################################################################

"""
PNG to MP4 Converter

High-performance PNG to MP4 converter using hardware-accelerated encoding
to create professional-grade video files from individual PNG images.
"""

import os
import subprocess
from tkinter import filedialog, Tk, messagebox
from datetime import datetime

def convert_png_to_mp4(source_dir, destination_dir):
    """Convert individual PNG files to MP4 videos."""
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Get list of PNG files
        png_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]
        if not png_files:
            raise FileNotFoundError("No PNG files found in source directory.")

        total_files = len(png_files)
        processed_files = 0
        failed_files = []

        for png_file in png_files:
            try:
                input_path = os.path.join(source_dir, png_file)
                base_name = os.path.splitext(png_file)[0]
                output_path = os.path.join(destination_dir, f'{base_name}.mp4')

                # Handle file collisions
                counter = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(destination_dir, f'{base_name}_{counter}.mp4')
                    counter += 1

                # FFmpeg command with improved settings
                ffmpeg_command = [
                    'ffmpeg',
                    '-hwaccel', 'auto',
                    '-loop', '1',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-preset', 'slow',
                    '-crf', '18',
                    '-t', '5',  # 5-second duration
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-metadata', f'creation_time={datetime.now().isoformat()}',
                    output_path
                ]

                # Run FFmpeg
                process = subprocess.run(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )

                if process.returncode == 0:
                    processed_files += 1
                    print(f"Processed {processed_files}/{total_files}: {png_file}")
                else:
                    failed_files.append(png_file)
                    print(f"Error processing {png_file}: {process.stderr}")

            except Exception as e:
                failed_files.append(png_file)
                print(f"Error processing {png_file}: {str(e)}")

        # Show completion message
        if failed_files:
            messagebox.showwarning(
                "Conversion Complete",
                f"Processed {processed_files}/{total_files} files.\n"
                f"Failed files: {', '.join(failed_files)}"
            )
        else:
            messagebox.showinfo(
                "Success",
                f"Successfully converted all {total_files} files!"
            )

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return False

    return True

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

        # Select destination directory
        destination_dir = filedialog.askdirectory(title="Select Destination Directory for Videos")
        if not destination_dir:
            print("No destination directory selected, exiting.")
            return

        convert_png_to_mp4(source_dir, destination_dir)

    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
