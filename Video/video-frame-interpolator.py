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
# Script Name: video-frame-interpolator.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced video frame interpolation tool that converts videos to 120 FPS    
#              using FFmpeg's minterpolate filter. Features parallel processing,        
#              automatic dimension detection, and batch folder processing with GUI.                                               
#
# Usage: python3 video-frame-interpolator.py
#
# Dependencies: ffmpeg-bar, ffprobe, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Requires FFmpeg with minterpolate support. Uses enhanced multi-threading       
#        with (cores × 12) workers for maximum performance. Supports MP4, MKV,   
#        AVI, and MOV input formats with automatic resolution preservation.                                                    
#                                                                                
####################################################################################

"""
Video Frame Interpolator

Advanced video frame interpolation tool that converts videos to 120 FPS using
FFmpeg's minterpolate filter with parallel processing and automatic optimization.
"""

import tkinter as tk
from tkinter import filedialog
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def select_source_folder():
    folder_path = filedialog.askdirectory(title="Select Source Folder")
    entry_source_folder.delete(0, tk.END)
    entry_source_folder.insert(0, folder_path)

def select_output_folder():
    folder_path = filedialog.askdirectory(title="Select Output Folder")
    entry_output_folder.delete(0, tk.END)
    entry_output_folder.insert(0, folder_path)

def get_video_dimensions(video_path):
    try:
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        width = video_info['streams'][0]['width']
        height = video_info['streams'][0]['height']
        return f"{width}x{height}"
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return None

def render_video(args):
    source_video, output_path, resolution = args
    try:
        render_command = (
            f"ffmpeg-bar -i \"{source_video}\" "
            f'-vf "minterpolate=fps=120,scale={resolution}" '
            f"-c:v libx264 -pix_fmt yuv420p -color_range 1 "
            f"\"{output_path}\""
        )
        print(f"Running render command: {render_command}")
        subprocess.run(render_command, shell=True)
        print(f"Video rendered successfully. Output saved to {output_path}")
    except Exception as e:
        print(f"Error rendering video {source_video}: {e}")

def process_folder(source_folder, output_folder):
    tasks = []
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
                source_video = os.path.join(root, filename)
                relative_path = os.path.relpath(root, source_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                root_filename = os.path.splitext(filename)[0]
                output_filename = f"{root_filename}_h264_minterpolate-120fps.mp4"
                output_path = os.path.join(output_dir, output_filename)

                # Get the video dimensions
                resolution = get_video_dimensions(source_video)
                if not resolution:
                    print(f"Skipping video {source_video} due to error in getting dimensions.")
                    continue

                tasks.append((source_video, output_path, resolution))

    # Use a ThreadPoolExecutor to process the tasks in parallel
    num_workers = os.cpu_count() * 12  # Double the number of workers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(render_video, task) for task in tasks]
        for future in as_completed(futures):
            future.result()  # Ensure any exceptions are raised

def render_videos():
    source_folder = entry_source_folder.get()
    output_folder = entry_output_folder.get()
    if not source_folder or not output_folder:
        print("Please select both source folder and output folder.")
        return

    print(f"Source folder: {source_folder}")
    print(f"Output folder: {output_folder}")

    process_folder(source_folder, output_folder)

if __name__ == "__main__":
    app = tk.Tk()
    app.title("Batch Video Renderer")

    frame = tk.Frame(app)
    frame.pack(padx=10, pady=10)

    btn_select_source = tk.Button(frame, text="Select Source Folder", command=select_source_folder)
    btn_select_source.grid(row=0, column=0, pady=5)

    entry_source_folder = tk.Entry(frame, width=50)
    entry_source_folder.grid(row=0, column=1, padx=5)

    btn_select_output = tk.Button(frame, text="Select Output Folder", command=select_output_folder)
    btn_select_output.grid(row=1, column=0, pady=5)

    entry_output_folder = tk.Entry(frame, width=50)
    entry_output_folder.grid(row=1, column=1, padx=5)

    btn_render = tk.Button(frame, text="Render Videos", command=render_videos)
    btn_render.grid(row=2, columnspan=2, pady=10)

    app.mainloop()
