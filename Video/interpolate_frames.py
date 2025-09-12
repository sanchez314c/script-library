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
# Script Name: interpolate_frames.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced frame interpolation tool that increases frame rates of    
#              PNG image sequences using OpenCV linear interpolation. Features        
#              multi-processing with configurable FPS conversion and progress tracking.                                               
#
# Usage: python3 interpolate_frames.py
#
# Dependencies: opencv-python, numpy, tkinter, concurrent.futures                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Uses cv2.addWeighted for smooth linear interpolation, supports custom       
#        source/target FPS ratios, includes multi-core processing with up to 16   
#        workers, and features real-time progress tracking with time estimation.                                                    
#                                                                                
####################################################################################

"""
Frame Interpolation Tool

Advanced frame interpolation tool that increases frame rates of PNG image
sequences using OpenCV linear interpolation with professional-grade processing.
"""

import os
import sys
import time
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from glob import glob
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import threading

# Check if OpenCV is available, otherwise show instructions
try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: This script requires OpenCV and NumPy.")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("frame-interpolator")

def create_progress_dialog(total: int) -> Tuple[tk.Toplevel, ttk.Progressbar, tk.Label, tk.Label, tk.Tk]:
    """Create a progress dialog window."""
    root = tk.Tk()
    root.withdraw()
    
    window = tk.Toplevel(root)
    window.title("Interpolating Frames")
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
    
    progress_label = tk.Label(frame, text="Starting frame interpolation...", anchor=tk.W)
    progress_label.pack(fill=tk.X, pady=(0, 10))
    
    progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, 
                                  length=360, mode='determinate', maximum=total)
    progress_bar.pack(fill=tk.X)
    
    time_label = tk.Label(frame, text="", anchor=tk.W)
    time_label.pack(fill=tk.X, pady=(10, 0))
    
    return window, progress_bar, progress_label, time_label, root

def process_frame_pair(args: Tuple) -> List[Tuple[str, np.ndarray]]:
    """
    Process a pair of frames to generate interpolated frames between them.
    
    Args:
        args: Tuple containing (frame1_path, frame2_path, pair_idx, target_fps, source_fps, output_dir)
        
    Returns:
        List of tuples (output_path, frame_data) for writing later
    """
    frame1_path, frame2_path, pair_idx, target_fps, source_fps, output_dir = args
    
    try:
        # Read the frames
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            logger.error(f"Error reading frames {frame1_path} or {frame2_path}")
            return []
        
        # Calculate number of frames to insert between the two original frames
        num_new_frames = int(target_fps / source_fps) - 1
        
        results = []
        
        # Create interpolated frames
        for j in range(1, num_new_frames + 1):
            alpha = j / (num_new_frames + 1)
            interpolated_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            output_frame_path = os.path.join(output_dir, f'frame_{pair_idx:04d}_{j:02d}.png')
            results.append((output_frame_path, interpolated_frame))
        
        # Also save the first frame of the pair
        output_frame_path_1 = os.path.join(output_dir, f'frame_{pair_idx:04d}_00.png')
        results.append((output_frame_path_1, frame1))
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing frames {frame1_path} and {frame2_path}: {e}")
        return []

def interpolate_frames(input_dir: str, output_dir: str, target_fps: int = 60, source_fps: int = 30) -> Tuple[int, int]:
    """
    Interpolate frames to achieve a higher frame rate.
    
    Args:
        input_dir: Directory containing source frames
        output_dir: Directory to save interpolated frames
        target_fps: Target frames per second
        source_fps: Source frames per second
        
    Returns:
        Tuple of (total_frames_created, error_count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all frame files (assuming they are named frame_XXXX.png)
    frame_files = sorted(glob(os.path.join(input_dir, 'frame_*.png')))
    
    if not frame_files:
        logger.error('No frames found in input directory.')
        return 0, 0
    
    num_frames = len(frame_files)
    if num_frames < 2:
        logger.error('Not enough frames to interpolate. Need at least 2 frames.')
        return 0, 0
    
    # Check if the first frame can be read to validate dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        logger.error(f"Unable to read first frame {frame_files[0]}")
        return 0, 0
    
    height, width, _ = first_frame.shape
    logger.info(f"Found {num_frames} frames of size {width}x{height}")
    
    # Create progress dialog
    window, progress_bar, progress_label, time_label, root = create_progress_dialog(num_frames - 1)
    
    # Prepare data for parallel processing
    frame_pairs = []
    for i in range(num_frames - 1):
        frame_pairs.append((frame_files[i], frame_files[i+1], i, target_fps, source_fps, output_dir))
    
    # Variables for tracking progress
    completed = 0
    results = []
    start_time = time.time()
    error_count = 0
    
    # Update function for the progress dialog
    def update_progress():
        nonlocal completed
        while completed < len(frame_pairs):
            if completed > 0:
                elapsed = time.time() - start_time
                pairs_per_second = completed / elapsed
                remaining_pairs = len(frame_pairs) - completed
                
                if pairs_per_second > 0:
                    remaining_seconds = remaining_pairs / pairs_per_second
                    minutes, seconds = divmod(int(remaining_seconds), 60)
                    time_label.config(text=f"Estimated time remaining: {minutes}m {seconds}s")
            
            time.sleep(0.1)
            root.update()
    
    # Start update thread
    update_thread = threading.Thread(target=update_progress)
    update_thread.daemon = True
    update_thread.start()
    
    # Process function to handle progress updates
    def process_and_update(args):
        nonlocal completed
        frame1_path, _, pair_idx, _, _, _ = args
        
        # Update progress dialog
        progress_bar['value'] = pair_idx + 1
        file_name = os.path.basename(frame1_path)
        progress_label.config(text=f"Processing frame pair {pair_idx + 1} of {len(frame_pairs)}: {file_name}")
        
        # Process frame pair
        result = process_frame_pair(args)
        
        # Update completed count
        completed += 1
        
        return result
    
    # Use multi-processing for frame interpolation
    try:
        max_workers = min(os.cpu_count() or 4, 16)  # Limit to avoid memory issues
        logger.info(f"Using {max_workers} CPU cores for processing")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process frame pairs and collect results
            for frame_outputs in executor.map(process_and_update, frame_pairs):
                results.extend(frame_outputs)
    
        # Write all frames to disk
        logger.info(f"Writing {len(results)} interpolated frames to disk")
        for output_path, frame_data in results:
            cv2.imwrite(output_path, frame_data)
        
        # Handle the last frame separately
        output_frame_path_last = os.path.join(output_dir, f'frame_{num_frames - 1:04d}_00.png')
        cv2.imwrite(output_frame_path_last, cv2.imread(frame_files[-1]))
        
    except Exception as e:
        logger.exception(f"Error during interpolation: {e}")
        error_count += 1
    finally:
        # Close progress dialog
        window.destroy()
        root.destroy()
    
    total_frames = len(results) + 1  # +1 for the last frame
    return total_frames, error_count

def ask_fps_values() -> Tuple[Optional[int], Optional[int]]:
    """Ask user for source and target FPS values"""
    root = tk.Tk()
    root.withdraw()
    
    # Create a dialog
    dialog = tk.Toplevel(root)
    dialog.title("Frame Rate Settings")
    dialog.geometry("300x150")
    dialog.resizable(False, False)
    
    # Center dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"+{x}+{y}")
    
    frame = tk.Frame(dialog, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Source FPS
    source_label = tk.Label(frame, text="Source FPS:")
    source_label.grid(row=0, column=0, sticky=tk.W, pady=5)
    source_var = tk.StringVar(value="30")
    source_entry = tk.Entry(frame, textvariable=source_var, width=10)
    source_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
    
    # Target FPS
    target_label = tk.Label(frame, text="Target FPS:")
    target_label.grid(row=1, column=0, sticky=tk.W, pady=5)
    target_var = tk.StringVar(value="60")
    target_entry = tk.Entry(frame, textvariable=target_var, width=10)
    target_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
    
    result = [None, None]
    
    def on_ok():
        try:
            result[0] = int(source_var.get())
            result[1] = int(target_var.get())
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for FPS values.")
    
    def on_cancel():
        dialog.destroy()
    
    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=10)
    
    ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
    ok_button.pack(side=tk.LEFT, padx=5)
    
    cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
    cancel_button.pack(side=tk.LEFT, padx=5)
    
    # Wait for dialog to close
    root.wait_window(dialog)
    root.destroy()
    
    return result[0], result[1]

def main():
    """Main program execution."""
    # Create root window for dialogs
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Ask for FPS values
        source_fps, target_fps = ask_fps_values()
        if source_fps is None or target_fps is None:
            logger.info("Operation cancelled by user.")
            return
        
        if target_fps <= source_fps:
            messagebox.showerror("Invalid Settings", 
                                "Target FPS must be greater than source FPS.")
            return
        
        # Select source folder
        input_dir = filedialog.askdirectory(title='Select the folder containing source frames')
        if not input_dir:
            logger.info("No source folder selected. Exiting.")
            return
        
        # Select destination folder
        output_dir = filedialog.askdirectory(title='Select the destination folder for interpolated frames')
        if not output_dir:
            logger.info("No destination folder selected. Exiting.")
            return
        
        # Run frame interpolation
        total_frames, error_count = interpolate_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            target_fps=target_fps,
            source_fps=source_fps
        )
        
        if total_frames > 0:
            # Show completion message
            messagebox.showinfo(
                "Processing Complete",
                f"Frame interpolation complete:\n"
                f"- Increased frame rate from {source_fps} to {target_fps} FPS\n"
                f"- Created {total_frames} interpolated frames\n"
                f"- Errors: {error_count}\n\n"
                f"Frames saved to:\n{output_dir}"
            )
        else:
            messagebox.showerror(
                "Processing Failed",
                "Frame interpolation failed. Check the log for details."
            )
        
    except Exception as e:
        logger.exception("An error occurred:")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        root.destroy()

if __name__ == "__main__":
    main()

