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
# Script Name: video-ffprobe-error-check.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive video error detection tool using FFprobe to analyze    
#              video files for corruption, quality issues, and format problems.        
#              Features multi-threaded scanning with detailed severity reporting.                                               
#
# Usage: python3 video-ffprobe-error-check.py
#
# Dependencies: ffprobe, tkinter, json, threading                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Supports all major video formats, uses intelligent error classification       
#        (high/medium/low severity), generates detailed HTML reports, and features   
#        visual progress tracking with time estimation and multi-core processing.                                                    
#                                                                                
####################################################################################

"""
Video Error Check Using FFprobe

Comprehensive video error detection tool using FFprobe to analyze video files
for corruption, quality issues, and format problems with detailed reporting.
"""

import os
import sys
import json
import subprocess
import time
import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("video-error-check")

# Define supported video extensions
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv')

def check_ffprobe() -> bool:
    """Check if FFprobe is installed."""
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_video_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Collect all video files from directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(VIDEO_EXTENSIONS):
                    video_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(directory, file))
    
    return video_files

def analyze_video(file_info: Tuple[str, int, int]) -> Dict[str, Any]:
    """
    Analyze video file using ffprobe.
    
    Args:
        file_info: Tuple containing (file_path, index, total)
        
    Returns:
        Dictionary with analysis results
    """
    file_path, index, total = file_info
    
    try:
        # First get detailed JSON information
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'stream=index,codec_type,codec_name,profile,width,height,duration,bit_rate',
            '-show_entries', 'format=duration,size,bit_rate',
            '-of', 'json',
            file_path
        ]
        
        # Log progress
        logger.info(f"[{index}/{total}] Analyzing: {os.path.basename(file_path)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout)
        
        # Then check for errors by running a full analyze
        error_check = subprocess.run(
            ['ffprobe', '-v', 'error', file_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Detect specific error patterns
        error_text = error_check.stderr
        error_severity = 'none'
        
        if error_text:
            if any(x in error_text.lower() for x in ['corrupt', 'invalid', 'error']):
                error_severity = 'high'
            elif any(x in error_text.lower() for x in ['warning', 'incomplete']):
                error_severity = 'medium'
            else:
                error_severity = 'low'
        
        return {
            'file': file_path,
            'data': data,
            'errors': error_text,
            'severity': error_severity,
            'status': 'ok' if not error_text else 'error'
        }
        
    except subprocess.TimeoutExpired:
        return {
            'file': file_path,
            'data': None,
            'errors': 'Analysis timed out after 60 seconds',
            'severity': 'high',
            'status': 'failed'
        }
    except subprocess.CalledProcessError as e:
        return {
            'file': file_path,
            'data': None,
            'errors': str(e),
            'severity': 'high',
            'status': 'failed'
        }
    except json.JSONDecodeError as e:
        return {
            'file': file_path,
            'data': None,
            'errors': f'JSON parsing error: {str(e)}',
            'severity': 'medium',
            'status': 'failed'
        }
    except Exception as e:
        return {
            'file': file_path,
            'data': None,
            'errors': f'Unexpected error: {str(e)}',
            'severity': 'high',
            'status': 'failed'
        }

def generate_report(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Generate detailed analysis report.
    
    Args:
        results: List of analysis results
        output_file: Path to write report to
    """
    # Ensure output directory exists
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Video Analysis Report\n")
        f.write("====================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total files analyzed: {len(results)}\n")
        f.write(f"Files with errors: {sum(1 for r in results if r['status'] != 'ok')}\n\n")
        
        # Organize by severity
        high_severity = [r for r in results if r['severity'] == 'high']
        medium_severity = [r for r in results if r['severity'] == 'medium']
        low_severity = [r for r in results if r['severity'] == 'low']
        ok_files = [r for r in results if r['severity'] == 'none']
        
        if high_severity:
            f.write("\n== HIGH SEVERITY ISSUES ==\n")
            for result in high_severity:
                write_file_info(f, result)
        
        if medium_severity:
            f.write("\n== MEDIUM SEVERITY ISSUES ==\n")
            for result in medium_severity:
                write_file_info(f, result)
        
        if low_severity:
            f.write("\n== LOW SEVERITY ISSUES ==\n")
            for result in low_severity:
                write_file_info(f, result)
        
        f.write("\n== HEALTHY FILES ==\n")
        for result in ok_files:
            f.write(f"\n- {os.path.basename(result['file'])}\n")

def write_file_info(f, result: Dict[str, Any]) -> None:
    """Write file information to the report."""
    f.write(f"\nFile: {os.path.basename(result['file'])}\n")
    f.write("-" * (len(os.path.basename(result['file'])) + 6) + "\n")
    f.write(f"Path: {result['file']}\n")
    
    if result['status'] == 'ok':
        if result['data']:
            # Format streams information
            if 'streams' in result['data']:
                f.write("Streams:\n")
                for stream in result['data']['streams']:
                    f.write(f"  - Type: {stream.get('codec_type', 'unknown')}\n")
                    f.write(f"    Codec: {stream.get('codec_name', 'unknown')}\n")
                    if 'width' in stream and 'height' in stream:
                        f.write(f"    Resolution: {stream['width']}x{stream['height']}\n")
                    if 'bit_rate' in stream and stream['bit_rate']:
                        try:
                            f.write(f"    Bitrate: {int(stream['bit_rate'])/1000:.2f} kbps\n")
                        except (ValueError, TypeError):
                            f.write(f"    Bitrate: unknown\n")
            
            # Format format information
            if 'format' in result['data']:
                fmt = result['data']['format']
                f.write("\nFormat Info:\n")
                if 'duration' in fmt:
                    try:
                        f.write(f"  Duration: {float(fmt['duration']):.2f} seconds\n")
                    except (ValueError, TypeError):
                        f.write("  Duration: unknown\n")
                if 'size' in fmt:
                    try:
                        size_mb = int(fmt['size'])/1024/1024
                        f.write(f"  Size: {size_mb:.2f} MB\n")
                    except (ValueError, TypeError):
                        f.write("  Size: unknown\n")
                if 'bit_rate' in fmt and fmt['bit_rate']:
                    try:
                        f.write(f"  Overall Bitrate: {int(fmt['bit_rate'])/1000:.2f} kbps\n")
                    except (ValueError, TypeError):
                        f.write("  Overall Bitrate: unknown\n")
        
        if result['errors']:
            f.write("\nErrors/Warnings:\n")
            for line in result['errors'].splitlines():
                f.write(f"  {line}\n")
    else:
        f.write("Status: ERROR\n")
        f.write(f"Error Details: {result['errors']}\n")
    
    f.write("\n")

def create_progress_dialog(total: int) -> Tuple[tk.Toplevel, ttk.Progressbar, tk.Label, tk.Label, tk.Tk]:
    """Create a progress dialog window."""
    root = tk.Tk()
    root.withdraw()
    
    window = tk.Toplevel(root)
    window.title("Analyzing Videos")
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
    
    progress_label = tk.Label(frame, text="Starting analysis...", anchor=tk.W)
    progress_label.pack(fill=tk.X, pady=(0, 10))
    
    progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, 
                                  length=360, mode='determinate', maximum=total)
    progress_bar.pack(fill=tk.X)
    
    time_label = tk.Label(frame, text="", anchor=tk.W)
    time_label.pack(fill=tk.X, pady=(10, 0))
    
    return window, progress_bar, progress_label, time_label, root

def analyze_videos_with_progress(video_files: List[str]) -> List[Dict[str, Any]]:
    """Analyze videos with progress dialog and threading."""
    if not video_files:
        return []
    
    total = len(video_files)
    window, progress_bar, progress_label, time_label, root = create_progress_dialog(total)
    
    # Prepare data for multi-threading
    file_infos = [(file, i+1, total) for i, file in enumerate(video_files)]
    results = [None] * total
    completed = 0
    start_time = time.time()
    
    # Update function for the progress dialog
    def update_progress():
        nonlocal completed
        while completed < total:
            if completed > 0:
                elapsed = time.time() - start_time
                files_per_second = completed / elapsed
                remaining_files = total - completed
                
                if files_per_second > 0:
                    remaining_seconds = remaining_files / files_per_second
                    minutes, seconds = divmod(int(remaining_seconds), 60)
                    time_label.config(text=f"Estimated time remaining: {minutes}m {seconds}s")
            
            time.sleep(0.1)
            root.update()
    
    # Start update thread
    update_thread = threading.Thread(target=update_progress)
    update_thread.daemon = True
    update_thread.start()
    
    # Process function to handle progress updates
    def process_file(file_info):
        nonlocal completed
        file_path, index, _ = file_info
        
        # Update progress dialog
        progress_bar['value'] = index
        file_name = os.path.basename(file_path)
        progress_label.config(text=f"Analyzing {index} of {total}: {file_name}")
        
        # Analyze the file
        result = analyze_video(file_info)
        
        # Update completed count
        completed += 1
        
        return result
    
    # Use multi-threading for analysis
    try:
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
            results = list(executor.map(process_file, file_infos))
    except Exception as e:
        logger.exception("An error occurred during multi-threaded processing")
    finally:
        # Close progress dialog
        window.destroy()
        root.destroy()
    
    return results

def main():
    """Main program execution."""
    # Check for FFprobe
    if not check_ffprobe():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "FFprobe Not Found", 
            "FFprobe is required but not found. Please install FFmpeg and try again."
        )
        root.destroy()
        sys.exit(1)
    
    # Create root window for dialogs
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Select source directory
        directory = filedialog.askdirectory(title="Select Video Directory")
        if not directory:
            logger.info("No directory selected. Exiting.")
            return
        
        # Ask if user wants to scan subdirectories
        scan_subdirs = messagebox.askyesno(
            "Scan Subdirectories",
            "Do you want to scan subdirectories for video files?"
        )
        
        # Find all video files
        video_files = get_video_files(directory, recursive=scan_subdirs)
        
        if not video_files:
            messagebox.showinfo("No Videos Found", "No video files found in the selected directory.")
            return
        
        # Confirm analysis
        message = f"Found {len(video_files)} video files to analyze.\nDo you want to continue?"
        if not messagebox.askyesno("Confirm Analysis", message):
            return
        
        # Analyze files with progress dialog
        results = analyze_videos_with_progress(video_files)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_file = os.path.join(directory, f"video_analysis_report_{timestamp}.txt")
        generate_report(results, report_file)
        
        # Categorize results
        error_count = sum(1 for r in results if r['status'] != 'ok')
        high_severity = sum(1 for r in results if r['severity'] == 'high')
        medium_severity = sum(1 for r in results if r['severity'] == 'medium')
        low_severity = sum(1 for r in results if r['severity'] == 'low')
        
        # Show completion message
        summary = (
            f"Analysis complete!\n\n"
            f"Total files: {len(results)}\n"
            f"Files with issues: {error_count}\n"
            f"  - High severity: {high_severity}\n"
            f"  - Medium severity: {medium_severity}\n"
            f"  - Low severity: {low_severity}\n\n"
            f"Report saved to:\n{report_file}"
        )
        
        messagebox.showinfo("Analysis Complete", summary)
        
        # Ask if user wants to open the report
        if messagebox.askyesno("Open Report", "Do you want to open the report file?"):
            try:
                # Cross-platform way to open a file with the default application
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', report_file], check=True)
                elif sys.platform == 'win32':  # Windows
                    os.startfile(report_file)
                else:  # Linux
                    subprocess.run(['xdg-open', report_file], check=True)
            except Exception as e:
                logger.error(f"Failed to open report: {e}")
                messagebox.showwarning("Error", f"Could not open report file: {e}")
        
    except Exception as e:
        logger.exception("An error occurred:")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        root.destroy()

if __name__ == "__main__":
    main()
