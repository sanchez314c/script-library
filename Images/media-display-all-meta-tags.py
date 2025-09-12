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
# Script Name: images-display-all-meta-tags.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible metadata tag discovery tool that recursively
#     scans directories for all files and displays date/time related metadata.
#     Uses parallel processing and provides intuitive GUI interface.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI directory selection with native macOS dialogs
#     - Parallel processing using ThreadPoolExecutor for efficiency
#     - Comprehensive metadata tag discovery and filtering
#     - Case-insensitive date/time tag filtering with sorted display
#     - Detailed logging with desktop output support
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-display-all-meta-tags.py
#
####################################################################################
Title in Title Case
------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Clear, concise description of script purpose and functionality.
    Multiple lines if needed.

Features:
    - Feature one
    - Feature two
    - Additional features

Requirements:
    - Python version
    - Required packages
    - System dependencies

Usage:
    python script-name.py [arguments]

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_directory(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    dir_path = filedialog.askdirectory(title=prompt)
    return dir_path

def get_date_time_tags(file_path):
    try:
        # Running exiftool for each file
        result = subprocess.run(
            ['exiftool', '-s', '-G0:1', file_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        # Use grep-like functionality to filter for 'date' or 'time'
        return [line for line in result.stdout.split('\n') if 'date' in line.lower() or 'time' in line.lower()]
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []

def main():
    # Select the source directory using tkinter GUI
    source_directory = select_directory("Select the Source Directory for scanning")
    if not source_directory:
        logging.error("Source directory not selected. Exiting.")
        return

    # Get a list of all files in the source directory
    file_paths = [os.path.join(root, name)
                  for root, _, files in os.walk(source_directory)
                  for name in files]

    # Check if file paths are found
    if not file_paths:
        logging.error("No files found in the source directory. Exiting.")
        return

    logging.info(f"Found {len(file_paths)} files in source directory. Starting to process...")

    # Define a set to store unique date/time tags
    unique_tags = set()

    # Using ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all file paths to the executor
        future_to_file = {executor.submit(get_date_time_tags, file_path): file_path for file_path in file_paths}

        # As each future is completed, process its result
        for future in as_completed(future_to_file):
            file_tags = future.result()
            unique_tags.update(file_tags)

    # Sort and print unique date/time tags
    unique_tags_sorted = sorted(unique_tags)
    print("\nUnique Date/Time Tags Found:")
    print("\n".join(unique_tags_sorted))

    logging.info("Scanning complete.")

if __name__ == "__main__":
    main()
