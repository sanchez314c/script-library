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
# Script Name: images-list-datetime-metadata.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible metadata datetime field scanner that recursively
#     processes directories to extract and catalog all date-related metadata.
#     Features JSON extraction and automatic desktop output with sorted results.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI folder selection with native macOS dialogs
#     - Parallel processing (multi-core optimization for macOS)
#     - Recursive directory scanning with JSON metadata extraction
#     - Case-insensitive field matching and sorted output generation
#     - Automatic desktop output (datetime_fields.txt) with progress tracking
#     - Comprehensive error handling and reporting with desktop logging
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - pyexiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-list-datetime-metadata.py
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

import multiprocessing
from tkinter import filedialog, Tk
import os
import exiftool

def process_file(file_path):
    date_time_fields = set()
    with exiftool.ExifTool() as et:
        metadata = et.execute_json(f'-G -j "{file_path}"')
    for item in metadata:
        for key, value in item.items():
            if isinstance(value, str) and 'date' in key.lower():
                date_time_fields.add(key)
    return date_time_fields

def worker(file_paths):
    date_time_fields = set()
    for file_path in file_paths:
        try:
            date_time_fields.update(process_file(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return date_time_fields

def main():
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder")

    # Correctly build the list of file paths
    files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # Chunk files for multiprocessing
    n = 10  # Adjust this number based on your needs
    chunks = [files[i:i + n] for i in range(0, len(files), n)]

    results = pool.map(worker, chunks)

    # Combine results
    all_date_time_fields = set().union(*results)

    # Output
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_file = os.path.join(desktop_path, 'datetime_fields.txt')
    with open(output_file, 'w') as f:
        for field in sorted(all_date_time_fields):
            f.write(f"{field}\n")

    print(f"Process complete. Output saved to: {output_file}")
