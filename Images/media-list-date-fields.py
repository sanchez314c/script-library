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
# Script Name: images-list-date-fields.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible datetime metadata field discovery tool that
#     recursively scans directories to extract every date-related metadata
#     field. Results are automatically saved to desktop with comprehensive reporting.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI folder selection with native macOS dialogs
#     - Parallel processing (auto-optimized for macOS systems)
#     - Recursive directory scanning with metadata field extraction
#     - Automatic desktop file output (datetime_fields.txt)
#     - Set-based duplicate elimination with sorted results
#     - Desktop output support with detailed progress tracking
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - pyexiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-list-date-fields.py
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
import multiprocessing
from tkinter import filedialog, Tk
import pyexiftool

def process_file(file_path):
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(file_path)
    date_time_fields = set()
    for key, value in metadata.items():
        if isinstance(value, str) and 'date' in key.lower():
            date_time_fields.add(key)
    return date_time_fields

def process_folder(folder_path):
    date_time_fields = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            date_time_fields.update(process_file(file_path))
    return date_time_fields

def main():
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder")

    pool = multiprocessing.Pool(processes=10)
    result = pool.map(process_folder, [folder_path])
    pool.close()
    pool.join()

    date_time_fields = set()
    for res in result:
        date_time_fields.update(res)

    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_file = os.path.join(desktop_path, 'datetime_fields.txt')
    with open(output_file, 'w') as f:
        for field in date_time_fields:
            f.write(field + '\n')

    print("Process complete. Output saved to:", output_file)

if __name__ == "__main__":
    main()
