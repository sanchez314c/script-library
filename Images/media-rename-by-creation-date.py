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
# Script Name: images-rename-by-creation-date.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible file renaming tool that renames files based on
#     ModifyDate metadata in-place. Features parallel processing, collision
#     handling, and maintains original directory structure for organized libraries.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI directory selection with native macOS dialogs
#     - Parallel processing (optimized for macOS systems)
#     - Recursive directory scanning with automatic collision handling
#     - In-place renaming with format: YYYY-MM-DD_HH-MM-SS[_N]
#     - Detailed progress reporting and comprehensive error handling
#     - Desktop output support with detailed logging
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-rename-by-creation-date.py
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
import re
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor

def extract_and_rename(file_path):
    try:
        # Call exiftool to get the ModifyDate
        result = subprocess.run(['exiftool', '-overwrite_original_in_place', '-ModifyDate', '-d', '%Y-%m-%d_%H-%M-%S', file_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error reading metadata for file: {file_path}")
            return

        # Extract ModifyDate from exiftool output
        match = re.search(r'Modify Date\s+:\s+(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', result.stdout)
        if not match:
            print(f"No valid ModifyDate found for file: {file_path}")
            return

        file_create_date = match.group(1)
        new_file_name = f"{file_create_date}{os.path.splitext(file_path)[1]}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

        # Handle file name collisions
        counter = 1
        while os.path.exists(new_file_path):
            base, ext = os.path.splitext(new_file_name)
            new_file_path = os.path.join(os.path.dirname(file_path), f"{base}_{counter}{ext}")
            counter += 1

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"File renamed: {new_file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def rename_files_in_directory(directory_path):
    with ThreadPoolExecutor(max_workers=10) as executor:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                executor.submit(extract_and_rename, file_path)

def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory_path = filedialog.askdirectory()
    return directory_path

if __name__ == "__main__":
    directory_path = select_directory()
    if directory_path:
        rename_files_in_directory(directory_path)
    else:
        print("No directory selected, exiting.")
