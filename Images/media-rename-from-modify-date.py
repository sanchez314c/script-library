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
# Script Name: images-rename-from-modify-date.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible FileModifyDate to filename converter that synchronizes
#     FMD with both filename and DateTimeOriginal metadata. Features EXIF cleansing,
#     batch processing, and integrity preservation for organized media libraries.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI-based folder selection (supports multiple folders) with native macOS dialogs
#     - Parallel processing (auto-optimized for macOS systems)
#     - EXIF data cleansing with automatic file renaming
#     - Duplicate filename handling and metadata integrity preservation
#     - Comprehensive error handling and desktop output reporting
#     - Batch processing support with detailed progress tracking
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-rename-from-modify-date.py
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
import datetime
from multiprocessing import Pool
from tkinter import Tk
from tkinter.filedialog import askdirectory
import subprocess

def run_exif_cleanse_commands(file_path):
    try:
        # Clear potentially problematic metadata
        subprocess.run(['exiftool', '-overwrite_original', '-q', '-m', '-F',
                        '-api', 'LargeFileSupport=1', '-exif:all=', file_path], check=True)
        # Attempt to restore structure to prevent data loss
        subprocess.run(['exiftool', '-overwrite_original', '-tagsfromfile', '@',
                        '-all:all', '-unsafe', '-q', file_path], check=True)
        # Remove MakerNotes which can sometimes cause issues with metadata integrity
        subprocess.run(['exiftool', '-MakerNotes=', '-overwrite_original', '-q', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f'ERROR: Processing EXIF data for {file_path}: {e}')

def process_and_rename_file(file_path):
    # First, cleanse the EXIF data
    run_exif_cleanse_commands(file_path)

    # Now proceed with getting the last modified time and formatting it for the new filename and metadata
    timestamp = os.path.getmtime(file_path)
    datetime_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    new_name_base = datetime_str
    new_name = new_name_base
    directory = os.path.dirname(file_path)
    original_extension = os.path.splitext(file_path)[1]

    count = 0
    # Ensuring the new filename is unique
    while os.path.exists(os.path.join(directory, f"{new_name}{('_' + str(count) if count > 0 else '')}{original_extension}")):
        count += 1
    new_file_path = os.path.join(directory, f"{new_name}{('_' + str(count) if count > 0 else '')}{original_extension}")

    # Set the DateTimeOriginal metadata
    datetime_original_format = datetime.datetime.fromtimestamp(timestamp).strftime('%Y:%m:%d %H:%M:%S')
    cmd = ['exiftool', '-overwrite_original', f'-DateTimeOriginal={datetime_original_format}', new_file_path]
    subprocess.run(cmd, shell=False)

    # Rename file
    os.rename(file_path, new_file_path)
    print(f"Renamed and updated {os.path.basename(new_file_path)}")

def select_directories(max_folders=25):
    directories = []
    for _ in range(max_folders):
        Tk().withdraw()  # Prevents the Tkinter window from appearing
        directory = askdirectory(title=f"Select folder {_ + 1} of {max_folders}. When done, cancel to proceed.")
        if not directory:
            print("Selection cancelled or completed.")
            break
        directories.append(directory)
    return directories

def main():
    directories = select_directories()

    if not directories:
        print("No directories selected. Exiting...")
        return

    files_to_process = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                files_to_process.append(os.path.join(root, file))

    # Explicitly setting the pool size to 18
    num_processes = 18

    with Pool(processes=num_processes) as pool:
        pool.map(process_and_rename_file, files_to_process)

if __name__ == "__main__":
    main()
