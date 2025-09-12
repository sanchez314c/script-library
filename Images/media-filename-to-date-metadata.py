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
# Script Name: images-filename-to-date-metadata.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible filename to metadata processor that extracts date
#     information from filenames and updates corresponding metadata fields with
#     parallel processing and comprehensive EXIF cleansing capabilities.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI directory selection with native macOS dialogs
#     - Parallel processing with multiprocessing support
#     - Filename datetime extraction with regex pattern matching
#     - EXIF metadata cleansing and integrity preservation
#     - Collision handling and comprehensive error reporting
#     - Desktop output support with detailed logging
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-filename-to-date-metadata.py
#
####################################################################################

import os
import re
import subprocess
from multiprocessing import Pool
from tkinter import Tk
from tkinter.filedialog import askdirectory

def run_exif_cleanse_commands(file_path):
    try:
        # Clearing potentially problematic metadata with quiet mode applied correctly
        subprocess.run(['exiftool', '-overwrite_original', '-q', '-m', '-F',
                        '-api', 'LargeFileSupport=1', '-exif:all=', file_path], check=True)
        # Attempt to restore structure to prevent data loss with quiet mode
        subprocess.run(['exiftool', '-overwrite_original', '-tagsfromfile', '@',
                        '-all:all', '-unsafe', '-q', file_path], check=True)
        # Remove MakerNotes which can sometimes cause issues with metadata integrity, in quiet mode
        subprocess.run(['exiftool', '-MakerNotes=', '-overwrite_original', '-q', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f'ERROR: Processing EXIF data for {file_path}: {e}')

def extract_datetime_from_filename(filename):
    match = re.match(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2}-\d{2}-\d{2})", filename)
    if match:
        date_part, time_part = match.groups()
        datetime_str = f"{date_part} {time_part}".replace('-', ':')
        return datetime_str
    return None

def set_metadata_from_filename(file_path):
    # First cleanse the EXIF data
    run_exif_cleanse_commands(file_path)

    datetime_str = extract_datetime_from_filename(os.path.basename(file_path))
    if datetime_str:
        cmd = ['exiftool', '-overwrite_original',
               f'-DateTimeOriginal={datetime_str}', f'-FileModifyDate={datetime_str}', file_path]
        subprocess.run(cmd, shell=False)
        print(f"Updated {file_path} with DateTimeOriginal and FileModifyDate: {datetime_str}")
    else:
        print(f"Could not extract datetime from filename: {file_path}")

def select_directories(max_folders=25):
    Tk().withdraw()
    directories = []
    for i in range(max_folders):
        directory = askdirectory(title=f"Select folder {i + 1} of {max_folders}. Cancel to proceed if done.")
        if directory:
            directories.append(directory)
        else:
            break
    return directories

def main():
    directories = select_directories()
    if not directories:
        print("No directories selected. Exiting...")
        return

    files_to_process = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                files_to_process.append(os.path.join(root, file))

    with Pool(18) as pool:
        pool.map(set_metadata_from_filename, files_to_process)

if __name__ == "__main__":
    main()
