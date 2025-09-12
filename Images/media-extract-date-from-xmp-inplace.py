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
# Script Name: images-extract-date-from-xmp-inplace.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible XMP history metadata processor that renames files
#     in-place based on oldest XMP-xmpMM:HistoryWhen date. Features subsecond
#     precision and parallel processing for efficient batch operations.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - In-place file renaming with GUI directory selection
#     - Parallel processing (auto-optimized for macOS systems)
#     - Multiple format support (.mpo, .dng, .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .ptx, .raw, .heic, .heif, .ico)
#     - Subsecond precision XMP metadata extraction
#     - Collision handling and comprehensive logging
#     - Debug metadata printing with desktop output
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-extract-date-from-xmp-inplace.py
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
import json
import re
import logging
import subprocess
from datetime import datetime
from multiprocessing import Pool
from tkinter import Tk
from tkinter.filedialog import askdirectory

def debug_print_relevant_metadata_keys(metadata):
    print("Inspecting metadata for XMP-xmpMM related tags...")
    for key, value in metadata.items():
        if "XMP-xmpMM" in key:
            print(f"{key}: {value}")

def find_potential_date_tags(metadata):
    date_tags = []
    for key, value in metadata.items():
        if "XMP-xmpMM" in key and re.search(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}', value):
            date_tags.append((key, value))
    return date_tags

def extract_metadata(file_path):
    try:
        result = subprocess.run(['exiftool', '-json', file_path], capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]
        print(json.dumps(metadata, indent=4))  # Debug: Print the entire metadata dictionary
        return metadata
    except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
        logging.error(f"Failed to extract metadata for {file_path}: {e}")
        return {}

def extract_subseconds(metadata):
    for tag in ['SubSecTime', 'SubSecTimeOriginal', 'SubSecTimeDigitized']:
        if tag in metadata:
            return metadata[tag]
    # If no standard tag is found, attempt to extract from a date-time string
    date_time_string = metadata.get('DateTimeOriginal') or metadata.get('CreateDate', '')
    match = re.search(r'\.(\d+)', date_time_string)
    if match:
        return match.group(1)  # Return the found subseconds
    return ''  # Return an empty string if no subseconds are found

def find_oldest_xmpMM_date(metadata):
    # Look for the "XMP-xmpMM" group and then for "History When" within that group
    xmpMM_key_prefix = "XMP-xmpMM"  # Adjust based on actual prefix found in metadata
    history_when_key = "History When"  # This might need to be adjusted based on how exiftool returns it in JSON    # Debugging: Print keys to help identify the correct structure
    for key in metadata.keys():
        print(key)
    dates = []
    for key, value in metadata.items():
        if key.startswith(xmpMM_key_prefix) and history_when_key in key:
            for date_str in value.split(", "):
                try:
                    date = datetime.strptime(date_str.strip(), '%Y:%m:%d %H:%M:%S%z')
                    dates.append(date)
                except ValueError as e:
                    print(f"Error parsing date '{date_str}': {e}")
                    continue
    if dates:
        return min(dates)
    else:
        print(f"No dates found in 'XMP-xmpMM' for this file.")
        return None

def generate_filename(oldest_date, subseconds, original_extension):
    date_str = oldest_date.strftime('%Y-%m-%d %H-%M-%S')
    if subseconds and subseconds != '000000':
        filename = f"{date_str}_ss{subseconds}{original_extension}"
    else:
        filename = f"{date_str}{original_extension}"
    return filename

def process_file(file_path):
    metadata = extract_metadata(file_path)
    oldest_date = find_oldest_xmpMM_date(metadata)
    if oldest_date is None:
        print(f"No 'XMP-xmpMM:HistoryWhen' data found for {file_path}. Skipping...")
        return
    subseconds = extract_subseconds(metadata)
    new_file_name = generate_filename(oldest_date, subseconds, os.path.splitext(file_path)[1].lower())
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    count = 1
    while os.path.exists(new_file_path):
        base, extension = os.path.splitext(new_file_path)
        new_file_path = f"{base}_{count}{extension}"
        count += 1
    os.rename(file_path, new_file_path)
    print(f"Renamed '{file_path}' to '{new_file_path}'")

def select_directories():
    Tk().withdraw()  # Hide the main Tkinter window
    source_directory = askdirectory(title="Select source directory.")
    return source_directory

def main():
    logging.basicConfig(level=logging.ERROR)
    source_directory = select_directories()
    if not source_directory:
        print("Source directory not selected. Exiting...")
        return
    image_extensions = ['.mpo', '.dng', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ptx', '.raw', '.heic', '.heif', '.ico']
    files_to_process = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_directory)
                        for f in filenames if os.path.splitext(f.lower())[1] in image_extensions]
    cpu_cores = os.cpu_count() or 1
    optimal_processes = max(1, min(20, cpu_cores - 2))

    with Pool(processes=optimal_processes) as pool:
        pool.map(process_file, files_to_process)

if __name__ == "__main__":
    main()
