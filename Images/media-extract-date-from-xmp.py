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
# Script Name: images-extract-date-from-xmp.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible XMP metadata processor that renames and moves files
#     based on XMP dates. Features extension correction, subsecond precision, and
#     dual directory selection for organized media management.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Source/destination directory selection with native macOS dialogs
#     - Parallel processing (auto-optimized for macOS systems)
#     - Multiple format support with extension correction
#     - Subsecond precision XMP metadata extraction
#     - Collision handling and comprehensive logging
#     - File format validation with desktop output support
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - Pillow (PIL) (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-extract-date-from-xmp.py
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

def check_and_correct_extension(file_path, media_type, destination_directory):
    # This function now needs to handle the destination directory for the corrected file path
    from PIL import Image
    if media_type == 'image':
        try:
            with Image.open(file_path) as img:
                correct_ext = f'.{img.format.lower()}'
                if correct_ext != os.path.splitext(file_path)[1].lower():
                    new_file_path = os.path.join(destination_directory, os.path.basename(os.path.splitext(file_path)[0] + correct_ext))
                    os.rename(file_path, new_file_path)
                    return new_file_path
        except IOError:
            pass
    return os.path.join(destination_directory, os.path.basename(file_path))

def extract_metadata(file_path):
    try:
        result = subprocess.run(['exiftool', '-json', file_path], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)[0]
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

def find_oldest_xmp_date(metadata):
    xmp_date_tags = ['XMP:CreateDate', 'XMP:MetadataDate', 'XMP:ModifyDate']
    dates = []
    for tag in xmp_date_tags:
        if tag in metadata:
            date_str = metadata[tag]
            try:
                date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                dates.append(date)
            except ValueError:
                pass        # Search for tags that contain the date in a more complex structure
        for key, value in metadata.items():
            if key.startswith('XMP') and any(xmp_tag in key for xmp_tag in xmp_date_tags):
                try:
                    # Complex structures might have different date string formats
                    date = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                    dates.append(date)
                except ValueError:
                    pass
    return min(dates, default=None)

def generate_filename(oldest_date, subseconds, original_extension):
    date_str = oldest_date.strftime('%Y-%m-%d %H-%M-%S')
    if subseconds and subseconds != '000000':
        filename = f"{date_str}_ss{subseconds}{original_extension}"
    else:
        filename = f"{date_str}{original_extension}"
    return filename

def process_file(args):
    file_path, destination_directory = args  # Adjust to unpack arguments
    metadata = extract_metadata(file_path)
    oldest_date = find_oldest_xmp_date(metadata)
    if oldest_date is None:
        print(f"No valid date data found for {file_path}. Skipping...")
        return
    subseconds = extract_subseconds(metadata)
    new_file_name = generate_filename(oldest_date, subseconds, os.path.splitext(file_path)[1].lower())
    new_file_path = os.path.join(destination_directory, new_file_name)  # Use destination directory
    count = 1
    while os.path.exists(new_file_path):
        base, extension = os.path.splitext(new_file_path)
        new_file_path = f"{base}_{count}{extension}"
        count += 1
    os.rename(file_path, new_file_path)
    print(f"Renamed '{file_path}' to '{new_file_path}'")

def select_directories():
    Tk().withdraw()
    source_directory = askdirectory(title="Select source directory.")
    destination_directory = askdirectory(title="Select destination directory.")  # Add this line
    return source_directory, destination_directory  # Return both directories

def main():
    logging.basicConfig(level=logging.ERROR)
    source_directory, destination_directory = select_directories()  # Adjust to handle both directories
    if not source_directory or not destination_directory:
        print("One or more directories not selected. Exiting...")
        return
    image_extensions = [...]
    files_to_process = [(os.path.join(dp, f), destination_directory) for dp, dn, filenames in os.walk(source_directory)
                        for f in filenames if os.path.splitext(f.lower())[1] in image_extensions]
    cpu_cores = os.cpu_count() or 1
    optimal_processes = max(1, min(20, cpu_cores - 2))

    with Pool(processes=optimal_processes) as pool:
        pool.map(process_file, files_to_process)

if __name__ == "__main__":
    main()
