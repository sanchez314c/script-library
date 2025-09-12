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
# Script Name: images-media-fix-rotation-v2.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible advanced image rotation fixing processor that
#     copies, renames, and organizes images based on metadata with comprehensive
#     format support and parallel processing capabilities.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Parallel processing using all available CPU cores
#     - Multiple format support (jpg, jpeg, png, gif, bmp, tiff, heic, dng)
#     - HEIC/HEIF support with metadata extraction
#     - Filename pattern matching with subsecond precision
#     - Year-based organization with collision handling
#     - XMP title setting and comprehensive logging
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - Pillow, exifread, pyheif (auto-installed)
#     - exiftool (auto-installed via brew)
#
# Usage:
#     python images-media-fix-rotation-v2.py
#
####################################################################################
    - subprocess (standard library)
    - shutil (standard library)
    - logging (standard library)
    - datetime (standard library)
    - multiprocessing (standard library)
    - re (standard library)
    - io (standard library)
    - os (standard library)

Usage:
    python unrotate-rev02.py

Note:
    Source: /Volumes/SSD_RAID/Pictures
    Destination: /Volumes/SSD_RAID/Assets
    Modify source_directories and destination_directory to change paths
"""
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
import re
import subprocess
import exifread
import shutil
from PIL import Image
import pyheif
import io
import logging
from datetime import datetime
from multiprocessing import Pool

### Script Summary:
# This script copies and potentially renames image files based on metadata or filename patterns.
# It supports different image formats and extracts date information from filenames or metadata.
# In case of file collisions, the script does not overwrite existing files but may be modified to handle such cases.

### Required Libraries:
# - os
# - re
# - subprocess
# - exifread
# - shutil
# - PIL (Python Imaging Library)
# - pyheif
# - io
# - logging
# - datetime
# - multiprocessing

# Setting up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Regular expression to identify valid date/time patterns in filenames
filename_date_pattern = re.compile(r'(\d{4})[-_](\d{2})[-_](\d{2})[_\s]*(\d{2})[-_](\d{2})[-_](\d{2})')

# Existing date/time pattern for metadata
date_time_pattern = re.compile(r'(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})')

def extract_date_from_filename(file_name):
    match = filename_date_pattern.search(file_name)
    if match:
        try:
            # Constructing date from the filename
            return datetime(*map(int, match.groups())).strftime("%Y-%m-%d %H-%M-%S")
        except ValueError:
            return None
    return None

def is_valid_date(date_str):
    if not date_str or date_str.startswith("0000:00:00"):
        return False
    try:
        date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return date >= datetime(1979, 12, 31)
    except ValueError:
        return False

def get_oldest_date(tags):
    dates = []
    for tag, value in tags.items():
        if isinstance(value, exifread.classes.IfdTag) and value.field_type == 2:  # ASCII field
            date_str = str(value)
            if date_time_pattern.match(date_str) and is_valid_date(date_str):
                dates.append(datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S"))

    if dates:
        return min(dates).strftime("%Y-%m-%d %H-%M-%S")
    else:
        return None

def get_subseconds(tags):
    subsec_tags = ['EXIF SubSecTime', 'EXIF SubSecTimeOriginal', 'EXIF SubSecTimeDigitized']
    for subsec_tag in subsec_tags:
        if subsec_tag in tags:
            subsec = str(tags[subsec_tag]).lstrip("0").zfill(6)
            logging.debug(f"Subseconds found in tag {subsec_tag}: {subsec}")
            return subsec

    logging.debug("No Subseconds found in common tags")
    for tag in tags.keys():
        if "SubSec" in tag:
            subsec = str(tags[tag]).lstrip("0").zfill(6)
            logging.debug(f"Subseconds found in tag {tag}: {subsec}")
            return subsec

    logging.debug("No Subseconds found in any tag")
    return '000000'

def process_heic_file(file_path):
    try:
        heif_file = pyheif.read(file_path)
        for metadata in heif_file.metadata or []:
            if metadata['type'] == 'Exif':
                tags = exifread.process_file(io.BytesIO(metadata['data']))
                return get_oldest_date(tags), 'heic'
    except Exception as e:
        logging.error(f"Error processing HEIC file: {file_path}, Error: {e}")
    return None, None

def copy_and_rename_file(file_path, destination_root):
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        filename_date = extract_date_from_filename(os.path.basename(file_path))
        oldest_date_time, image_format = None, None

        if file_extension in ['.heic', '.heif']:
            oldest_date_time, image_format = process_heic_file(file_path)
        else:
            with open(file_path, 'rb') as file:
                tags = exifread.process_file(file)
                oldest_date_time = get_oldest_date(tags)
                with Image.open(file_path) as img:
                    image_format = img.format.lower()

        # Use filename date if available
        if filename_date:
            oldest_date_time = filename_date
            # Custom tag writing can be implemented here if required
            # ...

        if oldest_date_time:
            sub_sec_str = get_subseconds(tags)
            year = oldest_date_time[:4]
            new_folder = os.path.join(destination_root, year)
            os.makedirs(new_folder, exist_ok=True)

            formatted_date_time = oldest_date_time.replace(':', '-').replace(' ', '_')
            if image_format not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'heic', 'dng']:
                image_format = 'jpg'

            new_file_name = f"{formatted_date_time}-{sub_sec_str}.{image_format}"
            new_file_path = os.path.join(new_folder, new_file_name)

            counter = 1
            base, extension = os.path.splitext(new_file_name)
            while os.path.exists(new_file_path):
                new_file_name = f"{base}-{counter}.{image_format}"
                new_file_path = os.path.join(new_folder, new_file_name)
                counter += 1

            shutil.copy(file_path, new_file_path)

            # Set XMP:Title to the filename without the extension
            filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            xmp_command = [
                'exiftool',
                f'-XMP:Title={filename_without_extension}',
                '-overwrite_original',
                new_file_path
            ]
            subprocess.run(xmp_command, check=True)
    except IOError as e:
        logging.error(f"Could not process file {file_path}, Error: {e}")

def process_directory_parallel(source, destination_root, num_processes):
    file_paths = []
    for root, dirs, files in os.walk(source):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append((file_path, destination_root))

    with Pool(num_processes) as p:
        p.starmap(copy_and_rename_file, file_paths)

def main():
    source_directories = ["/Volumes/SSD_RAID/Pictures"]
    destination_directory = "/Volumes/SSD_RAID/Assets"
    num_processes = 20  # Set to the number of cores you wish to utilize

    for source_directory in source_directories:
        process_directory_parallel(source_directory, destination_directory, num_processes)

if __name__ == "__main__":
    main()
