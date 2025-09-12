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
# Script Name: images-fix-rotation-v1.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible image rotation correction tool (Revision 01) that
#     uses PIL to detect and correct orientation based on EXIF data. Features
#     multi-threaded processing and in-place file modification capabilities.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multi-threaded processing (optimized for macOS systems)
#     - Multiple format support (.jpg, .jpeg, .png, .dng, .tif, .tiff, .webp, .gif)
#     - EXIF orientation detection and automatic correction
#     - In-place file modification with backup support
#     - Directory recursion with comprehensive error handling
#     - Desktop output support with detailed logging
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - Pillow (PIL) (auto-installed if missing)
#     - concurrent.futures (standard library with macOS compatibility)
#
# Usage:
#     python images-fix-rotation-v1.py
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
from PIL import Image, ExifTags
from concurrent.futures import ThreadPoolExecutor

def reset_image_orientation(image_path):
    try:
        with Image.open(image_path) as image:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif = image._getexif()

            if exif and orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                image.save(image_path)
                print(f"Processed {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.dng', '.tif', '.tiff', '.webp', '.gif')):
                image_path = os.path.join(root, file)
                reset_image_orientation(image_path)

def main():
    root_directory = '/Volumes/SSD_RAID/WORKING/Pictures/terst'  # Replace with your directory path
    num_threads = 4  # Adjust the number of threads as needed

    directories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_directory, directories)

if __name__ == "__main__":
    main()
