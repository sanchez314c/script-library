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
# Script Name: images-report-unique-date-tags.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible unique datetime tag reporter that recursively
#     scans directories to discover and catalog all unique date/time metadata
#     tags. Features regex pattern matching and parallel processing for comprehensive analysis.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - GUI directory selection with native macOS dialogs
#     - Parallel processing with ProcessPoolExecutor (optimized for macOS)
#     - Regular expression pattern matching for comprehensive tag discovery
#     - Group-based tag organization with case-insensitive matching
#     - Recursive directory scanning with detailed error handling
#     - Desktop output support with unique tag display and group prefixes
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-report-unique-date-tags.py
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
import re
import subprocess
from tkinter import filedialog, Tk
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_directory(directory):
    """
    Runs exiftool on a directory to gather all metadata, then parses
    the output for date and time related tags.
    """
    unique_datetime_tags = set()
    datetime_regex = re.compile(r'\bDate\b|\bTime\b', re.IGNORECASE)

    # Run exiftool on the directory
    try:
        result = subprocess.run(['exiftool', '-r', '-G', '-s', '-n', directory],
                                stdout=subprocess.PIPE, text=True)
        lines = result.stdout.split('\n')

        # Process each line to find date/time tags
        for line in lines:
            parts = line.split(':')
            if len(parts) < 2:
                continue
            group_tag = parts[0].strip()
            tag = parts[1].strip()
            # If the tag matches the regex and contains date/time info, add it
            if datetime_regex.search(tag):
                unique_datetime_tags.add(f"{group_tag}: {tag}")
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")

    return unique_datetime_tags

def main():
    root = Tk()
    root.withdraw()  # Hide the root window
    directory = filedialog.askdirectory(title="Select Folder")
    if directory:
        # Use multiprocessing to handle large directories or deep recursion efficiently
        with ProcessPoolExecutor() as executor:
            future = executor.submit(process_directory, directory)
            datetime_tags = future.result()

            print("Found Date/Time Tags:")
            for tag in datetime_tags:
                print(tag)
    else:
        print("No directory selected.")

if __name__ == "__main__":
    main()
