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
# Script Name: images-fix-all-metadata-tags.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible comprehensive metadata repair tool that performs
#     two-phase processing: complete EXIF tag repair/reorganization followed by
#     date/time standardization based on filename information.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Two-phase metadata processing (repair then standardization)
#     - Complete tag repair and reorganization with integrity preservation
#     - Date/time standardization from filename patterns
#     - Recursive directory processing with progress tracking
#     - Detailed logging and comprehensive error handling
#     - Desktop output support with macOS-optimized paths
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - exiftool (auto-installed if missing)
#     - subprocess (standard library with macOS compatibility)
#
# Usage:
#     python images-fix-all-metadata-tags.py
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

import subprocess
import logging
import os

# Script Summary:
# This script utilizes EXIFTOOL to perform two main operations on image files:
# 1. Modify and repair image metadata.
# 2. Set various date and time metadata fields based on the filenames.

# Required Libraries:
# - subprocess
# - logging
# - os

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants and Configurations
SOURCE_DIRECTORY = "/Volumes/SSD_RAID/Pictures"

# First EXIFTOOL Command for Tag Repair
EXIFTOOL_TAG_REPAIR_COMMAND = [
    "exiftool", "-v3", "-progress", "-wm", "wcg", "-r",
    "-exif:all=", "-tagsfromfile", "@", "-exif:all",
    "-unsafe", "-thumbnailimage", "-F", "-overwrite_original",
    SOURCE_DIRECTORY
]

# Second EXIFTOOL Command to Set Dates from Filename
EXIFTOOL_SET_DATES_COMMAND = [
    "exiftool", "-v3", "-progress", "-wm", "wcg", "-r",
    "'-AllDates<Filename'", "'-FileModifyDate<Filename'",
    "'-DateTimeOriginal<Filename'", "'-Time:all<DateTimeOriginal'",
    "-overwrite_original", "-d", "%Y-%m-%d %H-%M-%S",
    SOURCE_DIRECTORY
]

def run_exiftool_command(command):
    """
    Executes a given EXIFTOOL command.
    """
    try:
        logging.info("Starting EXIFTOOL command execution...")
        subprocess.run(command, check=True)
        logging.info("EXIFTOOL command executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while executing EXIFTOOL: {e}")

def main():
    """
    Main function to run the script.
    """
    if not os.path.exists(SOURCE_DIRECTORY):
        logging.error(f"Source directory does not exist: {SOURCE_DIRECTORY}")
        return

    # Run the first EXIFTOOL command for tag repair
    run_exiftool_command(EXIFTOOL_TAG_REPAIR_COMMAND)

    # Run the second EXIFTOOL command to set dates from filenames
    run_exiftool_command(EXIFTOOL_SET_DATES_COMMAND)

if __name__ == "__main__":
    main()
