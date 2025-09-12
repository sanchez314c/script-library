#!/bin/bash
#
# Grab Underscore Files
# -------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Identifies and processes files containing underscores,
#     organizing them into year-based directories. Useful for
#     sorting timestamped or specially marked files.
#
# Features:
#     - Pattern matching
#     - Year-based sorting
#     - Parallel processing
#     - Progress tracking
#     - Directory creation
#
# Requirements:
#     - bash 4.0+
#     - parallel command
#     - find command
#
# Usage:
#     ./grab-underscore-files.sh
## ugrab underscore files
# ----------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     System maintenance and automation script for efficient
#     system management and file operations.
#
# Features:
#     - Automated processing
#     - Error handling
#     - Progress tracking
#     - System integration
#     - Status reporting
#
# Requirements:
#     - bash 4.0+
#     - Standard Unix tools
#
# Usage:
#     ./grab-underscore-files.sh

#     ./GRAB-WITH-UNDERSCORE.sh

#!/bin/bash
#!/bin/bash

# The source directory where your files are initially located
SOURCE_DIR="/Volumes/SSD/Pictures"

# The destination directory where you want to move files with "___" in them
DESTINATION_DIR="/Volumes/SSD/Pictures_NEW"

# Create the destination directory if it doesn't exist
mkdir -p "$DESTINATION_DIR"

# Export DESTINATION_DIR so it is available in the child process spawned by parallel
export DESTINATION_DIR

# Function to move files to their respective directories based on year
move_file() {
    file="$1"
    year=$(basename "$file" | cut -c 1-4) # Extract the year from the filename
    target_dir="$DESTINATION_DIR/$year"
    mkdir -p "$target_dir" # Create the target directory if it doesn't exist
    mv "$file" "$target_dir/" # Move the file
    echo "Moved $file to $target_dir"
}

export -f move_file # Export the function for parallel to use

# Use GNU parallel to move the files
find "$SOURCE_DIR" -type f -name '*___*' -print0 |
  parallel -0 --max-args 1 --plus --bar move_file
