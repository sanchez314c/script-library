#!/bin/bash
#
# Grab Random 100GB
# ----------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Randomly selects files up to 100GB total size from specified
#     source directories. Supports multiple file types and ensures
#     random distribution.
#
# Features:
#     - Size-based selection
#     - Random file picking
#     - Multiple source support
#     - Progress tracking
#     - Type filtering
#
# Requirements:
#     - bash 4.0+
#     - find command
#     - du command
#     - shuf command
#
# Usage:
#     ./grab-random-100gb.sh
## ugrab random 100gb
# ------------------
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
#     ./grab-random-100gb.sh

#     ./GRAB-100GB-RANDOM-FILES.sh

#!/bin/bash

SOURCE_DIR_1="/Volumes/RECOVER"
SOURCE_DIR_2="/Volumes/SSD/Pictures/All Assets"
DEST_DIR="/Volumes/SSD/Pictures/100GB Sample"
SIZE_LIMIT="100G"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Define a comprehensive list of image file extensions
IMAGE_EXTENSIONS=(
  "*.jpg" "*.jpeg" "*.png" "*.tiff" "*.tif"
  "*.cr2" "*.dng" "*.nef" "*.raf" "*.orf"
  "*.raw" "*.arw" "*.sr2" "*.srf" "*.rw2"
)

# Convert the array of extensions into a find command argument
FIND_ARGS=()
for ext in "${IMAGE_EXTENSIONS[@]}"; do
  FIND_ARGS+=(-o -iname "$ext")
done

# Find all image files in both source directories, shuffle the list,
# and copy until the size limit is reached
find "$SOURCE_DIR_1" "$SOURCE_DIR_2" -type f \( "${FIND_ARGS[@]:1}" \) -print0 |
  shuf --random-source=/dev/urandom -z |
  rsync -0 --files-from=- --copy-links --size-only --no-relative --recursive --max-size="$SIZE_LIMIT" "$SOURCE_DIR_1/" "$SOURCE_DIR_2/" "$DEST_DIR/"
