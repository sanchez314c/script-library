#!/bin/bash
#
# gopro-metadata-fix - GoPro Video Metadata Repair Tool
# ----------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Fixes metadata issues in GoPro video files by re-encoding with proper
#     handler names and metadata structures. Preserves video quality while
#     ensuring compatibility with various media players and editing software.
#
# Features:
#     - GoPro-specific metadata repair
#     - Codec preservation (copy mode)
#     - Proper handler name assignment
#     - Quality preservation
#     - Batch processing support
#
# Requirements:
#     - bash 4.0+
#     - ffmpeg with metadata support
#     - Read/write access to video files
#
# Usage:
#     ./gopro-metadata-fix.sh input_file.mp4 [output_file.mp4]
#

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 input_file.mp4 [output_file.mp4]"
    echo "Example: $0 GH014444.mp4 GH014444-fixed.mp4"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.*}-fixed.${INPUT_FILE##*.}}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "Processing GoPro metadata fix..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

# Apply GoPro metadata fix
ffmpeg -i "$INPUT_FILE" \
       -map_metadata -1 \
       -codec copy \
       -metadata:s handler='GoPro AVC encoder' \
       -metadata:s handler_name='GoPro AVC' \
       "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "GoPro metadata fix completed successfully!"
    echo "Fixed file: $OUTPUT_FILE"
else
    echo "Error: Failed to process GoPro metadata fix."
    exit 1
fi