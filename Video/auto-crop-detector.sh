#!/bin/bash
#
# auto-crop-detector - Automatic Video Crop Detection and Application
# ----------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Automatically detects and applies optimal cropping parameters to videos
#     using FFmpeg's cropdetect filter. Processes multiple video formats and
#     maintains original quality with intelligent border detection.
#
# Features:
#     - Automatic black border detection
#     - Multi-format support
#     - Hardware acceleration
#     - Batch processing
#     - Progress logging
#     - GUI directory selection
#
# Requirements:
#     - bash 4.0+
#     - ffmpeg with cropdetect filter
#     - zenity for GUI
#     - bc for calculations
#
# Usage:
#     ./auto-crop-detector.sh
#

# Ensure script fails on errors
set -euo pipefail

# Function to show GUI dialog
show_dialog() {
    local title="$1"
    local text="$2"
    zenity --info --title="$title" --text="$text" 2>/dev/null || true
}

# Function to show error dialog
show_error() {
    local text="$1"
    zenity --error --text="$text" 2>/dev/null || true
}

# Function to select directory
select_directory() {
    local title="$1"
    zenity --file-selection --directory --title="$title" 2>/dev/null || true
}

# GUI for folder selection
SOURCE_DIR=$(select_directory "Select the Source Folder")
if [ -z "$SOURCE_DIR" ]; then
    show_error "No source directory selected. Exiting."
    exit 1
fi

OUTPUT_DIR=$(select_directory "Select the Output Folder")
if [ -z "$OUTPUT_DIR" ]; then
    show_error "No output directory selected. Exiting."
    exit 1
fi

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/crop_log.txt"
echo "Starting crop detection at $(date)" > "$LOG_FILE"

# Supported video formats
SUPPORTED_FORMATS=("mp4" "avi" "mov" "mkv" "m4v")

# Function to detect crop parameters
detect_crop() {
    local file="$1"
    local duration
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    local analysis_duration
    analysis_duration=$(echo "$duration * 0.1" | bc)  # Analyze 10% of the video

    # Detect crop parameters using multiple sample points
    local crop_params
    crop_params=$(ffmpeg -i "$file" -t "$analysis_duration" -vf "cropdetect=round=2:skip=1:reset=1" -f null - 2>&1 | \
                 awk '/crop=/ {print $NF}' | sort | uniq -c | sort -nr | head -n1 | awk '{print $2}')

    echo "Detected crop params for $file: $crop_params" >> "$LOG_FILE"
    echo "$crop_params"
}

# Function to crop video
crop_video() {
    local file="$1"
    local crop_params="$2"
    local basename
    basename=$(basename -- "$file")
    local extension="${file##*.}"
    local base="${basename%.*}"
    local counter=1
    local output_file="$OUTPUT_DIR/${base}_cropped.$extension"

    # Handle file collisions
    while [ -f "$output_file" ]; do
        output_file="$OUTPUT_DIR/${base}_cropped-${counter}.$extension"
        ((counter++))
    done

    if [ -z "$crop_params" ]; then
        echo "No crop parameters detected for $file. Skipping crop." >> "$LOG_FILE"
        return
    fi

    echo "Processing $file with crop parameters: $crop_params" >> "$LOG_FILE"

    # Use hardware acceleration if available
    ffmpeg -hwaccel auto -i "$file" \
           -vf "crop=$crop_params" \
           -c:v libx264 -preset medium -crf 18 \
           -c:a copy \
           -movflags +faststart \
           "$output_file" 2>> "$LOG_FILE"

    echo "Processed: $file -> $output_file" >> "$LOG_FILE"
}

# Export functions and variables
export -f detect_crop crop_video
export SOURCE_DIR OUTPUT_DIR LOG_FILE

# Main processing loop with progress indication
total_files=0
for format in "${SUPPORTED_FORMATS[@]}"; do
    count=$(find "$SOURCE_DIR" -type f -iname "*.$format" | wc -l)
    total_files=$((total_files + count))
done

if [ "$total_files" -eq 0 ]; then
    show_error "No supported video files found in source directory."
    exit 1
fi

processed=0
(
for format in "${SUPPORTED_FORMATS[@]}"; do
    while IFS= read -r -d '' file; do
        crop_params=$(detect_crop "$file")
        crop_video "$file" "$crop_params"
        processed=$((processed + 1))
        echo "$((processed * 100 / total_files))"
        echo "# Processing: $processed of $total_files files"
    done < <(find "$SOURCE_DIR" -type f -iname "*.$format" -print0)
done
) | zenity --progress \
    --title="Processing Videos" \
    --text="Starting video processing..." \
    --percentage=0 \
    --auto-close \
    --auto-kill

# Show completion message
show_dialog "Processing Complete" "Video processing complete.\nCheck $LOG_FILE for details."

echo "Processing complete at $(date)" >> "$LOG_FILE"
