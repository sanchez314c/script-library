#!/usr/bin/env bash
"""
Video Crop to 16:9 Aspect Ratio
-----------------------------
Author: sanchez314c@speedheathens.com
Date: 2025-01-24
Version: 1.0.1

Description:
    Automatically crops videos to 16:9 aspect ratio while maintaining
    maximum quality and resolution. Supports multiple input formats
    and provides options for output customization.

Features:
    - Intelligent 16:9 cropping
    - Quality preservation
    - Hardware acceleration
    - Batch processing
    - Progress tracking
    - Error handling

Requirements:
    - ffmpeg with libx264
    - zenity for GUI
    - bash 4.0+

Usage:
    ./video-crop-to-16x9.sh
"""

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
SOURCE_DIR=$(select_directory "Select Source Video Folder")
if [ -z "$SOURCE_DIR" ]; then
    show_error "No source directory selected. Exiting."
    exit 1
fi

OUTPUT_DIR=$(select_directory "Select Output Video Folder")
if [ -z "$OUTPUT_DIR" ]; then
    show_error "No output directory selected. Exiting."
    exit 1
fi

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/crop_log.txt"
echo "Starting 16:9 crop processing at $(date)" > "$LOG_FILE"

# Function to get video dimensions
get_video_dimensions() {
    local video_file="$1"
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$video_file"
}

# Function to crop video to 16:9
crop_video() {
    local video_file="$1"
    local output_dir="$2"
    local basename
    basename=$(basename "$video_file")
    local filename="${basename%.*}"
    local output_file="$output_dir/${filename}_16x9.mp4"
    local counter=1

    # Handle file collisions
    while [ -f "$output_file" ]; do
        output_file="$output_dir/${filename}_16x9_${counter}.mp4"
        ((counter++))
    done

    echo "Processing: $basename" >> "$LOG_FILE"

    # Get video dimensions
    local dimensions
    dimensions=$(get_video_dimensions "$video_file")
    local width height
    IFS='x' read -r width height <<< "$dimensions"

    # Calculate crop dimensions for 16:9
    local target_width=$width
    local target_height=$((width * 9 / 16))
    local crop_y=$(((height - target_height) / 2))

    if [ "$target_height" -gt "$height" ]; then
        target_height=$height
        target_width=$((height * 16 / 9))
        crop_y=0
    fi

    # Crop and encode
    if ffmpeg -hwaccel auto -i "$video_file" \
            -vf "crop=$target_width:$target_height:0:$crop_y" \
            -c:v libx264 -preset slow -crf 18 \
            -c:a copy \
            -movflags +faststart \
            "$output_file" 2>> "$LOG_FILE"; then
        echo "Successfully cropped: $output_file" >> "$LOG_FILE"
        return 0
    else
        echo "Failed to crop: $basename" >> "$LOG_FILE"
        return 1
    fi
}

# Export functions and variables
export -f crop_video get_video_dimensions
export OUTPUT_DIR LOG_FILE

# Count total files
total_files=0
for ext in mp4 avi mov mkv m4v mpg mpeg wmv; do
    count=$(find "$SOURCE_DIR" -type f -iname "*.$ext" | wc -l)
    total_files=$((total_files + count))
done

if [ "$total_files" -eq 0 ]; then
    show_error "No supported video files found in source directory."
    exit 1
fi

# Process files with progress bar
processed=0
success=0
failed=0

(
for ext in mp4 avi mov mkv m4v mpg mpeg wmv; do
    while IFS= read -r -d '' video_file; do
        if crop_video "$video_file" "$OUTPUT_DIR"; then
            ((success++))
        else
            ((failed++))
        fi
        ((processed++))
        echo "$((processed * 100 / total_files))"
        echo "# Processing: $processed of $total_files files"
    done < <(find "$SOURCE_DIR" -type f -iname "*.$ext" -print0)
done
) | zenity --progress \
    --title="Cropping Videos" \
    --text="Starting video cropping..." \
    --percentage=0 \
    --auto-close \
    --auto-kill

# Show completion message
completion_message="Video cropping complete:
- Total files: $total_files
- Successful: $success
- Failed: $failed
- Log file: $LOG_FILE"

show_dialog "Cropping Complete" "$completion_message"

echo "Processing complete at $(date)" >> "$LOG_FILE"
echo "$completion_message" >> "$LOG_FILE"

exit 0
