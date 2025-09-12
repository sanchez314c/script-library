#!/usr/bin/env bash
"""
Video Audio Extractor
-------------------
Author: sanchez314c@speedheathens.com
Date: 2025-01-24
Version: 1.0.1

Description:
    Extracts high-quality audio from video files using FFmpeg. Supports
    multiple input formats and provides options for output format and
    quality settings.

Features:
    - High-quality audio extraction
    - Multiple format support
    - Batch processing
    - Progress tracking
    - Error handling
    - Format conversion

Requirements:
    - ffmpeg with audio codecs
    - zenity for GUI
    - bash 4.0+

Usage:
    ./video-extract-audio.sh
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

# Function to select audio format
select_format() {
    zenity --list \
        --title="Select Output Format" \
        --column="Format" \
        --height=300 \
        "mp3" \
        "aac" \
        "wav" \
        "flac" \
        2>/dev/null || echo "mp3"
}

# GUI for folder selection
SOURCE_DIR=$(select_directory "Select Source Video Folder")
if [ -z "$SOURCE_DIR" ]; then
    show_error "No source directory selected. Exiting."
    exit 1
fi

OUTPUT_DIR=$(select_directory "Select Output Audio Folder")
if [ -z "$OUTPUT_DIR" ]; then
    show_error "No output directory selected. Exiting."
    exit 1
fi

# Select output format
FORMAT=$(select_format)

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/extraction_log.txt"
echo "Starting audio extraction at $(date)" > "$LOG_FILE"

# Function to extract audio
extract_audio() {
    local video_file="$1"
    local output_dir="$2"
    local format="$3"
    local basename
    basename=$(basename "$video_file")
    local filename="${basename%.*}"
    local output_file="$output_dir/${filename}.$format"
    local counter=1

    # Handle file collisions
    while [ -f "$output_file" ]; do
        output_file="$output_dir/${filename}_${counter}.$format"
        ((counter++))
    done

    # Select codec and quality based on format
    local codec_params=()
    case "$format" in
        "mp3")
            codec_params=(-codec:a libmp3lame -q:a 0)
            ;;
        "aac")
            codec_params=(-codec:a aac -b:a 256k)
            ;;
        "wav")
            codec_params=(-codec:a pcm_s16le)
            ;;
        "flac")
            codec_params=(-codec:a flac -compression_level 8)
            ;;
        *)
            echo "Unsupported format: $format" >> "$LOG_FILE"
            return 1
            ;;
    esac

    echo "Processing: $basename" >> "$LOG_FILE"

    if ffmpeg -i "$video_file" "${codec_params[@]}" "$output_file" 2>> "$LOG_FILE"; then
        echo "Successfully extracted: $output_file" >> "$LOG_FILE"
        return 0
    else
        echo "Failed to extract: $basename" >> "$LOG_FILE"
        return 1
    fi
}

# Export functions and variables
export -f extract_audio
export OUTPUT_DIR FORMAT LOG_FILE

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
        if extract_audio "$video_file" "$OUTPUT_DIR" "$FORMAT"; then
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
    --title="Extracting Audio" \
    --text="Starting audio extraction..." \
    --percentage=0 \
    --auto-close \
    --auto-kill

# Show completion message
completion_message="Audio extraction complete:
- Total files: $total_files
- Successful: $success
- Failed: $failed
- Output format: $FORMAT
- Log file: $LOG_FILE"

show_dialog "Extraction Complete" "$completion_message"

echo "Processing complete at $(date)" >> "$LOG_FILE"
echo "$completion_message" >> "$LOG_FILE"

exit 0
