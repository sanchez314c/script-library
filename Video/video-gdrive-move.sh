#!/usr/bin/env bash
"""
Google Drive Video Move Utility
----------------------------
Author: sanchez314c@speedheathens.com
Date: 2025-01-24
Version: 1.0.1

Description:
    Moves video files to Google Drive using rclone, with support for
    multiple formats and automatic organization. Includes progress
    tracking and error handling.

Features:
    - Secure Google Drive integration
    - Multiple format support
    - Batch processing
    - Progress tracking
    - Error handling
    - Bandwidth management

Requirements:
    - rclone with Google Drive config
    - zenity for GUI
    - bash 4.0+

Usage:
    ./video-gdrive-move.sh
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

# Function to select Google Drive remote
select_remote() {
    local remotes
    remotes=$(rclone listremotes)
    if [ -z "$remotes" ]; then
        show_error "No rclone remotes configured. Please configure Google Drive in rclone first."
        exit 1
    fi
    
    echo "$remotes" | zenity --list \
        --title="Select Google Drive Remote" \
        --column="Remote" \
        --height=300 \
        2>/dev/null || echo "${remotes%%:*}"
}

# GUI for folder selection
SOURCE_DIR=$(select_directory "Select Source Video Folder")
if [ -z "$SOURCE_DIR" ]; then
    show_error "No source directory selected. Exiting."
    exit 1
fi

# Select Google Drive remote
REMOTE=$(select_remote)
if [ -z "$REMOTE" ]; then
    show_error "No remote selected. Exiting."
    exit 1
fi

# Create log file
LOG_FILE="/tmp/gdrive_move_$$.log"
echo "Starting Google Drive move at $(date)" > "$LOG_FILE"

# Function to move file to Google Drive
move_to_gdrive() {
    local file="$1"
    local remote="$2"
    local basename
    basename=$(basename "$file")
    local year_month
    year_month=$(date -r "$file" +"%Y-%m")
    local remote_path="${remote}Videos/${year_month}/"

    echo "Moving: $basename to $remote_path" >> "$LOG_FILE"

    if rclone move "$file" "$remote_path" \
            --progress \
            --stats-one-line \
            --stats 1s \
            --transfers 2 \
            --checkers 4 \
            --drive-chunk-size 32M \
            --buffer-size 64M \
            2>> "$LOG_FILE"; then
        echo "Successfully moved: $basename" >> "$LOG_FILE"
        return 0
    else
        echo "Failed to move: $basename" >> "$LOG_FILE"
        return 1
    fi
}

# Export functions and variables
export -f move_to_gdrive
export REMOTE LOG_FILE

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
        if move_to_gdrive "$video_file" "$REMOTE"; then
            ((success++))
        else
            ((failed++))
        fi
        ((processed++))
        echo "$((processed * 100 / total_files))"
        echo "# Moving: $processed of $total_files files"
    done < <(find "$SOURCE_DIR" -type f -iname "*.$ext" -print0)
done
) | zenity --progress \
    --title="Moving to Google Drive" \
    --text="Starting file transfer..." \
    --percentage=0 \
    --auto-close \
    --auto-kill

# Show completion message
completion_message="File transfer complete:
- Total files: $total_files
- Successful: $success
- Failed: $failed
- Log file: $LOG_FILE"

show_dialog "Transfer Complete" "$completion_message"

echo "Processing complete at $(date)" >> "$LOG_FILE"
echo "$completion_message" >> "$LOG_FILE"

exit 0
