#!/bin/bash
#
# rename-by-oldest-date - Batch Rename Files Using Metadata Dates
# ------------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: March 3, 2025
# Version: 1.1.0
#
# Description:
#     Renames media files using their oldest available metadata date
#     (creation date, original date, or modification date) in a format
#     of YYYYMMDD_HHMMSS with collision handling for identical timestamps.
#
# Features:
#     - Metadata-based date extraction
#     - Priority-based date selection
#     - Collision detection and handling
#     - Progress tracking
#     - File extension flexibility
#     - Metadata preservation
#
# Requirements:
#     - bash 4.0+
#     - exiftool
#     - macOS file selection dialog (optional)
#
# Usage:
#     ./rename-by-oldest-date.sh [directory] [extension]
#

# Enable error handling
set -e

# Display error messages
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Display progress with timestamps
log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if exiftool is installed
if ! command -v exiftool &> /dev/null; then
    error_exit "exiftool is not installed. Please install exiftool and try again."
fi

# Get directory from arguments or use dialog for selection
get_directory() {
    local target_dir="$1"
    
    if [ -z "$target_dir" ]; then
        # If running in GUI environment, use native dialog
        if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
            target_dir=$(osascript -e 'tell application "Finder" to return POSIX path of (choose folder with prompt "Select Folder with Files to Rename")')
        else
            # Fallback to text prompt
            read -p "Enter directory path containing files to rename: " target_dir
        fi
    fi
    
    # Validate directory
    if [ ! -d "$target_dir" ]; then
        error_exit "Directory does not exist: $target_dir"
    fi
    
    echo "$target_dir"
}

# Get file extension to process
get_extension() {
    local ext="$1"
    
    if [ -z "$ext" ]; then
        # If running in GUI environment, use native dialog
        if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
            ext=$(osascript -e 'set the_result to text returned of (display dialog "Enter file extension to process (without dot, e.g. jpg):" default answer "jpeg")')
        else
            # Fallback to text prompt
            read -p "Enter file extension to process (without dot, e.g. jpg): " ext
            if [ -z "$ext" ]; then
                ext="jpeg"
            fi
        fi
    fi
    
    echo "$ext"
}

# Rename files based on metadata dates
rename_files() {
    local target_dir="$1"
    local extension="$2"
    local total_files=0
    local renamed_files=0
    local skipped_files=0
    local failed_files=0
    
    # Count total files
    total_files=$(find "$target_dir" -maxdepth 1 -type f -name "*.${extension}" | wc -l)
    log_progress "Found $total_files files with extension .$extension"
    
    # Change to target directory
    cd "$target_dir"
    
    # Process each file
    local counter=0
    for file in *."$extension"; do
        # Skip if no matching files
        if [ "$file" = "*.$extension" ]; then
            log_progress "No files with extension .$extension found in directory"
            return
        fi
        
        counter=$((counter + 1))
        log_progress "Processing file $counter of $total_files: $file"
        
        # Skip files that are already correctly named
        if [[ "$file" =~ ^[0-9]{8}_[0-9]{6}(-[0-9]+)?\.${extension}$ ]]; then
            log_progress "Skipping $file (already named correctly)"
            skipped_files=$((skipped_files + 1))
            continue
        fi
        
        # Get the oldest date from metadata (prioritizing creation dates)
        oldest_date=$(exiftool -time:all -d "%Y:%m:%d %H:%M:%S" "$file" 2>/dev/null | grep -E "Date/Time Original|Create Date|Date Created" | head -1 | awk -F': ' '{print $2}')
        
        # If no date found, use file modification date
        if [ -z "$oldest_date" ]; then
            oldest_date=$(exiftool -FileModifyDate -d "%Y:%m:%d %H:%M:%S" "$file" 2>/dev/null | awk -F': ' '{print $2}')
            
            # If still no date, use system file date
            if [ -z "$oldest_date" ]; then
                oldest_date=$(date -r "$file" "+%Y:%m:%d %H:%M:%S")
            fi
        fi
        
        # Format the date for filename (YYYYMMDD_HHMMSS)
        formatted_date=$(echo "$oldest_date" | sed 's/[^0-9]//g' | cut -c1-14)
        
        # Handle case where date might be incomplete
        if [ ${#formatted_date} -lt 14 ]; then
            log_progress "Warning: Incomplete date for $file: $oldest_date"
            failed_files=$((failed_files + 1))
            continue
        fi
        
        year=${formatted_date:0:4}
        month=${formatted_date:4:2}
        day=${formatted_date:6:2}
        hour=${formatted_date:8:2}
        minute=${formatted_date:10:2}
        second=${formatted_date:12:2}
        
        # Create base new filename
        base_name="${year}${month}${day}_${hour}${minute}${second}"
        new_name="${base_name}.${extension}"
        
        # Handle duplicate filenames by adding suffix
        suffix_counter=1
        while [ -e "$new_name" ] && [ "$file" != "$new_name" ]; do
            new_name="${base_name}-${suffix_counter}.${extension}"
            suffix_counter=$((suffix_counter + 1))
        done
        
        # Skip if filename already matches
        if [[ "$file" == "$new_name" ]]; then
            log_progress "Skipping $file (already named correctly)"
            skipped_files=$((skipped_files + 1))
            continue
        fi
        
        # Use exiftool to rename the file while preserving all metadata
        if exiftool "-FileName=$new_name" -P "$file" 2>/dev/null; then
            log_progress "Renamed: $file → $new_name"
            renamed_files=$((renamed_files + 1))
        else
            log_progress "Failed to rename: $file"
            failed_files=$((failed_files + 1))
        fi
    done
    
    # Clean up exiftool backup files
    rm -f *_original
    
    # Report summary
    log_progress "Rename operation complete"
    log_progress "Total files processed: $total_files"
    log_progress "Successfully renamed: $renamed_files"
    log_progress "Skipped (already correct): $skipped_files"
    log_progress "Failed to rename: $failed_files"
}

# Main script execution
main() {
    log_progress "Starting file rename process based on oldest dates..."
    
    # Get directory and extension
    local target_dir=$(get_directory "$1")
    local extension=$(get_extension "$2")
    
    # Perform rename operation
    rename_files "$target_dir" "$extension"
    
    log_progress "All files have been renamed in place with preserved metadata."
}

# Run the main function with arguments
main "$@"