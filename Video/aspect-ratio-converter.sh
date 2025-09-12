#!/bin/bash
#
# aspect-ratio-converter - 4:3 to 16:9 Aspect Ratio Converter
# ----------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Converts videos from 4:3 aspect ratio to 16:9 using intelligent cropping
#     and scaling. Maintains quality while adapting to modern display standards.
#
# Features:
#     - Intelligent aspect ratio conversion
#     - Hardware acceleration support
#     - Quality preservation
#     - Multi-core batch processing
#     - Progress tracking with time estimation
#     - Detailed error logging and reporting
#     - Handles videos with non-standard dimensions
#
# Requirements:
#     - ffmpeg with libx264
#     - zenity for GUI
#     - bash 4.0+
#
# Usage:
#     ./aspect-ratio-converter.sh

# Ensure script fails on errors
set -euo pipefail

# Check for required tools
check_requirements() {
    local missing=()
    
    if ! command -v ffmpeg &>/dev/null; then
        missing+=("ffmpeg")
    fi
    
    if ! command -v zenity &>/dev/null; then
        missing+=("zenity")
    fi
    
    if ! command -v parallel &>/dev/null; then
        missing+=("parallel (GNU parallel)")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: Missing required tools: ${missing[*]}"
        echo "Please install them and try again."
        echo "For macOS, use: brew install ffmpeg zenity parallel"
        exit 1
    fi
}

# Function to show GUI dialog
show_dialog() {
    local title="$1"
    local text="$2"
    zenity --info --title="$title" --text="$text" --width=400 2>/dev/null || true
}

# Function to show error dialog
show_error() {
    local text="$1"
    zenity --error --text="$text" --width=400 2>/dev/null || true
}

# Function to show question dialog
show_question() {
    local title="$1"
    local text="$2"
    zenity --question --title="$title" --text="$text" --width=400 2>/dev/null
    return $?
}

# Function to select directory
select_directory() {
    local title="$1"
    zenity --file-selection --directory --title="$title" 2>/dev/null || true
}

# Function to get user preferences
get_preferences() {
    # Use zenity forms to get user preferences
    local prefs
    prefs=$(zenity --forms --title="Conversion Preferences" \
        --text="Set conversion preferences:" \
        --add-combo="Video Quality:" --combo-values="High|Medium|Low" \
        --add-combo="Use Hardware Acceleration:" --combo-values="Yes|No" \
        --add-combo="Process Subdirectories:" --combo-values="Yes|No" \
        --add-combo="Output Format:" --combo-values="Same as input|MP4|MKV" \
        --width=500 2>/dev/null)
    
    if [ -z "$prefs" ]; then
        # User canceled, use defaults
        QUALITY_PRESET="high"
        USE_HARDWARE_ACCEL="yes"
        PROCESS_SUBDIRS="no"
        OUTPUT_FORMAT="same"
    else
        # Parse preferences
        IFS='|' read -r quality hw_accel subdirs format <<< "$prefs"
        
        case "${quality,,}" in
            "high") QUALITY_PRESET="high" ;;
            "medium") QUALITY_PRESET="medium" ;;
            "low") QUALITY_PRESET="low" ;;
            *) QUALITY_PRESET="high" ;;
        esac
        
        USE_HARDWARE_ACCEL="${hw_accel,,}"
        PROCESS_SUBDIRS="${subdirs,,}"
        
        case "${format,,}" in
            "mp4") OUTPUT_FORMAT="mp4" ;;
            "mkv") OUTPUT_FORMAT="mkv" ;;
            *) OUTPUT_FORMAT="same" ;;
        esac
    fi
    
    # Set ffmpeg options based on quality preset
    case "$QUALITY_PRESET" in
        "high")
            ENCODER_PRESET="slow"
            CRF_VALUE="18"
            ;;
        "medium")
            ENCODER_PRESET="medium"
            CRF_VALUE="22"
            ;;
        "low")
            ENCODER_PRESET="fast"
            CRF_VALUE="26"
            ;;
    esac
}

# Function to convert aspect ratio
convert_aspect_ratio() {
    local file="$1"
    local basename
    basename=$(basename -- "$file")
    local extension="${file##*.}"
    local base="${basename%.*}"
    local counter=1
    
    # Determine output format
    local output_ext="$extension"
    if [ "$OUTPUT_FORMAT" != "same" ]; then
        output_ext="$OUTPUT_FORMAT"
    fi
    
    local output_file="$OUTPUT_DIR/${base}_16x9.$output_ext"

    # Handle file collisions
    while [ -f "$output_file" ]; do
        output_file="$OUTPUT_DIR/${base}_16x9-${counter}.$output_ext"
        ((counter++))
    done

    # Get video dimensions
    local video_info
    video_info=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$file" 2>/dev/null)
    
    # Check if ffprobe succeeded
    if [ -z "$video_info" ]; then
        echo "Error: Could not get dimensions for $file" >> "$LOG_FILE"
        return 1
    fi
    
    local width
    local height
    IFS=',' read -r width height <<< "$video_info"
    
    # Calculate aspect ratio
    local ratio=$(bc -l <<< "scale=2; $width / $height")
    local current_ar=$(printf "%.2f" "$ratio")
    
    # Determine if we need letterboxing or cropping
    local filter_complex
    local method
    
    # If it's already close to 16:9 (1.78), just scale
    if (( $(echo "$current_ar > 1.7 && $current_ar < 1.85" | bc -l) )); then
        filter_complex="scale=-1:1080"
        method="scale"
    # If it's 4:3 (1.33) or similar, crop centered
    elif (( $(echo "$current_ar < 1.7" | bc -l) )); then
        # Calculate new dimensions for 16:9
        local new_width=$(bc -l <<< "scale=0; $height * 16 / 9 / 1")
        local crop_x=$(bc -l <<< "scale=0; ($width - $new_width) / 2 / 1")
        
        # Handle edge cases - if new width > original width, we need to pad
        if (( $(echo "$new_width > $width" | bc -l) )); then
            local pad_width=$(bc -l <<< "scale=0; $height * 16 / 9 / 1")
            local pad_x=$(bc -l <<< "scale=0; ($pad_width - $width) / 2 / 1")
            filter_complex="pad=$pad_width:$height:$pad_x:0:black"
            method="pad"
        else
            filter_complex="crop=$new_width:$height:$crop_x:0"
            method="crop"
        fi
    # If it's wider than 16:9, scale and letterbox
    else
        local new_height=$(bc -l <<< "scale=0; $width * 9 / 16 / 1")
        local pad_y=$(bc -l <<< "scale=0; ($height - $new_height) / 2 / 1")
        filter_complex="crop=$width:$new_height:0:$pad_y"
        method="crop_height"
    fi
    
    # Log the processing details
    {
        echo "Processing $file"
        echo "Original dimensions: ${width}x${height} (AR: $current_ar)"
        echo "Conversion method: $method"
        echo "Filter: $filter_complex"
    } >> "$LOG_FILE"
    
    # Set hardware acceleration flag if enabled
    local hw_flag=""
    if [ "$USE_HARDWARE_ACCEL" = "yes" ]; then
        hw_flag="-hwaccel auto"
    fi
    
    # Convert using settings based on user preferences
    if ffmpeg $hw_flag -i "$file" \
           -vf "$filter_complex" \
           -c:v libx264 -preset "$ENCODER_PRESET" -crf "$CRF_VALUE" \
           -c:a copy \
           -movflags +faststart \
           "$output_file" 2>> "$LOG_FILE"; then
        
        echo "Converted: $file -> $output_file" >> "$LOG_FILE"
        return 0
    else
        echo "Error converting $file" >> "$LOG_FILE"
        return 1
    fi
}

# Function to find video files
find_video_files() {
    local find_opts="-type f"
    
    # Add recursive flag if needed
    if [ "$PROCESS_SUBDIRS" = "no" ]; then
        find_opts="$find_opts -maxdepth 1"
    fi
    
    # Find supported video formats
    local formats_pattern=$(printf "\\( -iname \"*.%s\" \\) -o " "${SUPPORTED_FORMATS[@]}")
    formats_pattern=${formats_pattern% -o }
    
    find "$SOURCE_DIR" $find_opts $formats_pattern | sort
}

# Main function
main() {
    # Check for required tools
    check_requirements
    
    # Supported video formats
    SUPPORTED_FORMATS=("mp4" "avi" "mov" "mkv" "m4v" "mpg" "mpeg" "wmv")
    
    # Get user preferences
    get_preferences
    
    # GUI for folder selection
    SOURCE_DIR=$(select_directory "Select Source Folder (4:3 Videos)")
    if [ -z "$SOURCE_DIR" ]; then
        show_error "No source directory selected. Exiting."
        exit 1
    fi
    
    OUTPUT_DIR=$(select_directory "Select Output Folder (16:9 Videos)")
    if [ -z "$OUTPUT_DIR" ]; then
        show_error "No output directory selected. Exiting."
        exit 1
    fi
    
    # Create output directory and log file
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="$OUTPUT_DIR/conversion_log.txt"
    echo "Starting aspect ratio conversion at $(date)" > "$LOG_FILE"
    echo "Quality preset: $QUALITY_PRESET" >> "$LOG_FILE"
    echo "Hardware acceleration: $USE_HARDWARE_ACCEL" >> "$LOG_FILE"
    echo "Process subdirectories: $PROCESS_SUBDIRS" >> "$LOG_FILE"
    echo "Output format: $OUTPUT_FORMAT" >> "$LOG_FILE"
    echo "-------------------------------------------" >> "$LOG_FILE"
    
    # Find all video files
    mapfile -t VIDEO_FILES < <(find_video_files)
    total_files=${#VIDEO_FILES[@]}
    
    if [ "$total_files" -eq 0 ]; then
        show_error "No supported video files found in source directory."
        exit 1
    fi
    
    # Confirm with the user
    if ! show_question "Confirm Conversion" "Found $total_files video files to convert.\nQuality: $QUALITY_PRESET\nProcess subdirectories: $PROCESS_SUBDIRS\n\nProceed with conversion?"; then
        echo "Conversion canceled by user." >> "$LOG_FILE"
        show_dialog "Conversion Canceled" "Operation canceled by user."
        exit 0
    fi
    
    # Determine optimal number of jobs for parallel processing
    # Use number of CPU cores
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # Export variables and functions for parallel
    export -f convert_aspect_ratio
    export LOG_FILE OUTPUT_DIR ENCODER_PRESET CRF_VALUE USE_HARDWARE_ACCEL OUTPUT_FORMAT
    
    # Process files with progress bar
    echo "Starting conversion with $JOBS parallel jobs..." >> "$LOG_FILE"
    
    # Create a temporary file to track progress
    PROGRESS_FILE=$(mktemp)
    echo "0" > "$PROGRESS_FILE"
    
    # Start parallel processing with progress tracking
    (
        parallel --eta --progress --joblog "$OUTPUT_DIR/parallel_job.log" -j "$JOBS" convert_aspect_ratio ::: "${VIDEO_FILES[@]}" 2>/dev/null
        
        # Write 100% when done to ensure progress bar completes
        echo "100"
        echo "# Conversion complete!"
    ) | zenity --progress \
        --title="Converting Videos" \
        --text="Starting aspect ratio conversion..." \
        --percentage=0 \
        --auto-close \
        --width=500 \
        --height=100
    
    # Calculate statistics
    total_success=$(grep -c "Converted:" "$LOG_FILE" || echo 0)
    total_errors=$((total_files - total_success))
    
    # Show completion message
    completion_message="Video conversion complete.\n\n"
    completion_message+="Total files: $total_files\n"
    completion_message+="Successfully converted: $total_success\n"
    completion_message+="Failed: $total_errors\n\n"
    completion_message+="See $LOG_FILE for details."
    
    show_dialog "Conversion Complete" "$completion_message"
    
    # Finalize log
    echo "-------------------------------------------" >> "$LOG_FILE"
    echo "Conversion complete at $(date)" >> "$LOG_FILE"
    echo "Total files: $total_files" >> "$LOG_FILE"
    echo "Successfully converted: $total_success" >> "$LOG_FILE"
    echo "Failed: $total_errors" >> "$LOG_FILE"
    
    # Clean up
    rm -f "$PROGRESS_FILE"
}

# Run main function
main
