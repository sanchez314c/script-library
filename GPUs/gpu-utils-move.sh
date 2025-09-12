#!/bin/bash
#
# GPU Utilities - File Movement Script
# ----------------------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Handles movement and organization of GPU-related test files
#     and results. Maintains proper file structure and permissions.
#     Supports both single file and directory operations.
#
# Features:
#     - File organization by type
#     - Permission handling
#     - Timestamp preservation
#     - Error handling
#     - Log generation
#     - Backup creation
#
# Usage:
#     bash gpu-utils-move.sh [source] [destination] [--backup] [--verbose]

# Set default values
VERBOSE=false
CREATE_BACKUP=false
LOG_FILE="gpu-move.log"

# Display script usage
usage() {
    echo "Usage: $0 [source] [destination] [--backup] [--verbose]"
    echo
    echo "Options:"
    echo "  --backup    Create a backup of files before moving"
    echo "  --verbose   Display detailed information during execution"
    echo
    echo "Examples:"
    echo "  $0 ./results ./archive"
    echo "  $0 ./test.py ./scripts/ --backup"
    echo "  $0 ./benchmark_results/ ./results/ --verbose"
    exit 1
}

# Log messages
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" >> "$LOG_FILE"
    
    if $VERBOSE; then
        echo "$message"
    fi
}

# Create backup of a file or directory
create_backup() {
    local source="$1"
    local backup_dir="./backups/$(date +"%Y%m%d_%H%M%S")"
    
    mkdir -p "$backup_dir"
    
    if [ -f "$source" ]; then
        # Backup file
        cp -p "$source" "$backup_dir/"
        log "Created backup of file: $source in $backup_dir"
    elif [ -d "$source" ]; then
        # Backup directory
        cp -rp "$source" "$backup_dir/"
        log "Created backup of directory: $source in $backup_dir"
    else
        log "Error: Cannot create backup - source does not exist: $source"
        return 1
    fi
    
    return 0
}

# Move a file to destination
move_file() {
    local source="$1"
    local destination="$2"
    
    # Check if source exists
    if [ ! -f "$source" ]; then
        log "Error: Source file does not exist: $source"
        return 1
    fi
    
    # Check if destination is a directory
    if [ -d "$destination" ]; then
        # Move to directory
        mv -f "$source" "$destination/"
        result=$?
    else
        # Move with specific name
        mv -f "$source" "$destination"
        result=$?
    fi
    
    if [ $result -eq 0 ]; then
        log "Successfully moved: $source to $destination"
    else
        log "Error: Failed to move: $source to $destination"
    fi
    
    return $result
}

# Move a directory to destination
move_directory() {
    local source="$1"
    local destination="$2"
    
    # Check if source exists
    if [ ! -d "$source" ]; then
        log "Error: Source directory does not exist: $source"
        return 1
    fi
    
    # Check if destination is a directory
    if [ ! -d "$destination" ]; then
        mkdir -p "$destination"
        log "Created destination directory: $destination"
    fi
    
    # Move all contents from source to destination
    for item in "$source"/*; do
        if [ -f "$item" ]; then
            # Move file
            mv -f "$item" "$destination/"
            if [ $? -eq 0 ]; then
                log "Moved file: $item to $destination/"
            else
                log "Error: Failed to move: $item to $destination/"
            fi
        elif [ -d "$item" ]; then
            # Create subdirectory in destination
            local dirname=$(basename "$item")
            mkdir -p "$destination/$dirname"
            
            # Move subdirectory contents
            for subitem in "$item"/*; do
                if [ -e "$subitem" ]; then
                    mv -f "$subitem" "$destination/$dirname/"
                    if [ $? -eq 0 ]; then
                        log "Moved item: $subitem to $destination/$dirname/"
                    else
                        log "Error: Failed to move: $subitem to $destination/$dirname/"
                    fi
                fi
            done
            
            # Remove empty source subdirectory
            rmdir "$item" 2>/dev/null
        fi
    done
    
    # Remove empty source directory
    rmdir "$source" 2>/dev/null
    
    log "Completed moving directory: $source to $destination"
    return 0
}

# Parse command line arguments
if [ $# -lt 2 ]; then
    usage
fi

SOURCE="$1"
DESTINATION="$2"
shift 2

# Process optional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --backup)
            CREATE_BACKUP=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
    shift
done

# Initialize log file
echo "GPU File Movement Log - $(date)" > "$LOG_FILE"
log "Starting file movement operation"
log "Source: $SOURCE"
log "Destination: $DESTINATION"

# Create backup if requested
if $CREATE_BACKUP; then
    log "Creating backup before moving files"
    create_backup "$SOURCE"
    if [ $? -ne 0 ]; then
        log "Warning: Backup creation failed"
    fi
fi

# Perform move operation
if [ -f "$SOURCE" ]; then
    # Moving a single file
    log "Moving single file"
    move_file "$SOURCE" "$DESTINATION"
    result=$?
elif [ -d "$SOURCE" ]; then
    # Moving a directory
    log "Moving directory contents"
    move_directory "$SOURCE" "$DESTINATION"
    result=$?
else
    log "Error: Source does not exist: $SOURCE"
    result=1
fi

# Set proper permissions on destination
if [ -d "$DESTINATION" ]; then
    chmod -R u+rw "$DESTINATION"
    log "Updated permissions on destination"
fi

# Finalize
if [ $result -eq 0 ]; then
    log "File movement completed successfully"
    if $VERBOSE; then
        echo "Operation completed successfully. See $LOG_FILE for details."
    fi
    exit 0
else
    log "File movement completed with errors"
    echo "Operation completed with errors. See $LOG_FILE for details."
    exit 1
fi
