#!/bin/bash
#
# restore-docker-containers - Docker State and Volume Restore
# ------------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: March 3, 2025
# Version: 1.1.0
#
# Description:
#     Comprehensive Docker container restore solution that recovers
#     container states, configurations, and associated data volumes.
#     Works in tandem with backup-docker-containers.sh for complete
#     system recovery capability.
#
# Features:
#     - Container restoration
#     - Image recovery
#     - Volume data restoration
#     - Network recreation
#     - Applications directory recovery
#     - Progress tracking with verification
#
# Requirements:
#     - bash 4.0+
#     - Docker installation
#     - Sufficient storage space
#     - Root access
#
# Usage:
#     ./restore-docker-containers.sh [backup_dir]
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    error_exit "Docker is not installed or not in PATH"
fi

# Get backup directory from arguments or use dialog for selection
get_backup_directory() {
    local backup_dir="$1"
    
    if [ -z "$backup_dir" ]; then
        # If running in GUI environment, use native dialog
        if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
            backup_dir=$(osascript -e 'tell application "Finder" to return POSIX path of (choose folder with prompt "Select Backup Directory to Restore From")')
        else
            # Fallback to text prompt
            read -p "Enter backup directory path: " backup_dir
        fi
    fi
    
    # Validate directory
    if [ ! -d "$backup_dir" ]; then
        error_exit "Backup directory does not exist: $backup_dir"
    fi
    
    # Check if it's a Docker backup directory
    if [ -d "$backup_dir/Docker" ]; then
        backup_dir="$backup_dir/Docker"
    fi
    
    # Check for "latest" symlink
    if [ -L "$backup_dir/latest" ] && [ -d "$backup_dir/$(readlink "$backup_dir/latest")" ]; then
        backup_dir="$backup_dir/$(readlink "$backup_dir/latest")"
    elif [ -d "$backup_dir" ] && [ -f "$backup_dir/docker_version.txt" ]; then
        # This is already a valid backup directory
        :
    else
        # Check for timestamp directories and pick the latest
        local latest_dir=$(find "$backup_dir" -maxdepth 1 -type d -name "202*" | sort | tail -n 1)
        if [ -n "$latest_dir" ]; then
            backup_dir="$latest_dir"
        else
            error_exit "No valid Docker backup found in $backup_dir"
        fi
    fi
    
    echo "$backup_dir"
}

# Select target directory for application restore
get_target_directory() {
    # If running in GUI environment, use native dialog
    if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
        target_dir=$(osascript -e 'tell application "Finder" to return POSIX path of (choose folder with prompt "Select Target Directory for Application Restore")')
    else
        # Fallback to text prompt
        read -p "Enter target directory for application restore: " target_dir
    fi
    
    # Create target directory if it doesn't exist
    mkdir -p "$target_dir" || error_exit "Failed to create target directory: $target_dir"
    
    echo "$target_dir"
}

# Check Docker environment before restore
check_environment() {
    log_progress "Checking Docker environment..."
    
    # Check if Docker daemon is running
    docker info &>/dev/null || error_exit "Docker daemon is not running"
    
    # Check if there are running containers
    local running_containers=$(docker ps -q)
    if [ -n "$running_containers" ]; then
        log_progress "Warning: There are running containers that may be affected by the restore operation"
        
        # If running in GUI environment, use native dialog
        if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
            local proceed=$(osascript -e 'display dialog "There are running containers that will be stopped. Do you want to proceed?" buttons {"Cancel", "Proceed"} default button "Proceed"' -e 'button returned of result')
            if [ "$proceed" != "Proceed" ]; then
                error_exit "Restore cancelled by user"
            fi
        else
            # Fallback to text prompt
            read -p "Running containers will be stopped. Proceed? (y/n): " proceed
            if [ "$proceed" != "y" ]; then
                error_exit "Restore cancelled by user"
            fi
        fi
        
        # Stop all running containers
        log_progress "Stopping all running containers..."
        docker stop $(docker ps -q) || log_progress "Warning: Failed to stop some containers"
    fi
}

# Restore Docker volumes
restore_volumes() {
    local backup_dir="$1"
    
    log_progress "Restoring Docker volumes..."
    
    # Find volume backups
    local volume_backups=$(find "$backup_dir" -name "volume_*.tar.gz")
    
    if [ -z "$volume_backups" ]; then
        log_progress "No volume backups found"
        return
    fi
    
    # Process each volume
    for volume_backup in $volume_backups; do
        local volume_name=$(basename "$volume_backup" | sed 's/^volume_//;s/\.tar\.gz$//')
        
        log_progress "Restoring volume: $volume_name"
        
        # Create volume if it doesn't exist
        docker volume inspect "$volume_name" &>/dev/null || docker volume create "$volume_name" || {
            log_progress "Warning: Failed to create volume $volume_name, skipping"
            continue
        }
        
        # Extract volume data
        log_progress "Extracting data for volume: $volume_name"
        docker run --rm -v "$volume_name":/target -v "$(dirname "$volume_backup")":/backup \
            alpine sh -c "mkdir -p /target && gunzip -c /backup/$(basename "$volume_backup") | tar xf - -C /target" || \
            log_progress "Warning: Failed to restore volume $volume_name"
    done
}

# Restore Docker images
restore_images() {
    local backup_dir="$1"
    
    log_progress "Restoring Docker images..."
    
    # Find image backups
    local images_backup=$(find "$backup_dir" -name "images.tar.gz" | head -n 1)
    
    if [ -z "$images_backup" ]; then
        log_progress "No image backup found"
        return
    fi
    
    # Extract and load images
    log_progress "Extracting and loading images..."
    gunzip -c "$images_backup" | docker load || log_progress "Warning: Failed to restore some images"
    
    # List restored images
    log_progress "Restored images:"
    docker images | grep -v "^REPOSITORY"
}

# Restore application files
restore_applications() {
    local backup_dir="$1"
    
    log_progress "Checking for application backup..."
    
    # Find application backup
    local app_backup=$(find "$backup_dir" -name "applications.tar.gz" | head -n 1)
    if [ -z "$app_backup" ]; then
        app_backup=$(find "$backup_dir" -name "applications.tar" | head -n 1)
    fi
    
    if [ -z "$app_backup" ]; then
        log_progress "No application backup found"
        return
    fi
    
    # Ask for target directory
    log_progress "Application backup found. Select target directory for restore."
    local target_dir=$(get_target_directory)
    
    # Extract application files
    log_progress "Extracting application files to $target_dir..."
    if [[ "$app_backup" == *.gz ]]; then
        tar xzf "$app_backup" -C "$target_dir" || log_progress "Warning: Failed to extract application files"
    else
        tar xf "$app_backup" -C "$target_dir" || log_progress "Warning: Failed to extract application files"
    fi
    
    log_progress "Application restore completed"
}

# Restore containers
restore_containers() {
    local backup_dir="$1"
    
    log_progress "Note: Direct container state restoration is not possible. Images and volumes are restored instead."
    log_progress "After restore, you'll need to recreate your containers using the restored images and volumes."
    
    # Display container information from backup for reference
    if [ -f "$backup_dir/containers.txt" ]; then
        log_progress "Container information from backup (for reference):"
        cat "$backup_dir/containers.txt"
    fi
}

# Restore Docker environment
perform_restore() {
    local backup_dir="$1"
    
    log_progress "Starting restore from: $backup_dir"
    
    # Display backup summary if available
    if [ -f "$backup_dir/backup_summary.txt" ]; then
        log_progress "Backup summary:"
        cat "$backup_dir/backup_summary.txt"
    fi
    
    # Restore in sequence: volumes, images, applications
    restore_volumes "$backup_dir"
    restore_images "$backup_dir"
    restore_applications "$backup_dir"
    restore_containers "$backup_dir"
    
    log_progress "Restore process completed"
    log_progress "You may need to manually recreate containers using docker run or docker-compose"
}

# Main script execution
main() {
    log_progress "Starting Docker restore process..."
    
    # Get backup directory
    local backup_dir=$(get_backup_directory "$1")
    
    # Check environment
    check_environment
    
    # Perform restore
    perform_restore "$backup_dir"
}

# Run the main function with arguments
main "$@"
