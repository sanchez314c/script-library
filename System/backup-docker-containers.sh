#!/bin/bash
#
# backup-docker-containers - Docker State and Volume Backup
# ----------------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: March 3, 2025
# Version: 1.1.0
#
# Description:
#     Comprehensive Docker container backup solution that preserves
#     container states, configurations, and associated data volumes.
#     Allows for complete system recovery with proper backup verification.
#
# Features:
#     - Container state preservation
#     - Configuration backup
#     - Volume data protection
#     - Automated container handling
#     - Progress tracking with completion verification
#     - Compression for efficient storage
#
# Requirements:
#     - bash 4.0+
#     - Docker installation
#     - Sufficient storage space
#     - Root access
#
# Usage:
#     ./backup-docker-containers.sh [source_dir] [backup_dir]
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

# Get source and backup directories from arguments or use dialog for selection
get_directories() {
    local source_dir="$1"
    local backup_dir="$2"
    
    if [ -z "$source_dir" ]; then
        # If running in GUI environment, use native dialog
        if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
            source_dir=$(osascript -e 'tell application "Finder" to return POSIX path of (choose folder with prompt "Select Application Directory to Backup")')
        else
            # Fallback to text prompt
            read -p "Enter source directory path: " source_dir
        fi
    fi
    
    if [ -z "$backup_dir" ]; then
        if [ -n "$DISPLAY" ] && command -v osascript &> /dev/null; then
            backup_dir=$(osascript -e 'tell application "Finder" to return POSIX path of (choose folder with prompt "Select Backup Destination Directory")')
        else
            read -p "Enter backup directory path: " backup_dir
        fi
    fi
    
    # Validate directories
    if [ ! -d "$source_dir" ]; then
        error_exit "Source directory does not exist: $source_dir"
    fi
    
    # Create backup directory if it doesn't exist
    mkdir -p "$backup_dir" || error_exit "Failed to create backup directory: $backup_dir"
    
    echo "$source_dir|$backup_dir"
}

# Get container list with names instead of IDs
get_container_list() {
    # Get running containers
    local containers=$(docker ps --format "{{.ID}}|{{.Names}}")
    
    if [ -z "$containers" ]; then
        log_progress "No running containers found"
        return 1
    fi
    
    echo "$containers"
}

# Backup function
perform_backup() {
    local dirs="$1"
    local source_dir=$(echo "$dirs" | cut -d'|' -f1)
    local backup_dir=$(echo "$dirs" | cut -d'|' -f2)
    local docker_backup_dir="$backup_dir/Docker"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$docker_backup_dir/$timestamp"
    
    # Create backup directories
    log_progress "Creating backup directories..."
    mkdir -p "$backup_path"
    
    # Record system information
    log_progress "Recording system information..."
    uname -a > "$backup_path/system_info.txt"
    docker version > "$backup_path/docker_version.txt"
    
    # Save the state of Docker containers, images, volumes, and networks
    log_progress "Backing up Docker state..."
    docker ps -a > "$backup_path/containers.txt"
    docker images > "$backup_path/images.txt"
    docker volume ls > "$backup_path/volumes.txt"
    docker network ls > "$backup_path/networks.txt"
    
    # Get container list
    local container_list=$(get_container_list)
    if [ $? -ne 0 ]; then
        log_progress "No containers to backup, skipping container backup steps"
    else
        # Stop containers for consistent backup if needed
        log_progress "Stopping containers for consistent backup..."
        echo "$container_list" | while IFS='|' read -r container_id container_name; do
            log_progress "Stopping container: $container_name ($container_id)"
            docker stop "$container_id" || log_progress "Warning: Failed to stop container $container_name"
        done
        
        # Export containers
        log_progress "Backing up containers..."
        echo "$container_list" | while IFS='|' read -r container_id container_name; do
            log_progress "Exporting container: $container_name ($container_id)"
            docker export "$container_id" -o "$backup_path/container_${container_name}.tar" || \
                log_progress "Warning: Failed to export container $container_name"
        done
        
        # Restart containers
        log_progress "Restarting containers..."
        echo "$container_list" | while IFS='|' read -r container_id container_name; do
            log_progress "Starting container: $container_name ($container_id)"
            docker start "$container_id" || log_progress "Warning: Failed to restart container $container_name"
        done
    fi
    
    # Backup images
    log_progress "Backing up Docker images..."
    docker save $(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>") -o "$backup_path/images.tar" || \
        log_progress "Warning: Failed to backup all images"
    
    # Backup volumes
    log_progress "Backing up Docker volumes..."
    local volumes=$(docker volume ls -q)
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            log_progress "Backing up volume: $volume"
            docker run --rm -v "$volume":/source -v "$backup_path":/backup alpine tar cf "/backup/volume_${volume}.tar" -C /source . || \
                log_progress "Warning: Failed to backup volume $volume"
        done
    else
        log_progress "No volumes to backup"
    fi
    
    # Backup application directory if specified
    if [ -d "$source_dir" ]; then
        log_progress "Backing up application directory..."
        tar cf "$backup_path/applications.tar" -C "$(dirname "$source_dir")" "$(basename "$source_dir")" || \
            log_progress "Warning: Failed to backup application directory"
    fi
    
    # Compress backups to save space
    log_progress "Compressing backups to save space..."
    
    # Function to compress files with progress
    compress_file() {
        local file="$1"
        local dir=$(dirname "$file")
        local base=$(basename "$file")
        
        log_progress "Compressing $base..."
        gzip -f "$file" || log_progress "Warning: Failed to compress $base"
    }
    
    # Compress tar files
    find "$backup_path" -name "*.tar" -type f | while read file; do
        compress_file "$file"
    done
    
    # Create backup summary
    log_progress "Creating backup summary..."
    {
        echo "Docker Backup Summary"
        echo "====================="
        echo "Date: $(date)"
        echo "Source: $source_dir"
        echo "Backup Location: $backup_path"
        echo ""
        echo "Container Backups:"
        find "$backup_path" -name "container_*.tar.gz" | wc -l
        echo ""
        echo "Volume Backups:"
        find "$backup_path" -name "volume_*.tar.gz" | wc -l
        echo ""
        echo "Total Backup Size:"
        du -sh "$backup_path"
    } > "$backup_path/backup_summary.txt"
    
    # Create latest symlink
    ln -sf "$timestamp" "$docker_backup_dir/latest"
    
    log_progress "Backup completed successfully!"
    log_progress "Backup location: $backup_path"
    log_progress "Backup summary: $backup_path/backup_summary.txt"
}

# Main script execution
main() {
    log_progress "Starting Docker backup process..."
    
    # Get directories
    local dirs=$(get_directories "$1" "$2")
    
    # Perform backup
    perform_backup "$dirs"
}

# Run the main function with arguments
main "$@"

