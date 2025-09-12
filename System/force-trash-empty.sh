#!/bin/bash
####################################################################################
#                                                                                  #
#    ██████╗ ███████╗████████╗   ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗   #
#   ██╔════╝ ██╔════╝╚══██╔══╝   ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝   #
#   ██║  ███╗█████╗     ██║      ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝    #
#   ██║   ██║██╔══╝     ██║      ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝     #
#   ╚██████╔╝███████╗   ██║      ███████║╚███╔███╔╝██║██╗        ██║      ██║      #
#    ╚═════╝ ╚══════╝   ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝      #
#                                                                                  #
####################################################################################
#
# Script Name: force-trash-empty.sh                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible force trash emptying utility that safely empties
#     system and external volume Trash folders, bypassing standard OS restrictions
#     with comprehensive error handling and safety checks.
#
# Features:
#     - Universal macOS compatibility with user confirmation
#     - System Trash clearing (~/.Trash)
#     - External volume Trash support (/Volumes/*/.Trashes)
#     - Safety confirmations and error handling
#     - Comprehensive logging and status reporting
#     - Root-level access with sudo prompting
#
# Requirements:
#     - bash 4.0+ (standard on macOS)
#     - sudo access (prompted when needed)
#     - rm command (standard Unix tool)
#
# Usage:
#     ./force-trash-empty.sh
#
####################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Display header
echo -e "${BLUE}"
echo "####################################################################################"
echo "#                         FORCE TRASH EMPTY UTILITY                             #"
echo "#                    Universal macOS Trash Cleanup Tool                         #"
echo "####################################################################################"
echo -e "${NC}"

# Safety warning
warning "⚠️  CAUTION: This will PERMANENTLY delete all files in Trash folders!"
warning "⚠️  Recovery is NOT possible after this operation!"
echo ""

# Confirm user wants to proceed
read -p "Are you sure you want to empty ALL trash folders? (type 'YES' to confirm): " confirm
if [[ "$confirm" != "YES" ]]; then
    log "Operation cancelled by user."
    exit 0
fi

# Function to get trash size
get_trash_size() {
    local path="$1"
    if [[ -d "$path" ]]; then
        du -sh "$path" 2>/dev/null | cut -f1 || echo "Unknown"
    else
        echo "0B"
    fi
}

# Function to safely empty trash
empty_trash() {
    local trash_path="$1"
    local description="$2"
    
    if [[ ! -d "$trash_path" ]]; then
        log "$description: Not found, skipping..."
        return 0
    fi
    
    local size=$(get_trash_size "$trash_path")
    log "$description: Found ($size)"
    
    # Count items
    local item_count=$(find "$trash_path" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
    
    if [[ "$item_count" -eq 0 ]]; then
        log "$description: Already empty"
        return 0
    fi
    
    log "$description: Emptying $item_count items..."
    
    # Use sudo if needed and remove contents
    if [[ -w "$trash_path" ]]; then
        rm -rf "$trash_path"/* "$trash_path"/.[^.]* "$trash_path"/..?* 2>/dev/null || true
    else
        sudo rm -rf "$trash_path"/* "$trash_path"/.[^.]* "$trash_path"/..?* 2>/dev/null || true
    fi
    
    # Verify emptying
    local remaining=$(find "$trash_path" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$remaining" -eq 0 ]]; then
        success "$description: Successfully emptied"
    else
        error "$description: Failed to empty completely ($remaining items remaining)"
    fi
}

log "Starting trash emptying process..."

# Empty user's main trash
empty_trash "$HOME/.Trash" "User Trash (~/.Trash)"

# Empty external volume trash folders
log "Scanning for external volume trash folders..."
external_count=0

for volume in /Volumes/*/; do
    if [[ -d "$volume" && "$volume" != "/Volumes//" ]]; then
        volume_name=$(basename "$volume")
        trashes_path="${volume}.Trashes"
        
        if [[ -d "$trashes_path" ]]; then
            empty_trash "$trashes_path" "External Volume Trash ($volume_name/.Trashes)"
            ((external_count++))
        fi
        
        # Also check for user-specific trash in external volumes
        user_trash="${volume}.Trash"
        if [[ -d "$user_trash" ]]; then
            empty_trash "$user_trash" "External Volume User Trash ($volume_name/.Trash)"
            ((external_count++))
        fi
    fi
done

if [[ "$external_count" -eq 0 ]]; then
    log "No external volume trash folders found"
fi

# Final cleanup - empty macOS secure empty trash if available
if command -v osascript >/dev/null 2>&1; then
    log "Performing secure empty via macOS system call..."
    osascript -e 'tell application "Finder" to empty trash' 2>/dev/null || true
fi

# Final status report
echo ""
success "🎉 Trash emptying process completed!"
log "All accessible trash folders have been processed."
warning "⚠️  Remember: Deleted files cannot be recovered!"

# Optional: Show disk space freed (approximate)
if command -v df >/dev/null 2>&1; then
    echo ""
    log "Current disk usage:"
    df -h / | tail -1 | awk '{print "Available space: " $4 " (" $5 " used)"}'
fi

exit 0