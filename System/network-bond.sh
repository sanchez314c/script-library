#!/bin/bash
#
# network-bond.sh - Network Interface Bonding
# -------------------------------------------
# Author: sanchez314c@speedheathens.com
# Date: March 3, 2025
# Version: 1.1.0
#
# Description:
#     Creates and configures network interface bonding for improved
#     reliability and performance on macOS. Supports both Ethernet and Wi-Fi
#     interfaces with automatic failover.
#
# Features:
#     - Automatic interface detection
#     - Network bridge creation
#     - Intelligent route management
#     - Continuous status monitoring
#     - Robust error handling
#     - Graceful exit support
#     - Command-line options
#
# Requirements:
#     - macOS 10.14+
#     - Admin privileges
#     - Network tools (ifconfig, route, ping)
#
# Usage:
#     ./network-bond.sh [options]
#
# Options:
#     -h, --help             Show this help
#     -t, --test             Test mode (no changes)
#     -i, --interfaces INT   Specify interfaces (comma-separated)
#     -m, --monitor-only     Only monitor, don't create bridge
#     -d, --dns DNS          Specify DNS server for connectivity test
#

# Set up error handling and exit traps
set -e                # Exit on error
trap cleanup EXIT     # Call cleanup function on exit

# Configurable variables
TEST_SERVER="8.8.8.8"   # Default DNS for connectivity checking
MONITOR_INTERVAL=30     # Seconds between checks
LOG_FILE="/tmp/network-bond.log"
INTERFACES=()           # Will be populated by auto-detection or args

# ANSI color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${message}"
    echo "${message}" >> "${LOG_FILE}"
}

# Function to log error messages and exit
error() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Function to log success messages
success() {
    log "${GREEN}SUCCESS: $1${NC}"
}

# Function to log info messages
info() {
    log "${BLUE}INFO: $1${NC}"
}

# Function to log warning messages
warning() {
    log "${YELLOW}WARNING: $1${NC}"
}

# Cleanup function called on exit
cleanup() {
    info "Cleaning up and exiting..."
    # Kill any background processes we started
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" &>/dev/null || true
    fi
    info "Cleanup complete. Exiting"
}

# Show help information
show_help() {
    cat << EOF
Network Interface Bonding Script
-------------------------------
Usage: ${0} [OPTIONS]

OPTIONS:
  -h, --help              Show this help message
  -t, --test              Test mode - don't make changes
  -i, --interfaces IFACES Specify interfaces to bond (comma-separated)
  -m, --monitor-only      Only monitor existing interfaces
  -d, --dns SERVER        DNS server to use for connectivity tests
  -n, --no-monitor        Don't start monitoring (exit after setup)

Examples:
  ${0} --interfaces en0,en1
  ${0} --monitor-only --dns 1.1.1.1
EOF
}

# Parse command line options
parse_options() {
    MONITOR_ONLY=false
    TEST_MODE=false
    NO_MONITOR=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--test)
                TEST_MODE=true
                shift
                ;;
            -i|--interfaces)
                # Parse comma-separated list of interfaces
                IFS=',' read -ra INTERFACES <<< "$2"
                shift 2
                ;;
            -m|--monitor-only)
                MONITOR_ONLY=true
                shift
                ;;
            -d|--dns)
                TEST_SERVER="$2"
                shift 2
                ;;
            -n|--no-monitor)
                NO_MONITOR=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check if we have root privileges
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        if command -v sudo &>/dev/null; then
            warning "This script requires root privileges. Attempting to use sudo..."
            if ! sudo -n true 2>/dev/null; then
                error "Please run this script with sudo or as root."
                exit 1
            fi
        else
            error "This script requires root privileges. Please run as root."
            exit 1
        fi
    fi
}

# Function to identify network interfaces
identify_interfaces() {
    info "Identifying network interfaces..."
    
    # If interfaces were specified via arguments, use those
    if [[ ${#INTERFACES[@]} -gt 0 ]]; then
        info "Using specified interfaces: ${INTERFACES[*]}"
        return 0
    fi
    
    # Otherwise, try to auto-detect
    local ethernet_interfaces=()
    local wifi_interfaces=()
    
    # On macOS, usually en0 is Wi-Fi and en1 is Ethernet, but let's check
    if ifconfig en0 &>/dev/null; then
        if ifconfig en0 | grep -q "status: active"; then
            if ifconfig en0 | grep -q "802.11"; then
                wifi_interfaces+=("en0")
            else
                ethernet_interfaces+=("en0")
            fi
        fi
    fi
    
    if ifconfig en1 &>/dev/null; then
        if ifconfig en1 | grep -q "status: active"; then
            if ifconfig en1 | grep -q "802.11"; then
                wifi_interfaces+=("en1")
            else
                ethernet_interfaces+=("en1")
            fi
        fi
    fi
    
    # Add any additional interfaces up to en5
    for i in {2..5}; do
        if ifconfig en$i &>/dev/null; then
            if ifconfig en$i | grep -q "status: active"; then
                if ifconfig en$i | grep -q "802.11"; then
                    wifi_interfaces+=("en$i")
                else
                    ethernet_interfaces+=("en$i")
                fi
            fi
        fi
    done
    
    # Combine all active interfaces
    INTERFACES=( "${ethernet_interfaces[@]}" "${wifi_interfaces[@]}" )
    
    if [[ ${#INTERFACES[@]} -lt 2 ]]; then
        warning "Found fewer than 2 active network interfaces. Network bonding requires at least 2."
        info "Available interfaces: ${INTERFACES[*]}"
        if [[ "$TEST_MODE" = true ]]; then
            info "TEST MODE: Continuing anyway for testing purposes."
        elif [[ "$MONITOR_ONLY" = true ]]; then
            info "MONITOR ONLY: Will monitor available interfaces."
        else
            error "Cannot create network bond with fewer than 2 interfaces."
        fi
    else
        success "Found ${#INTERFACES[@]} active interfaces: ${INTERFACES[*]}"
    fi
}

# Function to create a network bridge
create_network_bridge() {
    if [[ "$MONITOR_ONLY" = true ]]; then
        info "Monitor only mode - skipping bridge creation."
        return 0
    fi
    
    info "Creating network bridge for interfaces: ${INTERFACES[*]}..."
    
    if [[ "$TEST_MODE" = true ]]; then
        info "TEST MODE: Would create bridge with: ${INTERFACES[*]}"
        return 0
    fi
    
    # Create bridge0 if it doesn't exist
    if ! ifconfig bridge0 &>/dev/null; then
        info "Creating bridge0 interface..."
        sudo ifconfig bridge0 create || error "Failed to create bridge0"
        success "Created bridge0 interface"
    else
        info "bridge0 interface already exists"
    fi
    
    # Add each interface to the bridge
    for interface in "${INTERFACES[@]}"; do
        info "Adding $interface to bridge0..."
        if ! sudo ifconfig bridge0 addm "$interface" 2>/dev/null; then
            warning "Failed to add $interface to bridge0. It may already be added or unavailable."
        else
            success "Added $interface to bridge0"
        fi
    done
    
    # Bring the bridge up
    info "Bringing bridge0 up..."
    if ! sudo ifconfig bridge0 up; then
        error "Failed to bring bridge0 up"
    fi
    success "Bridge0 is now active"
}

# Function to add routes for interfaces
add_routes() {
    if [[ "$MONITOR_ONLY" = true ]]; then
        info "Monitor only mode - skipping route management."
        return 0
    fi
    
    info "Setting up routes for interfaces: ${INTERFACES[*]}..."
    
    if [[ "$TEST_MODE" = true ]]; then
        info "TEST MODE: Would add routes for: ${INTERFACES[*]}"
        return 0
    fi
    
    # Check and add route for each interface
    for interface in "${INTERFACES[@]}"; do
        info "Checking route for $interface..."
        if ! route -n get default | grep -q "interface: $interface"; then
            info "Adding default route for $interface..."
            if ! sudo route -n add -net default -interface "$interface"; then
                warning "Failed to add route for $interface. It may already exist."
            else
                success "Added default route for $interface"
            fi
        else
            info "Route for $interface already exists"
        fi
    done
}

# Function to check if an interface has connectivity
check_connectivity() {
    local interface="$1"
    ping -c 1 -I "$interface" "$TEST_SERVER" &>/dev/null
    return $?
}

# Function to monitor and adjust routes
monitor_and_adjust_routes() {
    if [[ "$NO_MONITOR" = true ]]; then
        info "No-monitor mode enabled. Exiting after setup."
        return 0
    fi
    
    info "Starting network monitoring..."
    info "Checking connectivity every $MONITOR_INTERVAL seconds using $TEST_SERVER"
    info "Press Ctrl+C to stop monitoring"
    
    # Track monitor time for status reporting
    local start_time=$(date +%s)
    local monitor_count=0
    
    while true; do
        monitor_count=$((monitor_count + 1))
        local current_time=$(date +%s)
        local runtime=$((current_time - start_time))
        local hours=$((runtime / 3600))
        local minutes=$(( (runtime % 3600) / 60 ))
        local seconds=$((runtime % 60))
        
        info "Monitor check #$monitor_count (uptime: ${hours}h ${minutes}m ${seconds}s)"
        
        local working_interfaces=()
        
        # Check connectivity for each interface
        for interface in "${INTERFACES[@]}"; do
            info "Checking connectivity on $interface..."
            if check_connectivity "$interface"; then
                success "$interface has connectivity to $TEST_SERVER"
                working_interfaces+=("$interface")
                
                if [[ "$TEST_MODE" != true && "$MONITOR_ONLY" != true ]]; then
                    info "Setting $interface as default route"
                    sudo route change default -interface "$interface" 2>/dev/null || 
                    sudo route add default -interface "$interface" 2>/dev/null || 
                    warning "Failed to update route for $interface"
                fi
            else
                warning "$interface has no connectivity to $TEST_SERVER"
            fi
        done
        
        # Report status
        if [[ ${#working_interfaces[@]} -eq 0 ]]; then
            warning "No interfaces have connectivity!"
        else
            info "Working interfaces: ${working_interfaces[*]}"
        fi
        
        # Sleep before next check
        info "Next check in $MONITOR_INTERVAL seconds..."
        sleep "$MONITOR_INTERVAL"
    done
}

# Main function to run all steps
main() {
    # Create log file
    touch "$LOG_FILE" || warning "Could not create log file at $LOG_FILE"
    
    info "Network bonding script started"
    info "macOS $(sw_vers -productVersion)"
    
    # Parse command line options
    parse_options "$@"
    
    # Check privileges
    check_privileges
    
    # Identify interfaces
    identify_interfaces
    
    # Create bridge
    create_network_bridge
    
    # Add routes
    add_routes
    
    # Monitor routes
    monitor_and_adjust_routes
    
    success "Network bonding completed"
}

# Run the main function with all arguments
main "$@"
