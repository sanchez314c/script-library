#!/bin/bash

# AI Suite Installer - Text-Based Master Control
# Version: 2.0.0
# Last Updated: 2025-01-06
# Security: Enhanced error handling, flexible path detection, improved validation

set -euo pipefail  # Strict error handling

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging setup
readonly LOG_FILE="/tmp/ai_installer_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error_exit() {
    echo -e "${RED}❌ ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit "${2:-1}"
}

warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root (should not be)
    if [[ $EUID -eq 0 ]]; then
        error_exit "Do not run this script as root. It will use sudo when needed."
    fi
    
    # Check sudo access
    if ! sudo -n true 2>/dev/null; then
        log "Testing sudo access..."
        sudo -v || error_exit "Sudo access required but not available"
    fi
    
    # Install dialog if needed
    if ! command -v dialog &>/dev/null; then
        log "Installing dialog for menu interface..."
        sudo apt update && sudo apt install -y dialog || error_exit "Failed to install dialog"
    fi
    
    success "Prerequisites check completed"
}

# Determine script directory with fallback options
determine_script_dir() {
    local possible_dirs=(
        "/media/$USER/USB"
        "/media/heathen-admin/USB"
        "$(pwd)"
        "$(dirname "$0")"
        "/opt/ai-scripts"
        "$HOME/ai-scripts"
    )
    
    for dir in "${possible_dirs[@]}"; do
        if [[ -d "$dir" ]] && [[ -r "$dir" ]]; then
            log "Found script directory: $dir"
            echo "$dir"
            return 0
        fi
    done
    
    error_exit "Could not find script directory. Tried: ${possible_dirs[*]}"
}

SCRIPT_DIR=$(determine_script_dir)
cd "$SCRIPT_DIR" || error_exit "Failed to change to script directory: $SCRIPT_DIR"
log "Working in directory: $SCRIPT_DIR"

# List of scripts (in dependency order)
SCRIPTS=(
    "nvidia-470-driver-install.sh" "NVIDIA 470 Driver"
    "cuda-installer.sh" "CUDA 11.4 for K80s"
    "ubuntu-essentials-install.sh" "Ubuntu Base Setup"
    "network-system-tools.sh" "Security & Monitoring"
    "rocm-installer.sh" "ROCm 6.3.3 for RX580"
    "ai-ml-docker-frameworks.sh" "Conda & Docker Envs"
    "ollama-rx580-rocm-compiler.sh" "Ollama ROCm (RX580)"
    "ollama-k80-cuda-compiler.sh" "Ollama CUDA (K80s)"
    "exo-distributed-inferance-install.sh" "EXO GPU Cluster"
    "open-webui-baremetal-installer.sh" "Open WebUI"
    "comfyui-flux1-sd-installer.sh" "ComfyUI (SD & FLUX.1)"
    "whisper-tts-stt.sh" "Whisper TTS/STT"
    "ollama-rx580-rocm-uninstaller.sh" "Ollama ROCm Uninstaller"
    "ollama-k80-uninstaller.sh" "Ollama CUDA Uninstaller"
    "application-installer.sh" "Application Installer (TBD)"
)

# Auto-attend flag
AUTO_FLAG="--yes"

# Build dialog menu options
MENU_OPTIONS=()
for ((i=0; i<${#SCRIPTS[@]}; i+=2)); do
    MENU_OPTIONS+=("$((i/2 + 1))" "${SCRIPTS[$i+1]}")
done
MENU_OPTIONS+=("all" "Install All (Auto-Attend)")

# Temporary file for dialog output
TEMP_FILE=$(mktemp)

# Display menu
dialog --clear --title "Jason's AI Suite Installer" \
    --menu "Select a script to run or install all (Ctrl+C to exit):" 20 60 12 \
    "${MENU_OPTIONS[@]}" 2> "$TEMP_FILE"

# Get user choice
CHOICE=$(cat "$TEMP_FILE")
rm -f "$TEMP_FILE"

# Validate script before execution
validate_script() {
    local script="$1"
    local script_path="$SCRIPT_DIR/$script"
    
    # Check if script exists
    if [[ ! -f "$script_path" ]]; then
        warning "Script not found: $script"
        return 1
    fi
    
    # Check if script is readable
    if [[ ! -r "$script_path" ]]; then
        warning "Script not readable: $script"
        return 1
    fi
    
    # Check if it's a shell script
    if ! head -1 "$script_path" | grep -q '^#!.*sh'; then
        warning "Not a shell script: $script"
        return 1
    fi
    
    # Basic security check - no suspicious patterns
    if grep -q -E "(rm -rf /|wget.*\||curl.*\||eval|exec)" "$script_path"; then
        warning "Script contains potentially dangerous patterns: $script"
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    return 0
}

# Function to run a script
run_script() {
    local script="$1"
    local script_path="$SCRIPT_DIR/$script"
    
    log "Preparing to run: $script"
    
    # Skip TBD script if not implemented
    if [[ "$script" = "application-installer.sh" ]] && [[ ! -s "$script_path" ]]; then
        warning "$script not implemented yet—skipping."
        return 0
    fi
    
    # Validate script before running
    if ! validate_script "$script"; then
        error_exit "Script validation failed for: $script"
    fi
    
    # Make executable
    chmod +x "$script_path" || error_exit "Failed to make script executable: $script"
    
    log "Executing: $script"
    # Use timeout to prevent hanging scripts
    if timeout 3600 sudo "$script_path" "$AUTO_FLAG"; then
        success "$script completed successfully"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            error_exit "$script timed out after 1 hour"
        else
            error_exit "$script failed with exit code: $exit_code. Check $LOG_FILE for details."
        fi
    fi
}

# Main execution with enhanced error handling
main() {
    log "Starting AI Suite Installer"
    log "Log file: $LOG_FILE"
    
    # Initialize
    check_prerequisites
    
    # Process user choice
    if [[ "$CHOICE" = "all" ]]; then
        log "Installing all scripts in dependency order..."
        local failed_scripts=()
        local successful_scripts=()
        
        for ((i=0; i<${#SCRIPTS[@]}; i+=2)); do
            local script="${SCRIPTS[$i]}"
            
            # Skip uninstallers in "all" mode
            if [[ "$script" =~ "uninstaller" ]]; then
                log "Skipping uninstaller: $script"
                continue
            fi
            
            if run_script "$script"; then
                successful_scripts+=("$script")
            else
                failed_scripts+=("$script")
                warning "Failed to install: $script"
                
                # Ask if user wants to continue
                read -p "Continue with remaining installations? (y/N): " -r
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    break
                fi
            fi
        done
        
        # Summary
        echo -e "\n${BLUE}📊 Installation Summary:${NC}"
        echo "Successful: ${#successful_scripts[@]}"
        echo "Failed: ${#failed_scripts[@]}"
        
        if [[ ${#successful_scripts[@]} -gt 0 ]]; then
            echo -e "\n${GREEN}✅ Successfully installed:${NC}"
            printf ' - %s\n' "${successful_scripts[@]}"
        fi
        
        if [[ ${#failed_scripts[@]} -gt 0 ]]; then
            echo -e "\n${RED}❌ Failed to install:${NC}"
            printf ' - %s\n' "${failed_scripts[@]}"
            echo -e "\nCheck log file for details: $LOG_FILE"
        fi
        
        echo -e "\n${YELLOW}📋 Next Steps:${NC}"
        echo "- Review log file: $LOG_FILE"
        echo "- Uninstallers available for cleanup if needed"
        echo "- Reboot recommended after major installations"
        
    else
        # Single script installation
        local index=$(( (CHOICE - 1) * 2 ))
        if [[ $index -ge 0 ]] && [[ $index -lt ${#SCRIPTS[@]} ]]; then
            local script="${SCRIPTS[$index]}"
            local description="${SCRIPTS[$((index + 1))]}"
            
            log "Single script installation: $script"
            
            if run_script "$script"; then
                success "$description installation completed successfully!"
                echo -e "\n${BLUE}💡 You can run this installer again for more options${NC}"
            else
                error_exit "$description installation failed. Check $LOG_FILE for details."
            fi
        else
            error_exit "Invalid choice: $CHOICE"
        fi
    fi
    
    success "AI Suite Installer completed. Log: $LOG_FILE"
}

# Trap to ensure cleanup
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo -e "\n${RED}Script exited with error code: $exit_code${NC}"
        echo "Check log file for details: $LOG_FILE"
    fi
    # Remove temp files
    [[ -f "$TEMP_FILE" ]] && rm -f "$TEMP_FILE"
}
trap cleanup EXIT

# Run main function
main
