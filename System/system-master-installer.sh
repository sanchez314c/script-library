#!/bin/bash
# Master Interactive Installer for AI/ML Development Environment
# Created by Claude for Jason
# Date: March 2, 2025
#
# This script provides an interactive way to install all components
# of the AI/ML environment in the correct order with dependency checks.

# ANSI color codes for a retro terminal feel
BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
BOLD='\033[1m'
BLINK='\033[5m'
NC='\033[0m' # No Color

# Ensure script is run as root
if [ "$(id -u)" != "0" ]; then
    echo -e "${RED}${BOLD}This script must be run as root${NC}"
    exit 1
fi

# Record current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print ASCII art header
print_header() {
    clear
    echo -e "${CYAN}${BOLD}"
    cat << "EOF"
██████╗  █████╗ ██████╗ ██╗  ██╗██████╗  ██████╗  ██████╗ ██╗     
██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝██╔══██╗██╔═══██╗██╔═══██╗██║     
██║  ██║███████║██████╔╝█████╔╝ ██████╔╝██║   ██║██║   ██║██║     
██║  ██║██╔══██║██╔══██╗██╔═██╗ ██╔═══╝ ██║   ██║██║   ██║██║     
██████╔╝██║  ██║██║  ██║██║  ██╗██║     ╚██████╔╝╚██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝  ╚═════╝ ╚══════╝
                                                                   
 █████╗ ██╗██╗███╗   ███╗██╗         ██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     ███████╗██████╗ 
██╔══██╗██║██║████╗ ████║██║         ██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     ██╔════╝██╔══██╗
███████║██║██║██╔████╔██║██║         ██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     █████╗  ██████╔╝
██╔══██║██║██║██║╚██╔╝██║██║         ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     ██╔══╝  ██╔══██╗
██║  ██║██║██║██║ ╚═╝ ██║███████╗    ██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝╚═╝╚═╝     ╚═╝╚══════╝    ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝
EOF
    echo -e "${NC}"
    echo -e "${YELLOW}${BOLD}AI/ML Environment Installation System${NC}"
    echo -e "${YELLOW}Created by Claude for Jason - March 2, 2025${NC}"
    echo ""
}

# Function to print retro-style section header
print_section() {
    local title="$1"
    local length=${#title}
    local padding=$(( (60 - length) / 2 ))
    echo ""
    echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    printf "${BLUE}${BOLD}║%*s%s%*s║${NC}\n" $padding "" "$title" $((padding + (length % 2)))  ""
    echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Function to show spinner during installation
show_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    
    echo -n "  "
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "${MAGENTA}${BOLD}[%c]${NC}" "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b"
    done
    printf "   \b\b\b"
}

# Function to print status messages
print_status() {
    local status=$1
    local message=$2
    
    if [ $status -eq 0 ]; then
        echo -e "  ${GREEN}${BOLD}[SUCCESS]${NC} $message"
    else
        echo -e "  ${RED}${BOLD}[FAILED]${NC} $message"
    fi
}

# Function to print menu
print_menu() {
    print_section "INSTALLATION MENU"
    
    echo -e "${CYAN}${BOLD} HARDWARE DRIVERS${NC}"
    echo -e " ${GREEN}1.${NC} NVIDIA Driver 470 (for K80 GPUs)"
    echo -e " ${GREEN}2.${NC} CUDA 11.8 Toolkit"
    echo -e " ${GREEN}3.${NC} ROCm 6.3.3 (for RX580 GPU)"
    echo ""
    
    echo -e "${CYAN}${BOLD} SYSTEM SETUP${NC}"
    echo -e " ${GREEN}4.${NC} Ubuntu Essentials"
    echo -e " ${GREEN}5.${NC} AI/ML Conda Environment"
    echo -e " ${GREEN}6.${NC} Docker Environment"
    echo -e " ${GREEN}7.${NC} Application Installer"
    echo -e " ${GREEN}8.${NC} Network & System Tools"
    echo ""
    
    echo -e "${CYAN}${BOLD} AI APPLICATIONS${NC}"
    echo -e " ${GREEN}9.${NC} Whisper TTS/STT"
    echo -e " ${GREEN}10.${NC} Open WebUI"
    echo -e " ${GREEN}11.${NC} ComfyUI + Flux.1 + SD"
    echo -e " ${GREEN}12.${NC} LM Studio"
    echo -e " ${GREEN}13.${NC} Ollama Multi-GPU Compiler"
    echo -e " ${GREEN}14.${NC} EXO Distributed Inference"
    echo ""
    
    echo -e "${CYAN}${BOLD} SPECIAL OPTIONS${NC}"
    echo -e " ${GREEN}A.${NC} Install All (Sequential)"
    echo -e " ${GREEN}S.${NC} Install System Stack (1-8)"  
    echo -e " ${GREEN}D.${NC} Install AI Stack (9-13)"
    echo -e " ${GREEN}Q.${NC} Quit"
    echo ""
}

# Function to check if a script was successfully executed
check_dependency_status() {
    local dependency_file="$SCRIPT_DIR/.installed_$1"
    if [ -f "$dependency_file" ]; then
        return 0 # Dependency installed
    else
        return 1 # Dependency not installed
    fi
}

# Function to mark a script as successfully executed
mark_as_installed() {
    local dependency_name="$1"
    touch "$SCRIPT_DIR/.installed_$dependency_name"
}

# Function to check for dependencies before running a script
check_dependencies() {
    local script_name="$1"
    local missing_deps=()
    
    case "$script_name" in
        "cuda-toolkit")  # CUDA depends on NVIDIA driver
            if ! check_dependency_status "nvidia-driver"; then
                missing_deps+=("NVIDIA Driver 470")
            fi
            ;;
        "conda-environments")  # AI/ML Conda depends on drivers
            if ! check_dependency_status "nvidia-driver" && ! check_dependency_status "rocm-toolkit"; then
                missing_deps+=("Either NVIDIA Driver or ROCm")
            fi
            ;;
        "docker")  # Docker depends on essentials
            if ! check_dependency_status "ubuntu-essentials"; then
                missing_deps+=("Ubuntu Essentials")
            fi
            ;;
        "whisper")  # Whisper depends on Conda
            if ! check_dependency_status "conda-environments"; then
                missing_deps+=("AI/ML Conda Environment")
            fi
            ;;
        "open-webui")  # Open WebUI depends on Conda
            if ! check_dependency_status "conda-environments"; then
                missing_deps+=("AI/ML Conda Environment")
            fi
            ;;
        "comfyui")  # ComfyUI depends on Conda
            if ! check_dependency_status "conda-environments"; then
                missing_deps+=("AI/ML Conda Environment")
            fi
            ;;
        "ollama-compiler")  # Ollama depends on drivers
            if ! check_dependency_status "nvidia-driver" && ! check_dependency_status "rocm-toolkit"; then
                missing_deps+=("Either NVIDIA Driver or ROCm")
            fi
            ;;
        "exo")  # EXO depends on drivers and Conda
            if ! check_dependency_status "conda-environments"; then
                missing_deps+=("AI/ML Conda Environment")
            fi
            if ! check_dependency_status "nvidia-driver" && ! check_dependency_status "rocm-toolkit"; then
                missing_deps+=("Either NVIDIA Driver or ROCm")
            fi
            if ! check_dependency_status "ollama-compiler"; then
                missing_deps+=("Ollama Multi-GPU Compiler")
            fi
            ;;
    esac
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${YELLOW}${BOLD}WARNING: Missing dependencies:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo -e "${YELLOW}- $dep${NC}"
        done
        echo -e "${YELLOW}Do you want to continue anyway? (y/N)${NC}"
        read -r confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            return 1 # Abort
        fi
    fi
    
    return 0 # Continue
}

# Function to run a script with proper output handling
run_script() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/${script_name}.sh"
    
    print_section "Installing $script_name"
    
    # Check if the script exists
    if [ ! -f "$script_path" ]; then
        echo -e "${RED}${BOLD}Error: Script ${script_path} not found!${NC}"
        return 1
    fi
    
    # Make sure the script is executable
    chmod +x "$script_path"
    
    # Run the script with real-time output in a box
    echo -e "${CYAN}${BOLD}Running ${script_name}.sh...${NC}"
    echo -e "${YELLOW}Detailed output will be saved to ${script_name}.log${NC}"
    echo ""
    
    # Create a named pipe for capturing output
    local pipe="/tmp/installer_pipe.$$"
    mkfifo "$pipe"
    
    # Start logging to both screen and file
    tee "$SCRIPT_DIR/${script_name}.log" < "$pipe" &
    local tee_pid=$!
    
    # Draw the initial box
    local width=80
    local box_top="╔═$( printf '═%.0s' $(seq 1 $((width-4))) )═╗"
    local box_bottom="╚═$( printf '═%.0s' $(seq 1 $((width-4))) )═╝"
    local box_line="║ %-$((width-4))s ║"
    local box_title="║ ${BOLD}%-$((width-6))s${NC}${CYAN} ║"
    
    echo -e "${CYAN}${box_top}${NC}"
    printf "${CYAN}${box_title}${NC}\n" "LIVE OUTPUT - ${script_name} installation"
    echo -e "${CYAN}╠═$( printf '═%.0s' $(seq 1 $((width-4))) )═╣${NC}"
    
    # Run the script and direct output to the pipe
    (
        # Process each line with timestamps and colors
        bash "$script_path" 2>&1 | while IFS= read -r line; do
            # Add color based on line content
            if [[ "$line" == *"Success"* || "$line" == *"✅"* ]]; then
                printf "${CYAN}║ ${GREEN}%-$((width-6))s${NC}${CYAN} ║${NC}\n" "${line:0:$((width-6))}"
            elif [[ "$line" == *"Error"* || "$line" == *"❌"* ]]; then
                printf "${CYAN}║ ${RED}%-$((width-6))s${NC}${CYAN} ║${NC}\n" "${line:0:$((width-6))}"
            elif [[ "$line" == *"Warning"* || "$line" == *"⚠️"* ]]; then
                printf "${CYAN}║ ${YELLOW}%-$((width-6))s${NC}${CYAN} ║${NC}\n" "${line:0:$((width-6))}"
            else
                printf "${CYAN}║ ${WHITE}%-$((width-6))s${NC}${CYAN} ║${NC}\n" "${line:0:$((width-6))}"
            fi
        done
        
        # Send the exit status to a temp file since we can't easily get it from a piped process
        local script_exit=$?
        echo $script_exit > "/tmp/exit_status.$$"
        
        # Add a summary line before closing the box
        echo -e "${CYAN}╠═$( printf '═%.0s' $(seq 1 $((width-4))) )═╣${NC}"
        if [ $script_exit -eq 0 ]; then
            printf "${CYAN}${box_title}${NC}\n" "${GREEN}✅ Installation completed successfully${NC}"
        else
            printf "${CYAN}${box_title}${NC}\n" "${RED}❌ Installation failed with exit code $script_exit${NC}"
        fi
        
        # Close the box
        echo -e "${CYAN}${box_bottom}${NC}"
    ) > "$pipe" &
    
    local script_pid=$!
    wait $script_pid
    
    # Get the exit status from the temp file
    local exit_status=$(cat "/tmp/exit_status.$$")
    rm -f "/tmp/exit_status.$$" "$pipe"
    
    # Kill the tee process
    kill $tee_pid 2>/dev/null
    
    # Check if script succeeded
    if [ $exit_status -eq 0 ]; then
        mark_as_installed "$script_name"
        print_status 0 "$script_name successfully installed"
    else
        print_status 1 "$script_name installation failed. Check ${script_name}.log for details"
    fi
    
    return $exit_status
}

# Function to prompt for continuation
prompt_continue() {
    echo ""
    echo -e "${YELLOW}${BOLD}Press Enter to continue...${NC}"
    read -r
}

# Main function to handle menu and installation
main() {
    local choice
    
    while true; do
        print_header
        print_menu
        
        echo -e "${CYAN}${BOLD}Enter your choice:${NC} "
        read -r choice
        
        case "$choice" in
            1)
                run_script "nvidia-driver"
                prompt_continue
                ;;
            2)
                run_script "cuda-toolkit"
                prompt_continue
                ;;
            3)
                run_script "rocm-toolkit"
                prompt_continue
                ;;
            4)
                run_script "ubuntu-essentials"
                prompt_continue
                ;;
            5)
                run_script "conda-environments"
                prompt_continue
                ;;
            6)
                run_script "docker"
                prompt_continue
                ;;
            7)
                run_script "applications"
                prompt_continue
                ;;
            8)
                run_script "network-tools"
                prompt_continue
                ;;
            9)
                run_script "whisper"
                prompt_continue
                ;;
            10)
                run_script "open-webui"
                prompt_continue
                ;;
            11)
                run_script "comfyui"
                prompt_continue
                ;;
            12)
                run_script "lmstudio"
                prompt_continue
                ;;
            13)
                run_script "ollama-compiler"
                prompt_continue
                ;;
            14)
                run_script "exo"
                prompt_continue
                ;;
            A|a)
                print_section "Installing All Components"
                
                # Hardware drivers
                run_script "nvidia-driver"
                run_script "cuda-toolkit"
                run_script "rocm-toolkit"
                
                # System setup
                run_script "ubuntu-essentials"
                run_script "conda-environments"
                run_script "docker"
                run_script "applications"
                run_script "network-tools"
                
                # AI apps
                run_script "whisper"
                run_script "open-webui"
                run_script "comfyui"
                run_script "lmstudio"
                run_script "ollama-compiler"
                run_script "exo"
                
                print_section "Complete Setup Finished!"
                prompt_continue
                ;;
            S|s)
                print_section "Installing System Stack"
                
                # Hardware drivers
                run_script "nvidia-driver"
                run_script "cuda-toolkit"
                run_script "rocm-toolkit"
                
                # System setup
                run_script "ubuntu-essentials"
                run_script "conda-environments"
                run_script "docker"
                run_script "applications"
                run_script "network-tools"
                
                print_section "System Stack Installation Complete!"
                prompt_continue
                ;;
            D|d)
                print_section "Installing AI Stack"
                
                # AI apps
                run_script "whisper"
                run_script "open-webui"
                run_script "comfyui"
                run_script "lmstudio"
                run_script "ollama-compiler"
                run_script "exo"
                
                print_section "AI Stack Installation Complete!"
                prompt_continue
                ;;
            Q|q)
                print_section "Exiting Installer"
                echo -e "${GREEN}${BOLD}Thank you for using the DARKPOOL AI/ML Installer!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}${BOLD}Invalid option. Please try again.${NC}"
                sleep 2
                ;;
        esac
    done
}

# Start the installer
main