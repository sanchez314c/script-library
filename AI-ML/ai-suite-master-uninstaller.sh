#!/bin/bash
# Master Interactive Uninstaller for AI/ML Development Environment
# Created by Claude for Jason
# Date: March 2, 2025
#
# This script provides an interactive way to uninstall all components
# of the AI/ML environment in the correct order (reverse installation order).

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

# Determine the correct username: prefer SUDO_USER, then LOGNAME, then whoami
TARGET_USER="${SUDO_USER:-${LOGNAME:-$(whoami)}}"
if [ "$TARGET_USER" = "root" ]; then
    echo "⚠️ Warning: Running as root, trying to guess the real user..."
    # Try to find a non-root user from /home directory
    FIRST_USER=$(ls -1 /home | head -n 1)
    if [ -n "$FIRST_USER" ]; then
        TARGET_USER="$FIRST_USER"
        echo "ℹ️ Using first user found in /home: $TARGET_USER"
    fi
fi

# Function to print ASCII art header
print_header() {
    clear
    echo -e "${RED}${BOLD}"
    cat << "EOF"
██╗   ██╗███╗   ██╗██╗███╗   ██╗███████╗████████╗ █████╗ ██╗     ██╗     
██║   ██║████╗  ██║██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██║     ██║     
██║   ██║██╔██╗ ██║██║██╔██╗ ██║███████╗   ██║   ███████║██║     ██║     
██║   ██║██║╚██╗██║██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║     ██║     
╚██████╔╝██║ ╚████║██║██║ ╚████║███████║   ██║   ██║  ██║███████╗███████╗
 ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝
                                                                          
██████╗  █████╗ ██████╗ ██╗  ██╗██████╗  ██████╗  ██████╗ ██╗            
██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝██╔══██╗██╔═══██╗██╔═══██╗██║            
██║  ██║███████║██████╔╝█████╔╝ ██████╔╝██║   ██║██║   ██║██║            
██║  ██║██╔══██║██╔══██╗██╔═██╗ ██╔═══╝ ██║   ██║██║   ██║██║            
██████╔╝██║  ██║██║  ██║██║  ██╗██║     ╚██████╔╝╚██████╔╝███████╗       
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝  ╚═════╝ ╚══════╝       
EOF
    echo -e "${NC}"
    echo -e "${YELLOW}${BOLD}AI/ML Environment Uninstallation System${NC}"
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

# Function to show spinner during uninstallation
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
    print_section "UNINSTALLATION MENU"
    
    echo -e "${CYAN}${BOLD} AI APPLICATIONS${NC}"
    echo -e " ${RED}14.${NC} Ollama Custom Builds"
    echo -e " ${RED}13.${NC} EXO Distributed Inference"
    echo -e " ${RED}12.${NC} LM Studio"
    echo -e " ${RED}11.${NC} ComfyUI + Flux.1 + SD"
    echo -e " ${RED}10.${NC} Open WebUI"
    echo -e " ${RED}9.${NC} Whisper TTS/STT"
    echo ""
    
    echo -e "${CYAN}${BOLD} SYSTEM SETUP${NC}"
    echo -e " ${RED}8.${NC} Network & System Tools"
    echo -e " ${RED}7.${NC} Application Installer"
    echo -e " ${RED}6.${NC} Docker Environment"
    echo -e " ${RED}5.${NC} AI/ML Conda Environment"
    echo -e " ${RED}4.${NC} Ubuntu Essentials"
    echo ""
    
    echo -e "${CYAN}${BOLD} HARDWARE DRIVERS${NC}"
    echo -e " ${RED}3.${NC} ROCm 6.3.3 (for RX580 GPU)"
    echo -e " ${RED}2.${NC} CUDA 11.8 Toolkit"
    echo -e " ${RED}1.${NC} NVIDIA Driver 470 (for K80 GPUs)"
    echo ""
    
    echo -e "${CYAN}${BOLD} SPECIAL OPTIONS${NC}"
    echo -e " ${RED}A.${NC} Uninstall All (Sequential)"
    echo -e " ${RED}D.${NC} Uninstall AI Stack (13-9)"  
    echo -e " ${RED}S.${NC} Uninstall System Stack (8-4)"
    echo -e " ${RED}H.${NC} Uninstall Hardware Stack (3-1)"
    echo -e " ${RED}Q.${NC} Quit"
    echo ""
}

# Function to uninstall Ollama compiler outputs
uninstall_ollama_compiler() {
    print_section "Uninstalling Ollama Custom Builds"
    
    # Stop and disable services
    echo "🛑 Stopping Ollama services..."
    systemctl stop ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1 2>/dev/null || true
    systemctl disable ollama-rocm ollama-k80-gpu0 ollama-k80-gpu1 2>/dev/null || true
    
    # Remove service files
    echo "🗑️ Removing systemd service files..."
    rm -f /etc/systemd/system/ollama-rocm.service
    rm -f /etc/systemd/system/ollama-k80-gpu0.service
    rm -f /etc/systemd/system/ollama-k80-gpu1.service
    systemctl daemon-reload
    
    # Remove Ollama binaries
    echo "🧹 Removing Ollama binaries..."
    rm -f /usr/local/bin/ollama-rocm
    rm -f /usr/local/bin/ollama-k80-gpu0
    rm -f /usr/local/bin/ollama-k80-gpu1
    
    # Remove Ollama models and configs
    echo "🧹 Removing Ollama models and configs..."
    rm -rf /usr/share/ollama-rocm
    rm -rf /usr/share/ollama-k80-gpu0
    rm -rf /usr/share/ollama-k80-gpu1
    
    # Remove build directories
    echo "🧹 Removing build directories..."
    sudo -u "$TARGET_USER" rm -rf "/home/$TARGET_USER/AI/Ollama-build"
    
    print_status 0 "Ollama Custom Builds uninstalled"
}

# Function to uninstall EXO Distributed Inference
uninstall_exo() {
    print_section "Uninstalling EXO Distributed Inference"
    
    # Stop and disable services
    echo "🛑 Stopping EXO cluster..."
    systemctl stop exo-cluster 2>/dev/null || true
    systemctl disable exo-cluster 2>/dev/null || true
    
    # Remove service files
    echo "🗑️ Removing systemd service files..."
    rm -f /etc/systemd/system/exo-cluster.service
    systemctl daemon-reload
    
    # Remove EXO installation
    echo "🧹 Removing EXO installation..."
    sudo -u "$TARGET_USER" rm -rf "/home/$TARGET_USER/AI/EXO" "/home/$TARGET_USER/.config/exo"
    
    print_status 0 "EXO Distributed Inference uninstalled"
}

# Function to uninstall LM Studio
uninstall_lmstudio() {
    print_section "Uninstalling LM Studio"
    
    # Remove AppImage
    echo "🗑️ Removing LM Studio AppImage..."
    rm -f /usr/local/bin/lmstudio
    rm -f "/home/$TARGET_USER/Desktop/LM Studio.desktop"
    
    print_status 0 "LM Studio uninstalled"
}

# Function to uninstall ComfyUI
uninstall_comfyui() {
    print_section "Uninstalling ComfyUI and Flux.1"
    
    # Stop and disable service
    echo "🛑 Stopping ComfyUI service..."
    systemctl stop comfyui 2>/dev/null || true
    systemctl disable comfyui 2>/dev/null || true
    
    # Remove service file
    echo "🗑️ Removing systemd service file..."
    rm -f /etc/systemd/system/comfyui.service
    systemctl daemon-reload
    
    # Remove ComfyUI installation
    echo "🧹 Removing ComfyUI installation..."
    sudo -u "$TARGET_USER" rm -rf "/home/$TARGET_USER/AI/ComfyUI"
    
    print_status 0 "ComfyUI and Flux.1 uninstalled"
}

# Function to uninstall Open WebUI
uninstall_openwebui() {
    print_section "Uninstalling Open WebUI"
    
    # Stop and disable service
    echo "🛑 Stopping Open WebUI service..."
    systemctl stop open-webui 2>/dev/null || true
    systemctl disable open-webui 2>/dev/null || true
    
    # Remove service file
    echo "🗑️ Removing systemd service file..."
    rm -f /etc/systemd/system/open-webui.service
    systemctl daemon-reload
    
    # Remove Open WebUI installation
    echo "🧹 Removing Open WebUI installation..."
    sudo -u "$TARGET_USER" rm -rf "/home/$TARGET_USER/AI/OpenWeb-UI"
    
    print_status 0 "Open WebUI uninstalled"
}

# Function to uninstall Whisper
uninstall_whisper() {
    print_section "Uninstalling Whisper TTS/STT"
    
    # Remove binaries
    echo "🗑️ Removing Whisper binaries..."
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" activate darklake 2>/dev/null
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/pip" uninstall -y openai-whisper whisper-timestamped ffmpeg-python 2>/dev/null || true
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" deactivate 2>/dev/null
    
    # Remove Whisper models and scripts
    echo "🧹 Removing Whisper models and scripts..."
    sudo -u "$TARGET_USER" rm -rf "/home/$TARGET_USER/AI/Whisper"
    
    print_status 0 "Whisper TTS/STT uninstalled"
}

# Function to uninstall Network & System Tools
uninstall_network_tools() {
    print_section "Uninstalling Network & System Tools"
    
    # Disable firewall changes
    echo "🛡️ Resetting firewall..."
    ufw reset -y 2>/dev/null || true
    ufw disable 2>/dev/null || true
    
    # Remove monitoring tools
    echo "📊 Removing monitoring tools..."
    apt remove -y --purge glances neofetch netdata 2>/dev/null || true
    
    print_status 0 "Network & System Tools uninstalled"
}

# Function to uninstall Applications
uninstall_applications() {
    print_section "Uninstalling Applications"
    
    # Remove installed applications
    echo "🗑️ Removing applications..."
    apt remove -y --purge vlc gimp inkscape obs-studio audacity 2>/dev/null || true
    
    apt autoremove -y
    apt clean
    
    print_status 0 "Applications uninstalled"
}

# Function to uninstall Docker
uninstall_docker() {
    print_section "Uninstalling Docker"
    
    # Stop and disable Docker
    echo "🛑 Stopping Docker services..."
    systemctl stop docker docker.socket containerd 2>/dev/null || true
    systemctl disable docker docker.socket containerd 2>/dev/null || true
    
    # Remove Docker packages
    echo "🗑️ Removing Docker packages..."
    apt remove -y --purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>/dev/null || true
    
    # Remove Docker data
    echo "🧹 Removing Docker data..."
    rm -rf /var/lib/docker /var/lib/containerd
    
    # Remove user from Docker group
    echo "👤 Removing user from Docker group..."
    gpasswd -d "$TARGET_USER" docker 2>/dev/null || true
    
    apt autoremove -y
    apt clean
    
    print_status 0 "Docker uninstalled"
}

# Function to uninstall Conda environments
uninstall_conda() {
    print_section "Uninstalling AI/ML Conda Environments"
    
    # Remove conda environments
    echo "🧹 Removing Conda environments..."
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" env remove -n darklake 2>/dev/null || true
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" env remove -n darkpool-rocm 2>/dev/null || true
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" env remove -n darkpool-cuda0 2>/dev/null || true
    sudo -u "$TARGET_USER" "/home/$TARGET_USER/miniconda3/bin/conda" env remove -n darkpool-cuda1 2>/dev/null || true
    
    # Remove Miniconda
    echo "🗑️ Removing Miniconda installation..."
    sudo -u "$TARGET_USER" rm -rf "/home/$TARGET_USER/miniconda3"
    
    # Clean up .bashrc
    echo "🧹 Cleaning up .bashrc..."
    sudo -u "$TARGET_USER" sed -i '/# CUDA and ROCm Environment Variables/,/\[ -f "$HOME\/miniconda3\/etc\/profile.d\/conda.sh" \]/d' "/home/$TARGET_USER/.bashrc" 2>/dev/null || true
    sudo -u "$TARGET_USER" sed -i '/# Conda Aliases/,/alias dl="conda deactivate"/d' "/home/$TARGET_USER/.bashrc" 2>/dev/null || true
    
    print_status 0 "AI/ML Conda Environments uninstalled"
}

# Function to uninstall Ubuntu Essentials
uninstall_ubuntu_essentials() {
    print_section "Uninstalling Ubuntu Essentials"
    
    # Stop services
    echo "🛑 Stopping services..."
    systemctl stop ssh smbd nmbd 2>/dev/null || true
    systemctl disable ssh smbd nmbd 2>/dev/null || true
    
    # Remove modern tools
    echo "🗑️ Removing modern CLI tools..."
    apt remove -y --purge bat eza fd-find ripgrep fzf 2>/dev/null || true
    
    # Clean up symlinks
    echo "🧹 Cleaning up symlinks..."
    sudo -u "$TARGET_USER" rm -f "/home/$TARGET_USER/.local/bin/bat" "/home/$TARGET_USER/.local/bin/fd" 2>/dev/null || true
    
    # Remove Samba config
    echo "🗑️ Removing Samba configuration..."
    rm -rf /samba
    rm -f "/media/$TARGET_USER/llmRAID" 2>/dev/null || true
    
    # Clean up .bashrc
    echo "🧹 Cleaning up .bashrc..."
    sudo -u "$TARGET_USER" sed -i '/# Modern CLI Aliases/,/alias monitor="glances"/d' "/home/$TARGET_USER/.bashrc" 2>/dev/null || true
    
    apt autoremove -y
    apt clean
    
    print_status 0 "Ubuntu Essentials uninstalled"
}

# Function to uninstall ROCm
uninstall_rocm() {
    print_section "Uninstalling ROCm 6.3.3"
    
    # Remove ROCm packages
    echo "🗑️ Removing ROCm packages..."
    amdgpu-uninstall -a || true
    
    # Remove ROCm repo
    echo "🧹 Removing ROCm repository..."
    rm -f /etc/apt/sources.list.d/rocm.list
    rm -f /etc/apt/keyrings/rocm.gpg
    rm -f /etc/apt/sources.list.d/amdgpu.list
    
    # Remove environment variables
    echo "🧹 Cleaning up environment..."
    rm -f /etc/environment.d/rocm.conf
    
    # Remove library paths
    echo "🧹 Removing library paths..."
    rm -f /etc/ld.so.conf.d/rocm.conf
    ldconfig
    
    apt update
    apt autoremove -y
    apt clean
    
    print_status 0 "ROCm 6.3.3 uninstalled"
}

# Function to uninstall CUDA
uninstall_cuda() {
    print_section "Uninstalling CUDA 11.8"
    
    # Remove CUDA
    echo "🗑️ Removing CUDA Toolkit..."
    if [ -f /usr/local/cuda-11.8/bin/cuda-uninstaller ]; then
        /usr/local/cuda-11.8/bin/cuda-uninstaller || true
    fi
    
    # Clean up remaining files
    echo "🧹 Cleaning up CUDA files..."
    rm -rf /usr/local/cuda-11.8
    rm -f /usr/local/cuda
    
    # Remove environment variables
    echo "🧹 Cleaning up environment..."
    sed -i '/CUDA_HOME/d' /etc/environment
    
    # Remove library paths
    echo "🧹 Removing library paths..."
    rm -f /etc/ld.so.conf.d/cuda-11.8.conf
    ldconfig
    
    print_status 0 "CUDA 11.8 uninstalled"
}

# Function to uninstall NVIDIA Drivers
uninstall_nvidia_driver() {
    print_section "Uninstalling NVIDIA Driver 470"
    
    # Remove NVIDIA drivers
    echo "🗑️ Removing NVIDIA drivers..."
    apt remove --purge -y '^nvidia-.*' 2>/dev/null || true
    
    # Run NVIDIA uninstaller if available
    if command -v nvidia-uninstall &>/dev/null; then
        echo "🧹 Running NVIDIA uninstaller..."
        nvidia-uninstall --silent || true
    fi
    
    apt autoremove -y
    apt clean
    
    print_status 0 "NVIDIA Driver 470 uninstalled"
}

# Function to prompt for continuation
prompt_continue() {
    echo ""
    echo -e "${YELLOW}${BOLD}Press Enter to continue...${NC}"
    read -r
}

# Main function to handle menu and uninstallation
main() {
    local choice
    
    while true; do
        print_header
        print_menu
        
        echo -e "${CYAN}${BOLD}Enter your choice (to uninstall):${NC} "
        read -r choice
        
        case "$choice" in
            14)
                uninstall_ollama_compiler
                prompt_continue
                ;;
            13)
                uninstall_exo
                prompt_continue
                ;;
            12)
                uninstall_lmstudio
                prompt_continue
                ;;
            11)
                uninstall_comfyui
                prompt_continue
                ;;
            10)
                uninstall_openwebui
                prompt_continue
                ;;
            9)
                uninstall_whisper
                prompt_continue
                ;;
            8)
                uninstall_network_tools
                prompt_continue
                ;;
            7)
                uninstall_applications
                prompt_continue
                ;;
            6)
                uninstall_docker
                prompt_continue
                ;;
            5)
                uninstall_conda
                prompt_continue
                ;;
            4)
                uninstall_ubuntu_essentials
                prompt_continue
                ;;
            3)
                uninstall_rocm
                prompt_continue
                ;;
            2)
                uninstall_cuda
                prompt_continue
                ;;
            1)
                uninstall_nvidia_driver
                prompt_continue
                ;;
            A|a)
                print_section "Uninstalling All Components"
                uninstall_ollama_compiler
                uninstall_exo
                uninstall_lmstudio
                uninstall_comfyui
                uninstall_openwebui
                uninstall_whisper
                uninstall_network_tools
                uninstall_applications
                uninstall_docker
                uninstall_conda
                uninstall_ubuntu_essentials
                uninstall_rocm
                uninstall_cuda
                uninstall_nvidia_driver
                print_section "Complete Uninstallation Finished!"
                prompt_continue
                ;;
            D|d)
                print_section "Uninstalling AI Stack"
                uninstall_ollama_compiler
                uninstall_exo
                uninstall_lmstudio
                uninstall_comfyui
                uninstall_openwebui
                uninstall_whisper
                print_section "AI Stack Uninstallation Complete!"
                prompt_continue
                ;;
            S|s)
                print_section "Uninstalling System Stack"
                uninstall_network_tools
                uninstall_applications
                uninstall_docker
                uninstall_conda
                uninstall_ubuntu_essentials
                print_section "System Stack Uninstallation Complete!"
                prompt_continue
                ;;
            H|h)
                print_section "Uninstalling Hardware Stack"
                uninstall_rocm
                uninstall_cuda
                uninstall_nvidia_driver
                print_section "Hardware Stack Uninstallation Complete!"
                prompt_continue
                ;;
            Q|q)
                print_section "Exiting Uninstaller"
                echo -e "${GREEN}${BOLD}Thank you for using the DARKPOOL AI/ML Uninstaller!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}${BOLD}Invalid option. Please try again.${NC}"
                sleep 2
                ;;
        esac
    done
}

# Start the uninstaller
main