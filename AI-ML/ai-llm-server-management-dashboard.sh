#!/bin/bash

# LLM Server Service Manager
# Created by Cortana for Jason
# Date: February 25, 2025

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Service groups
OLLAMA_SERVICES="ollama-cpu ollama-rocm ollama-cuda0 ollama-cuda1"
INFERENCE_SERVICES="exo-cluster open-webui"
UI_SERVICES="comfyui whisper-stt gpu-monitor"
ALL_SERVICES="$OLLAMA_SERVICES $INFERENCE_SERVICES $UI_SERVICES"

print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${CYAN}             LLM Server Service Manager             ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
}

print_status() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${CYAN}                   Service Status                   ${BLUE}║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════╣${NC}"
    
    # Check Ollama services
    echo -e "${BLUE}║${YELLOW} Ollama Services:                                  ${BLUE}║${NC}"
    for service in $OLLAMA_SERVICES; do
        status=$(systemctl is-active $service 2>/dev/null)
        if [ "$status" = "active" ]; then
            echo -e "${BLUE}║${NC}  - $service: ${GREEN}●${NC} active                             ${BLUE}║${NC}"
        else
            echo -e "${BLUE}║${NC}  - $service: ${RED}○${NC} inactive                           ${BLUE}║${NC}"
        fi
    done
    
    # Check Inference services
    echo -e "${BLUE}║${YELLOW} Inference Services:                               ${BLUE}║${NC}"
    for service in $INFERENCE_SERVICES; do
        status=$(systemctl is-active $service 2>/dev/null)
        if [ "$status" = "active" ]; then
            echo -e "${BLUE}║${NC}  - $service: ${GREEN}●${NC} active                           ${BLUE}║${NC}"
        else
            echo -e "${BLUE}║${NC}  - $service: ${RED}○${NC} inactive                         ${BLUE}║${NC}"
        fi
    done
    
    # Check UI services
    echo -e "${BLUE}║${YELLOW} UI Services:                                      ${BLUE}║${NC}"
    for service in $UI_SERVICES; do
        status=$(systemctl is-active $service 2>/dev/null)
        if [ "$status" = "active" ]; then
            echo -e "${BLUE}║${NC}  - $service: ${GREEN}●${NC} active                            ${BLUE}║${NC}"
        else
            echo -e "${BLUE}║${NC}  - $service: ${RED}○${NC} inactive                          ${BLUE}║${NC}"
        fi
    done
    
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
}

print_gpu_status() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${CYAN}                    GPU Status                      ${BLUE}║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════╣${NC}"
    
    # Check NVIDIA GPUs
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}║${YELLOW} NVIDIA GPUs:                                     ${BLUE}║${NC}"
        nvidia_info=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
        while IFS=',' read -r idx name util mem_used mem_total || [ -n "$idx" ]; do
            echo -e "${BLUE}║${NC}  GPU $idx: ${name:0:20} ${CYAN}${util}%${NC} util, ${CYAN}${mem_used}/${mem_total}${NC} MB    ${BLUE}║${NC}"
        done <<< "$nvidia_info"
    else
        echo -e "${BLUE}║${NC}  No NVIDIA GPUs detected or nvidia-smi not found  ${BLUE}║${NC}"
    fi
    
    # Check AMD GPUs
    if command -v rocm-smi &> /dev/null; then
        echo -e "${BLUE}║${YELLOW} AMD GPUs:                                        ${BLUE}║${NC}"
        amd_info=$(rocm-smi --showuse --showmemuse --csv)
        # Skip header
        amd_info=$(echo "$amd_info" | tail -n +2)
        while IFS=, read -r idx _ util _ mem_used mem_total _ || [ -n "$idx" ]; do
            echo -e "${BLUE}║${NC}  GPU $idx: RX580 ${CYAN}${util}%${NC} util, ${CYAN}${mem_used}/${mem_total}${NC} MB        ${BLUE}║${NC}"
        done <<< "$amd_info"
    else
        echo -e "${BLUE}║${NC}  No AMD GPUs detected or rocm-smi not found        ${BLUE}║${NC}"
    fi
    
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
}

print_menu() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${CYAN}                      Menu                          ${BLUE}║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}1.${NC} Start all services                             ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}2.${NC} Stop all services                              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}3.${NC} Restart all services                           ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}4.${NC} Start Ollama services                          ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}5.${NC} Start inference services                       ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}6.${NC} Start UI services                              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}7.${NC} View service logs                              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}8.${NC} Open web interfaces                            ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}9.${NC} Check model storage                            ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}0.${NC} Exit                                           ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
    echo -e "${CYAN}Enter your choice:${NC} "
}

start_services() {
    local services=$1
    echo -e "${CYAN}Starting services: $services${NC}"
    for service in $services; do
        echo -n "Starting $service... "
        sudo systemctl start $service 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    done
}

stop_services() {
    local services=$1
    echo -e "${CYAN}Stopping services: $services${NC}"
    for service in $services; do
        echo -n "Stopping $service... "
        sudo systemctl stop $service 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    done
}

restart_services() {
    local services=$1
    echo -e "${CYAN}Restarting services: $services${NC}"
    for service in $services; do
        echo -n "Restarting $service... "
        sudo systemctl restart $service 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    done
}

view_logs() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${CYAN}                 Service Logs                      ${BLUE}║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}1.${NC} Ollama CPU                                    ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}2.${NC} Ollama ROCm                                   ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}3.${NC} Ollama CUDA (GPU 0)                           ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}4.${NC} Ollama CUDA (GPU 1)                           ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}5.${NC} EXO Cluster                                   ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}6.${NC} Open WebUI                                    ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}7.${NC} ComfyUI                                       ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}8.${NC} Whisper STT                                   ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}0.${NC} Back                                          ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
    echo -e "${CYAN}Enter your choice:${NC} "
    read choice
    
    case $choice in
        1) sudo journalctl -u ollama-cpu -f ;;
        2) sudo journalctl -u ollama-rocm -f ;;
        3) sudo journalctl -u ollama-cuda0 -f ;;
        4) sudo journalctl -u ollama-cuda1 -f ;;
        5) sudo journalctl -u exo-cluster -f ;;
        6) sudo journalctl -u open-webui -f ;;
        7) sudo journalctl -u comfyui -f ;;
        8) sudo journalctl -u whisper-stt -f ;;
        0) return ;;
        *) echo -e "${RED}Invalid choice${NC}" ;;
    esac
}

open_web_interfaces() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${CYAN}                Web Interfaces                     ${BLUE}║${NC}"
    echo -e "${BLUE}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}1.${NC} Open WebUI (http://localhost:3000)              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}2.${NC} ComfyUI (http://localhost:8188)                 ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}3.${NC} GPU Monitor (http://localhost:8484)             ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}4.${NC} EXO Dashboard (http://localhost:8080)           ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}5.${NC} Ollama CPU API (http://localhost:11434)         ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} ${MAGENTA}6.${NC} Ollama ROCm API (http://localhost:11435)        ${BLUE}║${NC}"
    echo -e "${