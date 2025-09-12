#!/bin/bash

# HEVC GPU Video Encoder
# Version: 2.0.0
# Last Updated: 2025-01-06
# Hardware: macOS VideoToolbox/Metal acceleration

set -euo pipefail

# Configuration
readonly DEFAULT_INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
readonly DEFAULT_OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"
readonly SUPPORTED_FORMATS=("*.mov" "*.mp4" "*.avi" "*.mkv")
readonly LOG_FILE="/tmp/hevc_encode_$(date +%Y%m%d_%H%M%S).log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging functions
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
    # Check if running on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        error_exit "This script requires macOS for VideoToolbox/Metal acceleration"
    fi
    
    # Check if ffmpeg is available
    if ! command -v ffmpeg &> /dev/null; then
        error_exit "ffmpeg is required but not installed. Install with: brew install ffmpeg"
    fi
    
    # Check ffmpeg VideoToolbox support
    if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q hevc_videotoolbox; then
        error_exit "ffmpeg does not support hevc_videotoolbox encoder"
    fi
    
    success "Prerequisites check passed"
}

# Get directory paths with validation
get_directories() {
    # Use provided arguments or defaults
    INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
    OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
    
    # Validate input directory
    if [[ ! -d "$INPUT_DIR" ]]; then
        error_exit "Input directory does not exist: $INPUT_DIR"
    fi
    
    if [[ ! -r "$INPUT_DIR" ]]; then
        error_exit "Input directory is not readable: $INPUT_DIR"
    fi
    
    # Create output directory if it doesn't exist
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        log "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR" || error_exit "Failed to create output directory: $OUTPUT_DIR"
    fi
    
    success "Using INPUT_DIR: $INPUT_DIR"
    success "Using OUTPUT_DIR: $OUTPUT_DIR"
}

# Count supported files
count_files() {
    local count=0
    for pattern in "${SUPPORTED_FORMATS[@]}"; do
        while IFS= read -r -d '' file; do
            ((count++))
        done < <(find "$INPUT_DIR" -maxdepth 1 -name "$pattern" -print0 2>/dev/null)
    done
    echo "$count"
}

log "Starting HEVC GPU encoding process..."
log "Log file: $LOG_FILE"

check_prerequisites
get_directories "$@"

# Count total files
total_files=$(count_files)
current_file=0

if [[ $total_files -eq 0 ]]; then
    warning "No supported video files found in: $INPUT_DIR"
    echo "Supported formats: ${SUPPORTED_FORMATS[*]}"
    exit 0
fi

log "Found $total_files files to process"
echo -e "${BLUE}📊 Processing $total_files files with VideoToolbox/Metal acceleration${NC}"
echo -e "${YELLOW}💡 Monitor GPU usage in Activity Monitor > Window > GPU History${NC}"
echo "=========================================="

# Process files with enhanced error handling
process_file() {
    local input_file="$1"
    local file_num="$2"
    local filename
    filename=$(basename "$input_file")
    
    # Determine output extension and filename
    local base_name="${filename%.*}"
    local output_file="$OUTPUT_DIR/${base_name}_HEVC.mp4"
    
    # Skip if output already exists
    if [[ -f "$output_file" ]]; then
        warning "Output file already exists, skipping: $filename"
        return 0
    fi
    
    log "[$file_num/$total_files] Processing: $filename"
    
    # Get input file info
    local input_size_bytes
    input_size_bytes=$(stat -f%z "$input_file" 2>/dev/null || echo "0")
    
    if [[ $input_size_bytes -eq 0 ]]; then
        warning "Input file is empty or unreadable: $filename"
        return 1
    fi
    
    # Encode with enhanced settings and error handling
    if ffmpeg -hwaccel videotoolbox \
        -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 8M \
        -maxrate 12M \
        -pix_fmt nv12 \
        -vf "scale_vt=-2:1080" \
        -profile:v main \
        -level:v 4.1 \
        -c:a copy \
        -map_metadata 0 \
        -movflags +faststart \
        -tag:v hvc1 \
        "$output_file" \
        -y \
        -loglevel error \
        -progress pipe:1 2>>"$LOG_FILE" | grep -E "frame=|speed=" || true; then
        
        # Verify output file was created and has reasonable size
        if [[ -f "$output_file" ]]; then
            local output_size_bytes
            output_size_bytes=$(stat -f%z "$output_file" 2>/dev/null || echo "0")
            
            if [[ $output_size_bytes -gt 0 ]]; then
                # Calculate compression statistics
                local input_size_human output_size_human compression_ratio
                input_size_human=$(du -h "$input_file" | cut -f1)
                output_size_human=$(du -h "$output_file" | cut -f1)
                
                if command -v bc &> /dev/null && [[ $input_size_bytes -gt 0 ]]; then
                    compression_ratio=$(echo "scale=1; ($input_size_bytes - $output_size_bytes) * 100 / $input_size_bytes" | bc)
                else
                    compression_ratio="N/A"
                fi
                
                success "$filename → Size: $input_size_human → $output_size_human (${compression_ratio}% reduction)"
                return 0
            else
                warning "Output file is empty: $filename"
                rm -f "$output_file"
                return 1
            fi
        else
            warning "Output file was not created: $filename"
            return 1
        fi
    else
        warning "FFmpeg encoding failed for: $filename"
        rm -f "$output_file" 2>/dev/null || true
        return 1
    fi
}

# Process all supported files
successful_files=0
failed_files=0

for pattern in "${SUPPORTED_FORMATS[@]}"; do
    while IFS= read -r -d '' input_file; do
        ((current_file++))
        
        if process_file "$input_file" "$current_file"; then
            ((successful_files++))
        else
            ((failed_files++))
        fi
        
        echo "=========================================="
    done < <(find "$INPUT_DIR" -maxdepth 1 -name "$pattern" -print0 2>/dev/null)
done

# Final summary
echo -e "\n${BLUE}📊 ENCODING SUMMARY${NC}"
echo "==========================================="
echo "Total files processed: $current_file"
echo "Successful: $successful_files"
echo "Failed: $failed_files"

if [[ $successful_files -gt 0 ]]; then
    echo -e "\n${GREEN}✅ Encoded files location: $OUTPUT_DIR${NC}"
    
    # Calculate total size savings if possible
    local total_input_size total_output_size
    if command -v du &> /dev/null; then
        # Get sizes of processed files
        total_input_size=$(find "$INPUT_DIR" -name "*.*" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1 || echo "N/A")
        total_output_size=$(find "$OUTPUT_DIR" -name "*_HEVC.mp4" -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1 || echo "N/A")
        
        if [[ "$total_input_size" != "N/A" && "$total_output_size" != "N/A" ]]; then
            echo "Total size before encoding: $total_input_size"
            echo "Total size after encoding: $total_output_size"
        fi
    fi
    
    echo -e "\n${YELLOW}💡 GPU acceleration info:${NC}"
    echo "- Used VideoToolbox/Metal hardware acceleration"
    echo "- Monitor GPU usage: Activity Monitor > Window > GPU History"
    echo "- Hardware encoder: hevc_videotoolbox"
fi

if [[ $failed_files -gt 0 ]]; then
    echo -e "\n${RED}⚠️  Some files failed to encode${NC}"
    echo "Check log file for details: $LOG_FILE"
fi

echo -e "\n${GREEN}✅ HEVC encoding process completed${NC}"
log "Encoding completed - Success: $successful_files, Failed: $failed_files"