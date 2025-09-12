#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"

# Clean up empty files from previous attempts
rm -f "$OUTPUT_DIR"/*.mp4

# Counter for progress
total_files=$(ls -1 "$INPUT_DIR"/*.mov 2>/dev/null | wc -l)
current_file=0

# Get CPU cores for parallel processing
cpu_cores=$(sysctl -n hw.ncpu)
threads=$((cpu_cores - 1)) # Leave one core free

echo "HEVC Encoding - Optimized for Intel Mac with AMD RX 580"
echo "========================================================"
echo "Total files to process: $total_files"
echo "Using $threads CPU threads for encoding"
echo ""
echo "NOTE: macOS doesn't expose AMD VCE to FFmpeg directly."
echo "Using optimized CPU encoding with multi-threading."
echo "----------------------------------------"

# Process each MOV file
for input_file in "$INPUT_DIR"/*.mov; do
    # Check if any MOV files exist
    [ -e "$input_file" ] || continue
    
    # Increment counter
    ((current_file++))
    
    # Get the filename without path
    filename=$(basename "$input_file")
    
    # Create output filename with .mp4 extension
    output_file="$OUTPUT_DIR/${filename%.mov}_HEVC.mp4"
    
    echo "[$current_file/$total_files] Processing: $filename"
    
    # Try VideoToolbox first (in case it works)
    echo "Attempting hardware encoding..."
    ffmpeg -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 6000k \
        -vf "scale=-2:1080" \
        -c:a copy \
        -map_metadata 0 \
        -tag:v hvc1 \
        "$output_file" \
        -y \
        -loglevel error \
        -stats 2>&1 | while IFS= read -r line; do
            if [[ $line == *"fps="* ]]; then
                echo -ne "\r$line"
            fi
        done
    
    # Check if encoding succeeded
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo "" # New line
        echo "✓ Hardware encoding successful!"
    else
        # Fallback to optimized software encoding
        echo ""
        echo "Hardware encoding failed, using optimized software encoding..."
        
        ffmpeg -i "$input_file" \
            -c:v libx265 \
            -preset faster \
            -crf 23 \
            -vf "scale=-2:1080" \
            -pix_fmt yuv420p \
            -x265-params "pools=$threads:frame-threads=2:log-level=error" \
            -c:a copy \
            -map_metadata 0 \
            -movflags +faststart \
            -tag:v hvc1 \
            "$output_file" \
            -y \
            -loglevel warning \
            -stats 2>&1 | while IFS= read -r line; do
                if [[ $line == *"fps="* ]]; then
                    echo -ne "\r$line"
                fi
            done
        
        echo "" # New line
        echo "✓ Software encoding completed"
    fi
    
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        # Get file sizes for comparison
        input_size=$(du -h "$input_file" | cut -f1)
        output_size=$(du -h "$output_file" | cut -f1)
        
        # Calculate compression ratio
        input_bytes=$(stat -f%z "$input_file" 2>/dev/null)
        output_bytes=$(stat -f%z "$output_file" 2>/dev/null)
        
        if [ -n "$input_bytes" ] && [ -n "$output_bytes" ] && [ "$input_bytes" -gt 0 ]; then
            ratio=$(echo "scale=1; ($input_bytes - $output_bytes) * 100 / $input_bytes" | bc)
            echo "  Size: $input_size → $output_size (${ratio}% reduction)"
        else
            echo "  Size: $input_size → $output_size"
        fi
    else
        echo "✗ Error processing: $filename"
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "Encoding complete!"
echo ""

# Show summary
successful_count=$(ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Summary:"
echo "Total files encoded successfully: $successful_count out of $total_files"

if [ $successful_count -gt 0 ]; then
    echo ""
    echo "Encoded files:"
    ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | awk '{printf "  %-60s %s\n", $9, $5}'
    
    echo ""
    total_input=$(du -ch "$INPUT_DIR"/*.mov 2>/dev/null | tail -1 | cut -f1)
    total_output=$(du -ch "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -1 | cut -f1)
    echo "Total size: $total_input → $total_output"
fi

echo ""
echo "Notes:"
echo "- VideoToolbox on Intel Macs primarily uses Intel Quick Sync"
echo "- AMD GPUs aren't directly supported for encoding in macOS"
echo "- Using optimized multi-threaded encoding for best performance"