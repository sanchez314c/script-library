#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/H264_1080p"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Counter for progress
total_files=$(ls -1 "$INPUT_DIR"/*.mov 2>/dev/null | wc -l)
current_file=0

echo "H.264 Encoding with VideoToolbox (Better Intel Mac GPU Support)"
echo "=============================================================="
echo "Total files to process: $total_files"
echo ""
echo "NOTE: H.264 VideoToolbox has better GPU support on Intel Macs"
echo "Monitor GPU usage in Activity Monitor > Window > GPU History"
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
    output_file="$OUTPUT_DIR/${filename%.mov}_H264.mp4"
    
    echo "[$current_file/$total_files] Processing: $filename"
    
    # Encode with H.264 VideoToolbox (better GPU support)
    ffmpeg -hwaccel videotoolbox \
        -i "$input_file" \
        -c:v h264_videotoolbox \
        -b:v 8000k \
        -vf "scale=-2:1080" \
        -c:a copy \
        -map_metadata 0 \
        -movflags +faststart \
        "$output_file" \
        -y \
        -loglevel warning \
        -stats 2>&1 | while IFS= read -r line; do
            if [[ $line == *"fps="* ]]; then
                echo -ne "\r$line"
            fi
        done
    
    echo "" # New line
    
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        # Get file sizes for comparison
        input_size=$(du -h "$input_file" | cut -f1)
        output_size=$(du -h "$output_file" | cut -f1)
        
        # Calculate compression ratio
        input_bytes=$(stat -f%z "$input_file" 2>/dev/null)
        output_bytes=$(stat -f%z "$output_file" 2>/dev/null)
        
        if [ -n "$input_bytes" ] && [ -n "$output_bytes" ] && [ "$input_bytes" -gt 0 ]; then
            ratio=$(echo "scale=1; ($input_bytes - $output_bytes) * 100 / $input_bytes" | bc)
            echo "✓ Success: $filename"
            echo "  Size: $input_size → $output_size (${ratio}% reduction)"
            echo "  Using H.264 VideoToolbox (GPU accelerated)"
        else
            echo "✓ Success: $filename ($input_size → $output_size)"
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
echo "- H.264 VideoToolbox has better GPU support than HEVC on Intel Macs"
echo "- Files are slightly larger than HEVC but encode much faster"
echo "- Check Activity Monitor for 'VTDecoderXPCService' process"