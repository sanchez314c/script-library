#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"

# Clean up empty files from previous attempts
rm -f "$OUTPUT_DIR"/*.mp4

# Counter for progress
total_files=$(ls -1 "$INPUT_DIR"/*.mov 2>/dev/null | wc -l)
current_file=0

echo "HEVC Encoding with VideoToolbox Hardware Acceleration"
echo "===================================================="
echo "Total files to process: $total_files"
echo ""
echo "Your AMD RX 580 supports hardware encoding via VideoToolbox!"
echo "Watch Activity Monitor > Window > GPU History during encoding"
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
    
    # Use VideoToolbox HEVC encoder with proper settings
    ffmpeg -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 5M \
        -maxrate 8M \
        -vf "scale=-2:1080" \
        -pix_fmt nv12 \
        -profile:v main \
        -level 4.0 \
        -c:a copy \
        -map_metadata 0 \
        -movflags +faststart \
        -tag:v hvc1 \
        "$output_file" \
        -y 2>&1 | while IFS= read -r line; do
            if [[ $line == *"frame="* ]] && [[ $line == *"fps="* ]]; then
                # Extract and display progress
                echo -ne "\r  Progress: $line"
            fi
        done
    
    echo "" # New line after progress
    
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
            echo "  Codec: HEVC (hevc_videotoolbox)"
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
    ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -n 20
    
    echo ""
    total_input=$(du -ch "$INPUT_DIR"/*.mov 2>/dev/null | tail -1 | cut -f1)
    total_output=$(du -ch "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -1 | cut -f1)
    echo "Total size: $total_input → $total_output"
fi

echo ""
echo "GPU Acceleration Notes:"
echo "- VideoToolbox uses hardware encoding when available"
echo "- The AMD RX 580 supports HEVC encoding through macOS drivers"
echo "- GPU usage may appear as 'Compute' or 'Video Processing' in Activity Monitor"
echo "- Look for 'VTEncoderXPCService' process during encoding"