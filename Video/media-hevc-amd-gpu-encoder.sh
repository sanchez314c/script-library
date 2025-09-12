#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"

# Clean up empty files from previous attempts
rm -f "$OUTPUT_DIR"/*.mp4

# Counter for progress
total_files=$(ls -1 "$INPUT_DIR"/*.mov 2>/dev/null | wc -l)
current_file=0

echo "Starting HEVC encoding for Intel Mac with AMD RX 580..."
echo "Total files to process: $total_files"
echo ""
echo "NOTE: On Intel Macs, VideoToolbox uses Intel Quick Sync Video (if available)"
echo "and AMD VCE/VCN through VideoToolbox framework."
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
    output_file="$OUTPUT_DIR/${filename%.mov}_HEVC.mp4"
    
    echo "[$current_file/$total_files] Processing: $filename"
    
    # Get input file info
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file" 2>/dev/null)
    
    # Encode with VideoToolbox hardware acceleration optimized for Intel/AMD
    # -hwaccel auto: Let FFmpeg choose the best hardware acceleration
    # -c:v hevc_videotoolbox: Use VideoToolbox HEVC encoder (works with AMD on macOS)
    ffmpeg -hwaccel auto \
        -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 6000k \
        -maxrate 8000k \
        -bufsize 12000k \
        -vf "scale=-2:1080:flags=lanczos" \
        -pix_fmt yuv420p \
        -color_primaries bt709 \
        -color_trc bt709 \
        -colorspace bt709 \
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
echo "Hardware acceleration notes for Intel Mac with AMD RX 580:"
echo "- VideoToolbox framework provides hardware encoding support"
echo "- The RX 580 supports H.265/HEVC encoding through macOS drivers"
echo "- Performance may vary based on macOS version and driver support"