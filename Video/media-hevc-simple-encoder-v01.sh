#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"

# Clean up empty files
rm -f "$OUTPUT_DIR"/*.mp4

# Counter for progress
total_files=$(ls -1 "$INPUT_DIR"/*.mov 2>/dev/null | wc -l)
current_file=0

echo "Starting HEVC encoding with VideoToolbox hardware acceleration..."
echo "Total files to process: $total_files"
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
    
    # Encode with VideoToolbox hardware acceleration
    # Using simpler settings that are known to work
    ffmpeg -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 6000k \
        -vf "scale=-2:1080" \
        -c:a copy \
        -map_metadata 0 \
        -tag:v hvc1 \
        "$output_file" \
        -y
    
    if [ $? -eq 0 ]; then
        # Get file sizes for comparison
        input_size=$(du -h "$input_file" | cut -f1)
        output_size=$(du -h "$output_file" | cut -f1)
        echo "✓ Success: $filename ($input_size → $output_size)"
    else
        echo "✗ Error processing: $filename"
    fi
    
    echo "----------------------------------------"
done

echo "Encoding complete!"
echo ""

# Show summary
echo "Summary:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs echo "Total files encoded:"
echo ""
echo "File sizes:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | awk '{print $9 ": " $5}'
echo ""
echo "Note: VideoToolbox (Metal) hardware acceleration was used for encoding."
echo "The hevc_videotoolbox encoder uses the GPU's Media Engine on Apple Silicon."