#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"

# Counter for progress
total_files=$(find "$INPUT_DIR" -maxdepth 1 -name "*.mov" | wc -l | tr -d ' ')
current_file=0

echo "Starting HEVC encoding with Metal acceleration..."
echo "Total files to process: $total_files"
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
    
    # Encode with Metal acceleration (VideoToolbox) and preserve metadata
    # Using hardware decoding and encoding for maximum GPU utilization
    ffmpeg -hwaccel videotoolbox \
        -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 6M \
        -pix_fmt nv12 \
        -vf "scale=-2:1080" \
        -profile:v main \
        -c:a copy \
        -map_metadata 0 \
        -movflags use_metadata_tags \
        -movflags +faststart \
        -tag:v hvc1 \
        "$output_file" \
        -y \
        -loglevel warning \
        -stats
    
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
echo "Encoded files are in: $OUTPUT_DIR"

# Show summary
echo ""
echo "Summary:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs echo "Total files encoded:"
total_size_before=$(du -sh "$INPUT_DIR"/*.mov 2>/dev/null | tail -1 | cut -f1)
total_size_after=$(du -sh "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -1 | cut -f1)
echo "Total size before: $total_size_before"
echo "Total size after: $total_size_after"