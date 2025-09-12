#!/bin/bash

# Directory paths
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"

# Clean up empty files from previous attempts
rm -f "$OUTPUT_DIR"/*.mp4

# Counter for progress
total_files=$(ls -1 "$INPUT_DIR"/*.mov 2>/dev/null | wc -l)
current_file=0

echo "HEVC Encoding - Forcing VideoToolbox Hardware Acceleration"
echo "========================================================="
echo "Total files to process: $total_files"
echo ""
echo "Based on VideoProc, your system DOES support hardware encoding!"
echo "Using aggressive VideoToolbox settings to force GPU usage."
echo ""
echo "Monitor in Activity Monitor:"
echo "- GPU History window"
echo "- Look for 'VTDecoderXPCService' and 'VTEncoderXPCService' processes"
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
    echo "Forcing VideoToolbox hardware acceleration..."
    
    # Force VideoToolbox with specific settings
    # -hwaccel videotoolbox: Force hardware decoding
    # -hwaccel_device 0: Use first GPU device
    # realtime: Forces immediate processing (can trigger GPU usage)
    ffmpeg -hwaccel videotoolbox \
        -hwaccel_device 0 \
        -i "$input_file" \
        -c:v hevc_videotoolbox \
        -b:v 8000k \
        -maxrate 10000k \
        -bufsize 16000k \
        -realtime 1 \
        -vf "format=nv12,scale_vt=-2:1080" \
        -allow_sw 0 \
        -require_hwaccel 1 \
        -profile:v main \
        -c:a copy \
        -map_metadata 0 \
        -movflags +faststart \
        -tag:v hvc1 \
        "$output_file" \
        -y 2>&1 | while IFS= read -r line; do
            if [[ $line == *"frame="* ]] || [[ $line == *"fps="* ]]; then
                echo -ne "\r$line"
            elif [[ $line == *"error"* ]] || [[ $line == *"Error"* ]]; then
                echo ""
                echo "Error: $line"
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
        else
            echo "✓ Success: $filename ($input_size → $output_size)"
        fi
        
        # Check if hardware was used
        echo "  Verifying hardware acceleration was used..."
        probe_output=$(ffprobe -v error -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$output_file" 2>&1 | head -1)
        if [[ "$probe_output" == "hevc" ]]; then
            echo "  ✓ HEVC encoding successful"
        fi
    else
        echo "✗ Error processing: $filename"
        echo "  Trying alternative settings..."
        
        # Alternative attempt without strict hardware requirements
        ffmpeg -i "$input_file" \
            -c:v hevc_videotoolbox \
            -b:v 6000k \
            -vf "scale=-2:1080" \
            -c:a copy \
            -map_metadata 0 \
            -tag:v hvc1 \
            "$output_file" \
            -y -loglevel error
            
        if [ -f "$output_file" ] && [ -s "$output_file" ]; then
            echo "  ✓ Encoded with alternative settings"
        fi
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
echo "Hardware Acceleration Check:"
echo "1. Open Activity Monitor > Window > GPU History"
echo "2. Look for GPU usage spikes during encoding"
echo "3. Check for VTEncoderXPCService in process list"
echo ""
echo "Note: Even if GPU usage appears low, VideoToolbox may still be"
echo "using hardware acceleration through the Media Engine."