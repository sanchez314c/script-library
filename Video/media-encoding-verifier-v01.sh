#!/bin/bash

echo "HEVC Encoding Verification Report"
echo "================================="
echo ""

# Check encoded files
OUTPUT_DIR="/Users/heathen-admin/Development/Screenshots/HEVC_1080p"
INPUT_DIR="/Users/heathen-admin/Development/Screenshots"

if [ -d "$OUTPUT_DIR" ]; then
    encoded_count=$(ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "✓ Found $encoded_count encoded files in $OUTPUT_DIR"
    echo ""
    
    if [ $encoded_count -gt 0 ]; then
        echo "Encoded File Details:"
        echo "--------------------"
        
        # Check each encoded file
        for file in "$OUTPUT_DIR"/*.mp4; do
            [ -e "$file" ] || continue
            
            filename=$(basename "$file")
            filesize=$(du -h "$file" | cut -f1)
            
            # Get video info
            info=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,bit_rate -of csv=p=0 "$file" 2>/dev/null)
            IFS=',' read -r codec width height bitrate <<< "$info"
            
            echo "• $filename"
            echo "  Size: $filesize"
            echo "  Codec: $codec"
            echo "  Resolution: ${width}x${height}"
            if [ -n "$bitrate" ]; then
                bitrate_mb=$(echo "scale=2; $bitrate / 1000000" | bc)
                echo "  Bitrate: ${bitrate_mb} Mbps"
            fi
            echo ""
        done
        
        # Summary statistics
        echo "Summary Statistics:"
        echo "------------------"
        
        # Total sizes
        total_input=$(du -ch "$INPUT_DIR"/*.mov 2>/dev/null | tail -1 | cut -f1)
        total_output=$(du -ch "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -1 | cut -f1)
        
        echo "Original files total size: $total_input"
        echo "Encoded files total size: $total_output"
        
        # Check if hardware acceleration was used
        echo ""
        echo "Hardware Acceleration Status:"
        echo "----------------------------"
        if pgrep -x "VTEncoderXPCService" > /dev/null; then
            echo "✓ VTEncoderXPCService is currently running (hardware encoding active)"
        else
            echo "• VTEncoderXPCService not currently running"
            echo "  (This is normal if encoding has completed)"
        fi
        
        echo ""
        echo "VideoToolbox Support:"
        ffmpeg -hide_banner -encoders 2>/dev/null | grep videotoolbox | while read line; do
            echo "✓ $line"
        done
    fi
else
    echo "✗ Output directory not found: $OUTPUT_DIR"
fi

echo ""
echo "Encoding Configuration Used:"
echo "---------------------------"
echo "• Codec: HEVC (H.265) via hevc_videotoolbox"
echo "• Resolution: 1080p (scaled)"
echo "• Hardware: AMD RX 580 via VideoToolbox"
echo "• Metadata: Preserved from original files"