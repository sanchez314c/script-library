#!/bin/bash

echo "Testing GPU hardware acceleration with VideoToolbox..."
echo "Monitor GPU usage in Activity Monitor > Window > GPU History"
echo ""

# Test file
TEST_FILE="/Users/heathen-admin/Development/Screenshots/Screen Recording 2025-06-24 at 3.03.06 AM.mov"
OUTPUT_FILE="/Users/heathen-admin/Development/Screenshots/HEVC_1080p/test_gpu.mp4"

# Show detailed encoding info
echo "Starting encode with hardware acceleration..."
ffmpeg -hwaccel videotoolbox \
    -hwaccel_output_format videotoolbox_vld \
    -i "$TEST_FILE" \
    -c:v hevc_videotoolbox \
    -b:v 8M \
    -maxrate 10M \
    -pix_fmt nv12 \
    -vf "scale=-2:1080" \
    -profile:v main \
    -level:v 4.0 \
    -c:a copy \
    -map_metadata 0 \
    -movflags use_metadata_tags \
    -movflags +faststart \
    -tag:v hvc1 \
    "$OUTPUT_FILE" \
    -y

echo ""
echo "Encoding complete. Check GPU usage in Activity Monitor."