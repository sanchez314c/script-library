#!/bin/bash
#
# Tag Files Red
# -----------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Applies red color tags to files in selected directories using
#     macOS metadata. Supports bulk tagging operations with GUI
#     folder selection.
#
# Features:
#     - GUI folder selection
#     - Bulk tag application
#     - Progress tracking
#     - Error handling
#     - macOS integration
#
# Requirements:
#     - bash 4.0+
#     - macOS
#     - xattr command
#     - osascript
#
# Usage:
#     ./tag-files-red.sh
## utag files red
# --------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     System maintenance and automation script for efficient
#     system management and file operations.
#
# Features:
#     - Automated processing
#     - Error handling
#     - Progress tracking
#     - System integration
#     - Status reporting
#
# Requirements:
#     - bash 4.0+
#     - Standard Unix tools
#
# Usage:
#     ./tag-files-red.sh

#     ./TAG-ALL-RED.sh

#!/bin/bash

# Use AppleScript to prompt for a folder selection
DIRECTORY=$(osascript <<'END'
tell application "Finder"
    set folderPath to choose folder with prompt "Select the folder:"
    POSIX path of folderPath
end tell
END
)

# Trim any whitespace and remove potential trailing slashes
DIRECTORY=$(echo "$DIRECTORY" | awk '{gsub(/^ +| +$/,"")} {print}' | sed 's:/*$::')

# Check if a directory was selected
if [ -z "$DIRECTORY" ]; then
    echo "No folder selected. Exiting script."
    exit 1
fi

# Function to add RED tag
add_red_tag() {
    local file="$1"
    xattr -wx com.apple.metadata:_kMDItemUserTags "$(printf "62 70 6C 69 73 74 30 30 A1 01 5F 10 03 52 45 44 08 0B 00 00 00 00 00 00 01 01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 19" | xxd -r -p)" "$file"
}

# Export the function so it can be used by find -exec
export -f add_red_tag

# Use find to iterate over each file in the selected directory and add the RED tag
find "$DIRECTORY" -type f -exec bash -c 'add_red_tag "$0"' {} \;

echo "All files in $DIRECTORY have been tagged with RED."
