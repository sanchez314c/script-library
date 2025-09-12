#!/bin/bash

# Fix rocBLAS dependency issues without replacing custom-compiled libraries
# This script creates minimal dummy packages manually

# Exit on any error
set -e

echo "=== Creating dummy packages for rocBLAS dependencies ==="
echo "This will fix APT dependency issues while keeping your custom libraries."

# Create working directories
echo -e "\n[1/5] Creating working directories..."
WORK_DIR=$(mktemp -d)
mkdir -p "$WORK_DIR/rocblas/DEBIAN"
mkdir -p "$WORK_DIR/rocblas-dev/DEBIAN"

# Create control file for rocblas
echo -e "\n[2/5] Creating control file for rocblas..."
cat > "$WORK_DIR/rocblas/DEBIAN/control" << EOF
Package: rocblas
Version: 4.3.0.60303-74~24.04
Section: libs
Priority: optional
Architecture: amd64
Maintainer: System Administrator <root@localhost>
Description: Dummy package for rocblas
 This is a dummy package that provides rocblas version information
 without replacing the custom-compiled files.
EOF

# Create control file for rocblas-dev
echo -e "\n[3/5] Creating control file for rocblas-dev..."
cat > "$WORK_DIR/rocblas-dev/DEBIAN/control" << EOF
Package: rocblas-dev
Version: 4.3.0.60303-74~24.04
Section: libdevel
Priority: optional
Architecture: amd64
Maintainer: System Administrator <root@localhost>
Description: Dummy package for rocblas-dev
 This is a dummy package that provides rocblas-dev version information
 without replacing the custom-compiled files.
EOF

# Build and install dummy packages
echo -e "\n[4/5] Building and installing dummy packages..."
dpkg-deb --build "$WORK_DIR/rocblas"
dpkg-deb --build "$WORK_DIR/rocblas-dev"

# Force installation of the dummy packages
sudo dpkg --force-all -i "$WORK_DIR/rocblas.deb"
sudo dpkg --force-all -i "$WORK_DIR/rocblas-dev.deb"

# Update system
echo -e "\n[5/5] Updating package system..."
sudo apt-get update

# Clean up
echo -e "\nCleaning up temporary files..."
rm -rf "$WORK_DIR"

echo -e "\n=== All done! ==="
echo "Your system should now be able to install and update packages"
echo "while keeping your custom-optimized rocBLAS libraries."
echo ""
echo "To verify, try running: sudo apt update && sudo apt upgrade"
echo "You should also now be able to install Go: sudo apt install golang-go"