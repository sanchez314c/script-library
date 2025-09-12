#!/bin/bash
#
# System Inspection Tool
# -------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Comprehensive system inspection and diagnostics tool that
#     analyzes system components, configuration, and performance
#     metrics.
#
# Features:
#     - System health check
#     - Configuration analysis
#     - Performance metrics
#     - Security scanning
#     - Detailed reporting
#
# Requirements:
#     - bash 4.0+
#     - System tools
#     - Root access
#
# Usage:
#     ./system-inspection.sh
## usystem inspection
# ------------------
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
#     ./system-inspection.sh

#     ./comprehensive_inspection.sh
#

# Description: Performs a comprehensive inspection of the system.
#!/bin/bash

LOGFILE="comprehensive_system_inspection.log"

# Start logging
exec > >(tee -a $LOGFILE)
exec 2>&1

echo "===== Comprehensive System Inspection Report ====="
echo "Generated on: $(date)"
echo ""

# System information
echo "===== System Information ====="
echo ""
echo "Hostname: $(hostname)"
echo "Uptime: $(uptime -p)"
echo "Kernel Version: $(uname -r)"
echo "OS Version: $(lsb_release -a 2>/dev/null)"
echo "Architecture: $(uname -m)"
echo ""

# CPU Information
echo "===== CPU Information ====="
echo ""
lscpu
echo ""

# Memory Information
echo "===== Memory Information ====="
echo ""
free -h
echo ""
echo "===== Memory Details ====="
echo ""
cat /proc/meminfo
echo ""

# Disk Usage
echo "===== Disk Usage ====="
echo ""
df -h
echo ""
echo "===== Detailed Disk Information ====="
echo ""
lsblk
echo ""

# Mounted Filesystems
echo "===== Mounted Filesystems ====="
echo ""
mount | column -t
echo ""

# Disk Health
echo "===== Disk Health ====="
echo ""
for disk in $(lsblk -nd --output NAME); do
    echo "Checking health of /dev/$disk"
    sudo smartctl -H /dev/$disk
done
echo ""

# Network Configuration
echo "===== Network Configuration ====="
echo ""
ip a
echo ""
echo "===== Network Routes ====="
echo ""
ip route
echo ""
echo "===== DNS Configuration ====="
echo ""
cat /etc/resolv.conf
echo ""

# Active Network Connections
echo "===== Active Network Connections ====="
echo ""
ss -tuln
echo ""

# Installed Packages
echo "===== Installed Packages ====="
echo ""
dpkg -l
echo ""

# Running Processes
echo "===== Running Processes ====="
echo ""
ps aux --sort=-%mem | head -n 20
echo ""

# System Logs
echo "===== System Logs (last 100 lines) ====="
echo ""
journalctl -xe -n 100
echo ""

# Service Status
echo "===== Service Status ====="
echo ""
systemctl list-units --type=service --state=running
echo ""
echo "===== Failed Services ====="
echo ""
systemctl --failed
echo ""

# Checking for Common Misconfigurations and Errors
echo "===== Common Misconfigurations and Errors ====="
echo ""

# Checking for broken packages
echo "Checking for broken packages..."
sudo dpkg --configure -a
sudo apt-get install -f
sudo apt-get check
echo ""

# Checking filesystem for errors
echo "Checking filesystem for errors..."
sudo touch /forcefsck
sudo fsck -N
echo ""

# Checking dmesg for hardware errors
echo "Checking dmesg for hardware errors..."
dmesg -T | grep -Ei 'error|fail|critical'
echo ""

# Checking syslog for recent errors
echo "Checking syslog for recent errors..."
grep -i error /var/log/syslog | tail -n 100
echo ""

# Security and Permissions Check
echo "===== Security and Permissions Check ====="
echo ""
echo "Checking for world-writable files..."
find / -xdev -type f -perm -0002 -ls
echo ""
echo "Checking for world-writable directories..."
find / -xdev -type d -perm -0002 -ls
echo ""
echo "Checking for SUID/SGID files..."
find / -xdev \( -perm -4000 -o -perm -2000 \) -print
echo ""

# Configuration File Checks
echo "===== Configuration File Checks ====="
echo ""
echo "Checking for syntax errors in /etc..."
sudo find /etc -name "*.conf" -exec sudo bash -c 'echo "{}"; sudo bash -n "{}"' \;
echo ""
echo "Checking for syntax errors in crontabs..."
sudo find /var/spool/cron /etc/cron.d /etc/cron.daily /etc/cron.hourly /etc/cron.monthly /etc/cron.weekly -type f -exec sudo bash -c 'echo "{}"; sudo bash -n "{}"' \;
echo ""

# Package Integrity
echo "===== Package Integrity ====="
echo ""
echo "Checking for package integrity issues..."
debsums -c
echo ""

# Kernel Messages
echo "===== Kernel Messages ====="
echo ""
dmesg | grep -Ei 'error|warn|critical'
echo ""

# File System Errors
echo "===== File System Errors ====="
echo ""
sudo fsck -f -n
echo ""

echo "===== End of Comprehensive System Inspection Report ====="

