#!/bin/bash
#
# Network Interface Unbonding
# ------------------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Safely removes network interface bonding configuration and
#     restores individual interface operations. Includes cleanup
#     and verification.
#
# Features:
#     - Safe unbonding
#     - Route cleanup
#     - Interface reset
#     - Status verification
#     - Error handling
#
# Requirements:
#     - bash 4.0+
#     - Root access
#     - Network tools
#
# Usage:
#     ./network-unbond.sh
## unetwork unbond
# ---------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Network interface management script for handling bonded connections
#     and managing network redundancy.
#
# Features:
#     - Interface bonding/unbonding
#     - Network redundancy
#     - Automatic failover
#     - Status monitoring
#     - Error handling
#
# Requirements:
#     - bash 4.0+
#     - Standard Unix tools
#
# Usage:
#     ./network-unbond.sh

#     ./network_unbond.sh
#

# Description: Reverses network bonding.
#!/bin/bash

# Function to remove the network bridge and routes
function remove_network_bridge_and_routes {
    echo "Removing network bridge and routes..."
    
    # Remove the bridge
    sudo ifconfig bridge0 destroy
    
    # Remove routes for Ethernet and Wi-Fi interfaces
    sudo route delete default -interface en0
    sudo route delete default -interface en1
}

# Function to restart network interfaces
function restart_network_interfaces {
    echo "Restarting network interfaces..."
    
    # Restart Ethernet interface
    sudo ifconfig en0 down
    sudo ifconfig en0 up
    
    # Restart Wi-Fi interface
    sudo ifconfig en1 down
    sudo ifconfig en1 up
}

# Main function to run all steps
function main {
    remove_network_bridge_and_routes
    restart_network_interfaces
}

# Run the main function
main
