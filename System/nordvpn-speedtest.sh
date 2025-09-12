#!/bin/bash
#
# NordVPN Speed Test
# ----------------
# Author: sanchez314c@speedheathens.com
# Date: 2025-01-24
# Version: 1.0.0
#
# Description:
#     Tests NordVPN server speeds using speedtest-cli, helping
#     identify optimal server connections. Includes result logging
#     and server comparison.
#
# Features:
#     - Server speed testing
#     - Result logging
#     - Server comparison
#     - Best server selection
#     - Error handling
#
# Requirements:
#     - bash 4.0+
#     - speedtest-cli
#     - OpenVPN
#     - Internet connection
#
# Usage:
#     ./nordvpn-speedtest.sh
## unordvpn speedtest
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
#     ./nordvpn-speedtest.sh

#     ./nord_speedtest_cli.sh
#

# Description: Runs a speed test for NordVPN servers.
#!/usr/bin/env bash

# Ensure SpeedTest CLI is installed
if ! command -v speedtest &> /dev/null; then
  echo "SpeedTest CLI is required but not installed. Install it using 'brew install speedtest-cli' or from https://www.speedtest.net/apps/cli"
  exit 1
fi

# Ensure OpenVPN is installed
if ! command -v openvpn &> /dev/null; then
  echo "OpenVPN is required but not installed. Install it using 'brew install openvpn'."
  exit 1
fi

# Directory to save temporary OpenVPN config files
config_dir=~/nordvpn-temp-configs
mkdir -p "$config_dir"

# File containing the server data (domains only)
server_list="/Users/heathen.admin/Library/Mobile Documents/com~apple~CloudDocs/Scripts/servers.txt"

# Ensure the server list file exists
if [[ ! -f "$server_list" ]]; then
  echo "Error: Server list file '$server_list' not found."
  exit 1
fi

# Embedded credentials
vpn_username="QjdXVMz6LX4VXyxGPLAeayRE"
vpn_password="5qquYQEqPFRewtVZQZQW4ZPb"

# Create a temporary credentials file
credentials_file=$(mktemp)
echo -e "$vpn_username\n$vpn_password" > "$credentials_file"
chmod 600 "$credentials_file"

# Function to test server speed using SpeedTest CLI
test_server_speed() {
  local server=$1
  local protocol="tcp"  # or "udp" if you prefer UDP

  echo "Testing speed for server: $server"

  # Construct the URL for the server's .ovpn file
  config_url="https://downloads.nordcdn.com/configs/files/ovpn_${protocol}/servers/${server}.${protocol}.ovpn"

  # Download the configuration file
  curl -s -o "$config_dir/$server.${protocol}.ovpn" "$config_url"

  if [[ ! -f "$config_dir/$server.${protocol}.ovpn" ]]; then
    echo "Configuration file for $server could not be downloaded."
    return
  fi

  # Disconnect any existing OpenVPN connection
  sudo killall openvpn &> /dev/null

  # Connect to the OpenVPN server
  sudo openvpn --config "$config_dir/$server.${protocol}.ovpn" --auth-user-pass "$credentials_file" --auth-nocache &

  # Wait a few seconds to ensure the connection is established
  sleep 20

  # Run the speed test and save the results to a variable
  result=$(speedtest --simple)  # Using --simple for concise output
  echo "$result"

  # Disconnect from the OpenVPN server
  sudo killall openvpn

  # Wait a few seconds to ensure the disconnection
  sleep 10

  # Append the result to a file
  echo -e "$server\n$result\n" >> speedtest_results.json
}

# Clear previous results
> speedtest_results.json

# Read the server list and test the speed of each server
while IFS= read -r server; do
  test_server_speed "$server"
done < "$server_list"

# Clean up the temporary credentials file
rm -f "$credentials_file"

# Sort the results and display the top 10 fastest servers
echo "Calculating the top 10 fastest servers..."
awk 'BEGIN { FS = " " } /Server:/ { server = $2; getline; getline; if ($2 ~ /Download/) { download = $3 } else { download = "0.0" }; if ($2 ~ /Upload/) { upload = $3 } else { upload = "0.0" }; printf "%s %.2f\n", server, download }' speedtest_results.json | sort -k2,2nr | head -n 10 > top10servers.txt

echo "These are the top 10 fastest servers and their download speeds (in Mbps):"
cat top10servers.txt
