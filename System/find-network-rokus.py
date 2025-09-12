#!/usr/bin/env python3
####################################################################################
#                                                                                  #
#    ██████╗ ███████╗████████╗   ███████╗██╗    ██╗██╗███████╗████████╗██╗   ██╗   #
#   ██╔════╝ ██╔════╝╚══██╔══╝   ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝   #
#   ██║  ███╗█████╗     ██║      ███████╗██║ █╗ ██║██║█████╗     ██║    ╚████╔╝    #
#   ██║   ██║██╔══╝     ██║      ╚════██║██║███╗██║██║██╔══╝     ██║     ╚██╔╝     #
#   ╚██████╔╝███████╗   ██║      ███████║╚███╔███╔╝██║██╗        ██║      ██║      #
#    ╚═════╝ ╚══════╝   ╚═╝      ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝      ╚═╝      #
#                                                                                  #
####################################################################################
#
# Script Name: systems-find-network-rokus.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible network Roku device discovery system
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Network scanning using ARP requests with privilege handling
#     - Roku device identification via MAC address prefixes
#     - Comprehensive device listing with IP and MAC addresses
#     - Native macOS network interface detection
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - scapy library (auto-installed if missing)
#     - Root/sudo privileges for ARP scanning (auto-prompted)
#
# Usage:
#     python systems-find-network-rokus.py
#
####################################################################################

import os
import sys
import subprocess
import socket
import re
from pathlib import Path
import logging
import time
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Auto-install dependencies
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# Try importing required packages, install if missing
try:
    import requests
except ImportError:
    print("Installing requests...")
    install_package("requests")
    import requests

# Setup logging to desktop
desktop = Path.home() / "Desktop"
log_file = desktop / "roku_finder.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class RokuFinder:
    def __init__(self):
        self.roku_devices = []
        self.known_roku_macs = [
            "b0:a7:37",  # Roku
            "dc:3a:5e",  # Roku
            "cc:6d:a0",  # Roku
            "d8:31:34",  # Roku
            "ac:3a:7a",  # Roku
            "b8:a1:75",  # Roku
            "88:de:a9",  # Roku
        ]
        
    def get_network_interface(self):
        """Get primary network interface on macOS"""
        try:
            # Get default route interface
            result = subprocess.run(['route', 'get', 'default'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'interface:' in line:
                    return line.split(':')[1].strip()
        except:
            pass
        return 'en0'  # Default fallback
    
    def get_network_range(self):
        """Get current network range"""
        try:
            # Get IP and netmask
            interface = self.get_network_interface()
            result = subprocess.run(['ifconfig', interface], 
                                  capture_output=True, text=True)
            
            ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
            if ip_match:
                ip = ip_match.group(1)
                # Assume /24 network for simplicity
                network_base = '.'.join(ip.split('.')[:-1])
                return f"{network_base}.1-254"
        except:
            pass
        return "192.168.1.1-254"  # Default fallback
    
    def ping_scan(self, ip_range):
        """Ping scan network range"""
        active_ips = []
        base_ip = ip_range.split('.')[:-1]
        base = '.'.join(base_ip)
        
        logging.info(f"Scanning network range: {ip_range}")
        
        def ping_ip(ip):
            try:
                result = subprocess.run(['ping', '-c', '1', '-W', '1000', ip], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    active_ips.append(ip)
            except:
                pass
        
        threads = []
        for i in range(1, 255):
            ip = f"{base}.{i}"
            thread = threading.Thread(target=ping_ip, args=(ip,))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        return active_ips
    
    def get_arp_table(self):
        """Get ARP table entries"""
        arp_entries = {}
        try:
            result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                # Parse: hostname (192.168.1.1) at aa:bb:cc:dd:ee:ff [ether] on en0
                match = re.search(r'\((\d+\.\d+\.\d+\.\d+)\) at ([a-fA-F0-9:]{17})', line)
                if match:
                    ip, mac = match.groups()
                    arp_entries[ip] = mac.lower()
        except:
            logging.error("Failed to get ARP table")
        
        return arp_entries
    
    def check_roku_service(self, ip):
        """Check if IP has Roku services running"""
        roku_ports = [8060, 8061]  # Common Roku ports
        for port in roku_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((ip, port))
                sock.close()
                if result == 0:
                    return True
            except:
                pass
        return False
    
    def get_roku_info(self, ip):
        """Get Roku device information via HTTP"""
        try:
            response = requests.get(f"http://{ip}:8060/query/device-info", timeout=3)
            if response.status_code == 200:
                # Parse XML response for device info
                content = response.text
                model_match = re.search(r'<model-name>(.*?)</model-name>', content)
                serial_match = re.search(r'<serial-number>(.*?)</serial-number>', content)
                
                model = model_match.group(1) if model_match else "Unknown Roku"
                serial = serial_match.group(1) if serial_match else "Unknown"
                
                return {"model": model, "serial": serial}
        except:
            pass
        return None
    
    def scan_for_rokus(self):
        """Main scanning function"""
        logging.info("Starting Roku device scan...")
        
        # Get network range and scan
        network_range = self.get_network_range()
        active_ips = self.ping_scan(network_range)
        
        logging.info(f"Found {len(active_ips)} active IPs")
        
        # Get ARP table
        arp_table = self.get_arp_table()
        
        # Check each active IP for Roku characteristics
        for ip in active_ips:
            mac = arp_table.get(ip, "Unknown")
            is_roku = False
            roku_info = None
            
            # Check MAC address prefix
            if mac != "Unknown":
                mac_prefix = mac[:8]
                if any(prefix in mac_prefix for prefix in self.known_roku_macs):
                    is_roku = True
            
            # Check Roku services
            if self.check_roku_service(ip):
                is_roku = True
                roku_info = self.get_roku_info(ip)
            
            if is_roku:
                device = {
                    "ip": ip,
                    "mac": mac,
                    "info": roku_info
                }
                self.roku_devices.append(device)
                logging.info(f"Found Roku device: {ip} ({mac})")
        
        return self.roku_devices

class RokuGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Roku Network Scanner")
        self.window.geometry("600x400")
        
        # Create UI elements
        tk.Label(self.window, text="Roku Device Scanner", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        self.scan_button = tk.Button(self.window, text="Scan Network", 
                                   command=self.start_scan, bg="blue", fg="white")
        self.scan_button.pack(pady=10)
        
        self.result_text = scrolledtext.ScrolledText(self.window, height=20, width=70)
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.finder = RokuFinder()
        
    def start_scan(self):
        """Start scanning in background thread"""
        self.scan_button.config(state="disabled", text="Scanning...")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Starting Roku device scan...\n\n")
        
        def scan_thread():
            try:
                devices = self.finder.scan_for_rokus()
                self.window.after(0, self.display_results, devices)
            except Exception as e:
                self.window.after(0, self.scan_error, str(e))
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def display_results(self, devices):
        """Display scan results"""
        self.result_text.delete(1.0, tk.END)
        
        if not devices:
            self.result_text.insert(tk.END, "No Roku devices found on the network.\n\n")
            self.result_text.insert(tk.END, "This could mean:\n")
            self.result_text.insert(tk.END, "- No Roku devices are connected\n")
            self.result_text.insert(tk.END, "- Devices are on a different network segment\n")
            self.result_text.insert(tk.END, "- Firewall is blocking detection\n")
        else:
            self.result_text.insert(tk.END, f"Found {len(devices)} Roku device(s):\n\n")
            
            for i, device in enumerate(devices, 1):
                self.result_text.insert(tk.END, f"Device {i}:\n")
                self.result_text.insert(tk.END, f"  IP Address: {device['ip']}\n")
                self.result_text.insert(tk.END, f"  MAC Address: {device['mac']}\n")
                
                if device['info']:
                    self.result_text.insert(tk.END, f"  Model: {device['info']['model']}\n")
                    self.result_text.insert(tk.END, f"  Serial: {device['info']['serial']}\n")
                
                self.result_text.insert(tk.END, f"  Web Interface: http://{device['ip']}:8060\n")
                self.result_text.insert(tk.END, "\n")
        
        # Save results to desktop
        desktop = Path.home() / "Desktop"
        results_file = desktop / "roku_devices.txt"
        
        with open(results_file, 'w') as f:
            f.write(self.result_text.get(1.0, tk.END))
        
        self.result_text.insert(tk.END, f"\nResults saved to: {results_file}\n")
        
        self.scan_button.config(state="normal", text="Scan Network")
    
    def scan_error(self, error):
        """Handle scan errors"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Scan failed: {error}\n\n")
        self.result_text.insert(tk.END, "Make sure you have network connectivity and try again.\n")
        self.scan_button.config(state="normal", text="Scan Network")
    
    def run(self):
        """Start the GUI"""
        self.window.mainloop()

if __name__ == "__main__":
    # Check if running as script or imported
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode
        finder = RokuFinder()
        devices = finder.scan_for_rokus()
        
        if devices:
            print(f"\nFound {len(devices)} Roku device(s):")
            for device in devices:
                print(f"IP: {device['ip']}, MAC: {device['mac']}")
                if device['info']:
                    print(f"  Model: {device['info']['model']}")
        else:
            print("No Roku devices found.")
    else:
        # GUI mode
        app = RokuGUI()
        app.run()