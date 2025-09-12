#!/usr/bin/env python
"""
Identify VS Code connections and resolve IP addresses to see who they're talking to
"""

import subprocess
import socket
import time
from datetime import datetime

def resolve_ip(ip):
    """Resolve IP to hostname"""
    try:
        hostname = socket.gethostbyaddr(ip)[0]
        return hostname
    except:
        return ip

def get_whois_info(ip):
    """Get basic info about an IP"""
    try:
        result = subprocess.run(['whois', ip], capture_output=True, text=True, timeout=5)
        lines = result.stdout.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['orgname', 'organization', 'netname']):
                return line.strip()
    except:
        pass
    return "Unknown"

def monitor_and_identify():
    print("🔍 Monitoring VS Code connections and identifying endpoints")
    print("💡 Send a message to Roo/Kilo now!")
    print("⏱️  Press Ctrl+C to stop\n")
    
    seen_ips = set()
    
    try:
        while True:
            result = subprocess.run(['lsof', '-i', '-P', '-n'], capture_output=True, text=True)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            for line in result.stdout.split('\n'):
                if 'Code' in line and ':443' in line and 'ESTABLISHED' in line:
                    # Extract the remote IP
                    parts = line.split()
                    for part in parts:
                        if '->' in part and ':443' in part:
                            remote_addr = part.split('->')[1].split(':')[0]
                            
                            if remote_addr not in seen_ips:
                                seen_ips.add(remote_addr)
                                hostname = resolve_ip(remote_addr)
                                org_info = get_whois_info(remote_addr)
                                
                                print(f"🚨 [{timestamp}] NEW CONNECTION")
                                print(f"   IP: {remote_addr}")
                                print(f"   Hostname: {hostname}")
                                print(f"   Organization: {org_info}")
                                
                                # Check if this looks like an AI service
                                if any(ai_term in hostname.lower() for ai_term in ['anthropic', 'openai', 'claude', 'openrouter']):
                                    print(f"   🎯 AI SERVICE DETECTED!")
                                elif any(ai_term in org_info.lower() for ai_term in ['anthropic', 'openai', 'claude', 'openrouter']):
                                    print(f"   🎯 AI SERVICE DETECTED!")
                                
                                print()
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("✅ Monitoring stopped")

if __name__ == "__main__":
    monitor_and_identify()