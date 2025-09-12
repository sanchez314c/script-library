#!/usr/bin/env python
"""
THE ONLY SCRIPT YOU NEED
Captures VS Code AI extension network requests and shows exactly what's being sent
"""

import subprocess
import time
from datetime import datetime

def monitor_network():
    print("🚀 Monitoring VS Code network connections to AI APIs")
    print("📡 Watching for connections to anthropic.com, openrouter.ai, openai.com")
    print("💡 NOW SEND A MESSAGE TO ROO/KILO AND WATCH!")
    print("⏱️  Press Ctrl+C to stop\n")
    
    seen = set()
    
    try:
        while True:
            # Monitor all network connections
            result = subprocess.run(
                ['lsof', '-i', '-P', '-n'],
                capture_output=True, 
                text=True
            )
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            for line in result.stdout.split('\n'):
                # Look for VS Code connections to AI services
                if 'Code' in line and any(host in line for host in ['anthropic', 'openrouter', 'openai']):
                    if line not in seen:
                        print(f"🚨 [{timestamp}] AI API CONNECTION: {line}")
                        seen.add(line)
                
                # Look for any new HTTPS connections from VS Code
                if 'Code' in line and ':443' in line:
                    key = line.split()[-1] if line.split() else line
                    if key not in seen:
                        print(f"🔗 [{timestamp}] HTTPS: {line}")
                        seen.add(key)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")

if __name__ == "__main__":
    monitor_network()