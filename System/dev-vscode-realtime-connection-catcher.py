#!/usr/bin/env python
"""
CATCH THEM RED-HANDED
Simple script to show VS Code making connections when you send messages
"""

import subprocess
import time
from datetime import datetime

print("🚨 WAITING FOR YOU TO SEND A MESSAGE TO ROO...")
print("🔍 This will show EXACTLY when VS Code connects to AI services")
print("💡 Send your message NOW!")
print()

seen = set()

try:
    while True:
        # Check for new connections
        result = subprocess.run(['lsof', '-i', '-P', '-n'], capture_output=True, text=True)
        
        for line in result.stdout.split('\n'):
            if 'Code' in line and any(ip in line for ip in ['37.16.29.120', '173.223.239.80', '104.18.3.115']):
                if line not in seen:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"🚨 [{timestamp}] CAUGHT: {line}")
                    seen.add(line)
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n✅ Done")