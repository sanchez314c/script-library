#!/bin/bash

echo "🔍 Starting network monitoring for VS Code API calls..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Monitor network connections from VS Code processes
echo "Monitoring network activity from VS Code processes..."

# Use netstat to monitor network connections
while true; do
    # Look for connections to AI API endpoints
    netstat -an | grep -E "(anthropic|openrouter|openai)" | while read line; do
        echo "$(date '+%H:%M:%S') - $line"
    done
    
    # Also monitor any HTTPS connections from VS Code
    lsof -i :443 -P | grep -i code | while read line; do
        echo "$(date '+%H:%M:%S') - HTTPS: $line"
    done
    
    sleep 1
done