# Development Tools

Utilities and tools for development, monitoring, and security tasks.

## Directory Structure

### Network-Monitoring/
Tools for monitoring network connections, API requests, and system activity
- `network-ai-request-monitor-v00.py` - Monitor VS Code AI extension network requests
- Detect connections to anthropic.com, openrouter.ai, openai.com
- Real-time HTTPS monitoring with timestamped logs

### Security-Tools/
Security utilities for key management and data protection
- `security-api-key-rotator-v00.py` - Comprehensive API key rotation tool
- Supports 15+ AI/ML services (OpenAI, Anthropic, Google, etc.)
- Automated config updates and environment variable management
- NordPass integration and logging

### Python-Utilities/
General-purpose Python development utilities

## Available Tools

### Network Monitoring
**AI Request Monitor (`network-ai-request-monitor-v00.py`)**
- Real-time monitoring of VS Code AI extension requests
- Identifies API calls to major AI services
- Timestamped logging with connection details
- HTTPS traffic analysis
- Usage: `python network-ai-request-monitor-v00.py`

### Security Tools  
**API Key Rotator (`security-api-key-rotator-v00.py`)**
- Multi-service API key rotation automation
- Supported services: OpenAI, Anthropic, Google, Mistral, OpenRouter, etc.
- Browser automation for web-based key rotation
- Configuration file updates (JSON format)
- Environment variable management (.bashrc/.zshrc)
- Comprehensive logging and reporting
- Usage: `python security-api-key-rotator-v00.py [--services openai anthropic] [--all]`

## Security Features
- All tools audited for security compliance
- No malicious code detected
- Enhanced error handling and validation
- Secure credential handling practices
- Comprehensive logging for audit trails

## Usage Notes
- Network monitoring requires appropriate system permissions
- Key rotation tool requires browser access for web-based services
- Python 3.6+ required for all utilities
- See individual tool documentation for specific dependencies