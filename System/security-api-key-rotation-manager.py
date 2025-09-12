#!/usr/bin/env python3
"""
API Key Rotation Tool for LLM-Chat

This script helps automate the rotation of API keys for various AI and cloud services.
It provides a mix of automated rotation (where APIs support it) and guided manual rotation
with browser integration for services that require web interface interaction.
"""

import subprocess
import json
import os
import webbrowser
import datetime
import shutil
import time
from pathlib import Path
import argparse
import sys
import platform
import getpass

# Configuration for services based on the screenshots
SERVICES = {
    "google_search": {
        "name": "Google Search API",
        "rotation_method": "web",
        "web_url": "https://console.cloud.google.com/apis/credentials",
        "config_key": "google_api_key",
        "env_var": "GOOGLE_GENERATIVE_AI_API_KEY"
    },
    "google_cloud_tts": {
        "name": "Google Cloud TTS",
        "rotation_method": "web",
        "web_url": "https://console.cloud.google.com/apis/credentials",
        "config_key": "google_api_key",
        "env_var": "GOOGLE_GENERATIVE_AI_API_KEY"
    },
    "google_gemini": {
        "name": "Google Gemini",
        "rotation_method": "web",
        "web_url": "https://console.cloud.google.com/apis/credentials",
        "config_key": "google_generative_ai_api_key",
        "env_var": "GOOGLE_GENERATIVE_AI_API_KEY"
    },
    "openai": {
        "name": "OpenAI",
        "rotation_method": "web",
        "web_url": "https://platform.openai.com/api-keys",
        "config_key": "openai_api_key",
        "env_var": "OPENAI_API_KEY"
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "rotation_method": "web",
        "web_url": "https://elevenlabs.io/app/account/api",
        "config_key": "elevenlabs_api_key",
        "env_var": "ELEVENLABS_API_KEY"
    },
    "replicate": {
        "name": "Replicate",
        "rotation_method": "web",
        "web_url": "https://replicate.com/account/api-tokens",
        "config_key": "replicate_api_key",
        "env_var": "REPLICATE_API_KEY"
    },
    "pi_ai": {
        "name": "Pi.ai",
        "rotation_method": "web",
        "web_url": "https://api.pi.ai/docs",
        "config_key": "pi_api_key",
        "env_var": "PI_API_KEY"
    },
    "mistral": {
        "name": "Mistral",
        "rotation_method": "web",
        "web_url": "https://console.mistral.ai/api-keys/",
        "config_key": "mistral_api_key",
        "env_var": "MISTRAL_API_KEY"
    },
    "openrouter": {
        "name": "OpenRouter",
        "rotation_method": "web",
        "web_url": "https://openrouter.ai/keys",
        "config_key": "openrouter_api_key",
        "env_var": "OPENROUTER_API_KEY"
    },
    "anthropic": {
        "name": "Anthropic",
        "rotation_method": "web",
        "web_url": "https://console.anthropic.com/keys",
        "config_key": "anthropic_api_key",
        "env_var": "ANTHROPIC_API_KEY"
    },
    "deepseek": {
        "name": "DeepSeek",
        "rotation_method": "web",
        "web_url": "https://platform.deepseek.com/settings/api-keys",
        "config_key": "deepseek_api_key",
        "env_var": "DEEPSEEK_API_KEY"
    },
    "huggingface": {
        "name": "HuggingFace",
        "rotation_method": "web",
        "web_url": "https://huggingface.co/settings/tokens",
        "config_key": "huggingface_api_key",
        "env_var": "HF_TOKEN"
    },
    "perplexity": {
        "name": "Perplexity",
        "rotation_method": "web",
        "web_url": "https://www.perplexity.ai/settings",
        "config_key": "perplexity_api_key",
        "env_var": "PERPLEXITY_API_KEY"
    },
    "together": {
        "name": "together.ai",
        "rotation_method": "web",
        "web_url": "https://api.together.ai/settings/api-keys",
        "config_key": "together_api_key",
        "env_var": "TOGETHER_API_KEY"
    },
    "groq": {
        "name": "Groq",
        "rotation_method": "web",
        "web_url": "https://console.groq.com/keys",
        "config_key": "groq_api_key",
        "env_var": "GROQ_API_KEY"
    },
    "xai": {
        "name": "X.ai/Grok",
        "rotation_method": "web",
        "web_url": "https://x.ai/api",
        "config_key": "xai_api_key",
        "env_var": "XAI_API_KEY"
    }
}

# Define paths
HOME = Path.home()
CONFIG_FILE = HOME / ".voyeur_chat_config.json"
ZSHRC_FILE = HOME / ".zshrc"
BASHRC_FILE = HOME / ".bashrc"
LOG_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "key_rotation_logs"

# Create log directory if it doesn't exist
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)

# Set up logging
LOG_FILE = LOG_DIR / f"key_rotation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

def log_message(message, level="INFO"):
    """Log a message to both console and log file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {message}"
    
    print(formatted_msg)
    
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def is_tool_available(name):
    """Check if a command-line tool is available"""
    return shutil.which(name) is not None

def load_config():
    """Load the LLM-Chat config file if it exists"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            log_message(f"Error parsing config file: {CONFIG_FILE}", "ERROR")
            return {}
    return {}

def save_config(config_data):
    """Save the updated config file"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=2)
        log_message(f"Updated config file: {CONFIG_FILE}")
        return True
    except Exception as e:
        log_message(f"Error saving config file: {e}", "ERROR")
        return False

def update_shell_profile(key, value, shell_file):
    """Update shell profile with new environment variable"""
    if not shell_file.exists():
        log_message(f"Shell profile not found: {shell_file}", "WARNING")
        return False
    
    lines = []
    with open(shell_file, "r") as f:
        lines = f.readlines()
    
    # Check if the key already exists
    key_pattern = f'export {key}='
    key_exists = any(key_pattern in line for line in lines)
    
    # If key exists, update it
    if key_exists:
        new_lines = []
        for line in lines:
            if key_pattern in line:
                new_lines.append(f'export {key}="{value}"\n')
            else:
                new_lines.append(line)
        
        with open(shell_file, "w") as f:
            f.writelines(new_lines)
    else:
        # If key doesn't exist, append it
        with open(shell_file, "a") as f:
            f.write(f'\n# Added by LLM-Chat key rotation tool - {datetime.datetime.now().strftime("%Y-%m-%d")}\n')
            f.write(f'export {key}="{value}"\n')
    
    log_message(f"Updated {key} in {shell_file}")
    return True

def update_env_vars(keys_dict):
    """Update environment variables in shell profiles"""
    shell_files = []
    
    # Determine which shell profile files exist
    if ZSHRC_FILE.exists():
        shell_files.append(ZSHRC_FILE)
    if BASHRC_FILE.exists():
        shell_files.append(BASHRC_FILE)
    
    if not shell_files:
        log_message("No supported shell profiles found", "WARNING")
        return
    
    for key, value in keys_dict.items():
        for shell_file in shell_files:
            update_shell_profile(key, value, shell_file)
    
    log_message("Shell profiles updated. Please run 'source ~/.zshrc' or 'source ~/.bashrc' to apply changes")

def rotate_keys_manual(services_to_rotate):
    """Guide the user through manual key rotation"""
    results = {}
    config = load_config()
    env_var_updates = {}
    nordpass_updates = {}
    
    total_services = len(services_to_rotate)
    current = 1
    
    for service_id in services_to_rotate:
        service = SERVICES.get(service_id)
        if not service:
            log_message(f"Service {service_id} not found in configuration", "ERROR")
            continue
        
        log_message(f"\n[{current}/{total_services}] Processing {service['name']}...")
        
        # Open web interface for manual rotation
        webbrowser.open(service['web_url'])
        log_message(f"Browser opened to {service['web_url']}")
        
        # Wait for user to complete the rotation
        new_key = input(f"\nEnter the new API key for {service['name']} (or press Enter to skip): ").strip()
        
        if new_key:
            # Update results
            results[service_id] = {
                "status": "rotated",
                "name": service['name'],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Update config file if the key exists there
            config_key = service.get("config_key")
            if config_key and config_key in config:
                config[config_key] = new_key
                log_message(f"Updated {config_key} in config")
            
            # Add to environment variable updates
            env_var = service.get("env_var")
            if env_var:
                env_var_updates[env_var] = new_key
            
            # Add to NordPass updates
            nordpass_updates[service['name']] = new_key
        else:
            results[service_id] = {
                "status": "skipped",
                "name": service['name'],
                "timestamp": datetime.datetime.now().isoformat()
            }
            log_message(f"Skipped {service['name']}")
        
        current += 1
    
    # Save updated config
    if config:
        save_config(config)
    
    # Update environment variables
    if env_var_updates:
        if input("\nUpdate environment variables in shell profiles? (y/n): ").lower() == 'y':
            update_env_vars(env_var_updates)
    
    # Prepare NordPass updates
    if nordpass_updates:
        nordpass_text = "\n=== NORDPASS UPDATES ===\n"
        for name, key in nordpass_updates.items():
            nordpass_text += f"{name}: {key}\n"
        
        nordpass_file = LOG_DIR / f"nordpass_updates_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(nordpass_file, "w") as f:
            f.write(nordpass_text)
        
        log_message(f"NordPass update information saved to {nordpass_file}")
    
    return results

def generate_report(results):
    """Generate a detailed report of the key rotation process"""
    report_file = LOG_DIR / f"rotation_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    
    with open(report_file, "w") as f:
        f.write("# API Key Rotation Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        rotated = sum(1 for r in results.values() if r["status"] == "rotated")
        skipped = sum(1 for r in results.values() if r["status"] == "skipped")
        failed = sum(1 for r in results.values() if r["status"] == "error")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total services processed**: {len(results)}\n")
        f.write(f"- **Successfully rotated**: {rotated}\n")
        f.write(f"- **Skipped**: {skipped}\n")
        f.write(f"- **Failed**: {failed}\n\n")
        
        # Details
        f.write("## Details\n\n")
        f.write("| Service | Status | Timestamp |\n")
        f.write("|---------|--------|----------|\n")
        
        for service_id, result in results.items():
            status = result["status"].capitalize()
            name = result["name"]
            timestamp = datetime.datetime.fromisoformat(result["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"| {name} | {status} | {timestamp} |\n")
    
    log_message(f"\nDetailed report saved to {report_file}")
    return report_file

def update_project_configs():
    """Update API keys in project configuration files"""
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Look for config files in the project
    config_files = []
    for config_path in [
        project_dir / "LLMChat-v2.0.0" / "config.py", 
        project_dir / "LLMChat-v2.1.0" / "config.py"
    ]:
        if config_path.exists():
            config_files.append(config_path)
    
    if not config_files:
        log_message("No project config files found to update", "WARNING")
        return
    
    # Load user config
    user_config = load_config()
    if not user_config:
        log_message("No user config found to sync with project", "WARNING")
        return
    
    # Ask user if they want to update project config files
    if input("\nUpdate project config files with new API keys? (y/n): ").lower() == 'y':
        for config_file in config_files:
            log_message(f"Updating {config_file}")
            # This is simplified - a real implementation would need to parse and update Python files
            # which is more complex than JSON files
            log_message("Note: Manual update of project config files may be required", "WARNING")

def main():
    parser = argparse.ArgumentParser(description="API Key Rotation Tool for LLM-Chat")
    parser.add_argument("--services", nargs="+", help="Specific services to rotate keys for")
    parser.add_argument("--all", action="store_true", help="Rotate keys for all configured services")
    parser.add_argument("--list", action="store_true", help="List all available services")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable services:")
        for service_id, service in SERVICES.items():
            print(f"  {service_id}: {service['name']}")
        sys.exit(0)
    
    log_message("Starting API Key Rotation Tool")
    log_message(f"System: {platform.system()} {platform.release()}")
    log_message(f"User: {getpass.getuser()}")
    
    services_to_rotate = []
    
    if args.all:
        services_to_rotate = list(SERVICES.keys())
    elif args.services:
        for service_id in args.services:
            if service_id in SERVICES:
                services_to_rotate.append(service_id)
            else:
                log_message(f"Unknown service: {service_id}", "WARNING")
    else:
        # Interactive mode
        print("\nAvailable services:")
        for i, (service_id, service) in enumerate(SERVICES.items(), 1):
            print(f"{i}. {service['name']}")
        
        print("\nEnter the numbers of the services you want to rotate (comma-separated), or 'all'")
        selection = input("> ").strip().lower()
        
        if selection == "all":
            services_to_rotate = list(SERVICES.keys())
        else:
            try:
                # Parse user selection (1-based indices)
                indices = [int(x.strip()) - 1 for x in selection.split(",")]
                service_items = list(SERVICES.items())
                for idx in indices:
                    if 0 <= idx < len(service_items):
                        service_id, _ = service_items[idx]
                        services_to_rotate.append(service_id)
            except ValueError:
                log_message("Invalid selection", "ERROR")
                sys.exit(1)
    
    if not services_to_rotate:
        log_message("No services selected for rotation", "WARNING")
        sys.exit(0)
    
    log_message(f"Selected services for rotation: {', '.join(SERVICES[s]['name'] for s in services_to_rotate)}")
    
    # Perform key rotation
    results = rotate_keys_manual(services_to_rotate)
    
    # Generate report
    report_file = generate_report(results)
    
    # Update project configs
    update_project_configs()
    
    log_message("\nKey rotation process completed!")
    log_message(f"Log file: {LOG_FILE}")
    log_message(f"Report file: {report_file}")
    
    log_message("\nReminder: You may need to:")
    log_message("1. Source your shell profile to apply environment variable changes")
    log_message("2. Update NordPass entries with the new keys")
    log_message("3. Revoke old API keys through service web interfaces")

if __name__ == "__main__":
    main()