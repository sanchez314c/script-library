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
# Script Name: systems-nordvpn-fetch.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible NordVPN server information retriever with API integration
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - API integration with credential handling and server filtering
#     - Multi-threaded server testing with performance metrics
#     - GUI progress tracking with native macOS dialogs
#     - Secure output file generation with collision handling
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - requests, tkinter (auto-installed if missing)
#     - Internet connection and write permissions
#
# Usage:
#     python systems-nordvpn-fetch.py
#
####################################################################################

import os
import sys
import json
import time
import argparse
import threading
import multiprocessing
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from datetime import datetime

try:
    import requests
    from requests.exceptions import RequestException
except ImportError:
    print("Missing required dependency: requests")
    print("Please install with: pip install requests")
    sys.exit(1)


# Constants
NORDVPN_API_URL = "https://api.nordvpn.com/v1/servers"
NORDVPN_COUNTRY_URL = "https://api.nordvpn.com/v1/countries"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Documents/NordVPN")
CONFIG_FILE = os.path.expanduser("~/.nordvpn_fetch_config.json")
REQUEST_TIMEOUT = 10  # seconds


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    try:
        import requests
        return True
    except ImportError:
        return False


def install_dependencies() -> bool:
    """Attempt to install missing dependencies."""
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        return True
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        return False


def select_directory(title: str = "Select Folder") -> Optional[str]:
    """
    Open a native macOS dialog to select a directory.
    
    Args:
        title: Dialog window title
        
    Returns:
        Selected directory path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected


def select_output_file(title: str = "Save Server List", 
                      default_name: str = "nordvpn_servers.txt") -> Optional[str]:
    """
    Open a native macOS dialog to select an output file.
    
    Args:
        title: Dialog window title
        default_name: Default filename
        
    Returns:
        Selected file path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


def get_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Get NordVPN API credentials from config file or prompt user.
    
    Returns:
        Tuple of (username, password) or (None, None) if cancelled
    """
    # Check if credentials exist in config file
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            username = config.get('username')
            password = config.get('password')
            
            # If credentials exist and are valid, return them
            if username and password:
                return username, password
        except Exception:
            # If there's an error reading the config file, continue with prompting
            pass
    
    # Prompt for credentials if not found in config
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    username = simpledialog.askstring("NordVPN Credentials", "Enter your NordVPN username:", parent=root)
    if not username:
        root.destroy()
        return None, None
    
    password = simpledialog.askstring("NordVPN Credentials", "Enter your NordVPN password:", show='*', parent=root)
    root.destroy()
    
    if not password:
        return None, None
    
    # Ask if credentials should be saved
    root = tk.Tk()
    root.withdraw()
    should_save = messagebox.askyesno(
        "Save Credentials",
        "Do you want to save your credentials for future use?\n" +
        "Note: Credentials will be stored in a local file."
    )
    root.destroy()
    
    if should_save:
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump({'username': username, 'password': password}, f)
            os.chmod(CONFIG_FILE, 0o600)  # Set permissions to user read/write only
        except Exception as e:
            print(f"Failed to save credentials: {e}")
    
    return username, password


def create_progress_ui() -> Tuple[tk.Tk, ttk.Progressbar, ttk.Label, ttk.Label]:
    """
    Create a progress tracking UI.
    
    Returns:
        Tuple of (window, progress_bar, status_label, detail_label)
    """
    window = tk.Tk()
    window.title("NordVPN Server Fetch")
    window.geometry("500x200")
    
    # Create frame for progress bar
    frame = ttk.Frame(window, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Add status label
    status_label = ttk.Label(frame, text="Initializing...", font=("Helvetica", 12))
    status_label.pack(pady=10)
    
    # Add detail label
    detail_label = ttk.Label(frame, text="")
    detail_label.pack(pady=5)
    
    # Add progress bar
    progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
    progress_bar.pack(pady=10)
    
    return window, progress_bar, status_label, detail_label


def fetch_countries(session: requests.Session) -> List[Dict[str, Any]]:
    """
    Fetch list of countries from NordVPN API.
    
    Args:
        session: Requests session
        
    Returns:
        List of country dictionaries
    """
    try:
        response = session.get(NORDVPN_COUNTRY_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching countries: {e}")
        return []


def fetch_servers(session: requests.Session, 
                 country_id: Optional[int] = None, 
                 progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
    """
    Fetch servers from NordVPN API.
    
    Args:
        session: Requests session
        country_id: Optional country ID to filter by
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of server dictionaries
    """
    try:
        url = NORDVPN_API_URL
        params = {}
        
        if country_id is not None:
            params['filters[country_id]'] = country_id
        
        if progress_callback:
            progress_callback("Connecting to NordVPN API...", "Fetching server data")
        
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        if progress_callback:
            progress_callback("Processing server data...", f"Received {len(response.text)} bytes")
        
        return response.json()
    except Exception as e:
        if progress_callback:
            progress_callback("Error fetching servers", str(e))
        print(f"Error fetching servers: {e}")
        return []


def test_server_speed(server: Dict[str, Any], session: requests.Session) -> float:
    """
    Test connection speed to a server.
    
    Args:
        server: Server dictionary
        session: Requests session
        
    Returns:
        Response time in milliseconds or -1 if failed
    """
    try:
        hostname = server.get('hostname', '')
        if not hostname:
            return -1
        
        # Use the hostname to ping the server
        start_time = time.time()
        response = session.get(f"https://{hostname}/", 
                              timeout=REQUEST_TIMEOUT,
                              verify=False)  # Skip SSL verification for speed test
        response_time = (time.time() - start_time) * 1000
        
        return response_time
    except Exception:
        return -1


def format_server_info(server: Dict[str, Any], speed: float) -> str:
    """
    Format server information for output.
    
    Args:
        server: Server dictionary
        speed: Speed test result
        
    Returns:
        Formatted server information string
    """
    hostname = server.get('hostname', 'unknown')
    country = server.get('country', {}).get('name', 'Unknown')
    city = server.get('city', {}).get('name', 'Unknown')
    load = server.get('load', -1)
    
    speed_str = f"{speed:.1f}ms" if speed > 0 else "timeout"
    
    return f"{hostname} | {country}, {city} | Load: {load}% | Ping: {speed_str}"


def process_servers(servers: List[Dict[str, Any]], 
                   max_servers: int = 50,
                   progress_callback: Optional[callable] = None) -> List[str]:
    """
    Process servers, test speeds, and format output.
    
    Args:
        servers: List of server dictionaries
        max_servers: Maximum number of servers to process
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of formatted server strings
    """
    if not servers:
        return []
    
    # Limit the number of servers to process
    servers = sorted(servers, key=lambda s: s.get('load', 100))[:max_servers]
    total_servers = len(servers)
    
    if progress_callback:
        progress_callback("Testing server speeds...", f"0/{total_servers} servers tested")
    
    results = []
    session = requests.Session()
    session.verify = False  # Disable SSL verification for speed tests
    
    # Process servers with multithreading
    with ThreadPoolExecutor(max_workers=min(10, multiprocessing.cpu_count() * 2)) as executor:
        future_to_server = {executor.submit(test_server_speed, server, session): server for server in servers}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_server)):
            server = future_to_server[future]
            speed = future.result()
            results.append((server, speed))
            
            if progress_callback:
                progress_callback(
                    "Testing server speeds...",
                    f"{i+1}/{total_servers} servers tested",
                    (i+1) / total_servers * 100
                )
    
    # Sort results by speed (fastest first)
    results.sort(key=lambda x: float('inf') if x[1] < 0 else x[1])
    
    # Format results
    return [format_server_info(server, speed) for server, speed in results]


def save_server_list(servers: List[str], output_file: str) -> bool:
    """
    Save server list to file.
    
    Args:
        servers: List of formatted server strings
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(f"# NordVPN Server List\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Servers: {len(servers)}\n\n")
            
            for server in servers:
                f.write(f"{server}\n")
        
        return True
    except Exception as e:
        print(f"Error saving server list: {e}")
        return False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Fetch and process NordVPN server information")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--country", type=str, help="Filter by country name")
    parser.add_argument("--max-servers", type=int, default=50, help="Maximum number of servers to include")
    parser.add_argument("--no-speed-test", action="store_true", help="Skip speed testing")
    return parser.parse_args()


def main() -> None:
    """Main function."""
    # Check and install dependencies if needed
    if not check_dependencies():
        print("Installing required dependencies...")
        if not install_dependencies():
            print("Failed to install dependencies. Please install manually:")
            print("pip install requests")
            return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get output file path
    output_file = args.output
    if not output_file:
        output_file = select_output_file()
        if not output_file:
            print("No output file selected. Exiting.")
            return
    
    # Create progress UI
    window, progress_bar, status_label, detail_label = create_progress_ui()
    
    # Create progress callback
    def update_progress(status: str, detail: str = "", percent: float = None) -> None:
        status_label.config(text=status)
        detail_label.config(text=detail)
        if percent is not None:
            progress_bar.config(value=percent)
        window.update()
    
    # Function to run the fetching process
    def run_fetch_process() -> None:
        try:
            # Get credentials if needed
            username, password = None, None
            
            # Initialize session
            session = requests.Session()
            
            # Create error handler
            def show_error(message: str) -> None:
                messagebox.showerror("Error", message)
                window.destroy()
            
            # Fetch countries
            update_progress("Fetching country list...", "Connecting to NordVPN API")
            countries = fetch_countries(session)
            
            if not countries:
                show_error("Failed to fetch country list. Please check your internet connection.")
                return
            
            # Filter by country if specified
            country_id = None
            if args.country:
                country_name = args.country.lower()
                country_matches = [c for c in countries if c.get('name', '').lower() == country_name]
                
                if not country_matches:
                    # If exact match not found, show close matches
                    close_matches = [c for c in countries if country_name in c.get('name', '').lower()]
                    if close_matches:
                        country_list = "\n".join([f"- {c.get('name')}" for c in close_matches[:10]])
                        show_error(f"Country '{args.country}' not found. Did you mean one of these?\n{country_list}")
                    else:
                        show_error(f"Country '{args.country}' not found in NordVPN's server list.")
                    return
                
                country_id = country_matches[0].get('id')
                update_progress("Fetching servers...", f"Filtering by country: {country_matches[0].get('name')}")
            else:
                update_progress("Fetching servers...", "Retrieving all available servers")
            
            # Fetch servers
            servers = fetch_servers(session, country_id, update_progress)
            
            if not servers:
                show_error("Failed to fetch server list. Please check your internet connection.")
                return
            
            update_progress("Processing servers...", f"Found {len(servers)} servers", 30)
            
            # Process servers
            if args.no_speed_test:
                # Skip speed testing
                formatted_servers = [format_server_info(server, -1) for server in servers[:args.max_servers]]
                for i, _ in enumerate(formatted_servers):
                    update_progress(
                        "Processing servers...",
                        f"{i+1}/{len(formatted_servers)} servers processed",
                        30 + (i+1) / len(formatted_servers) * 70
                    )
            else:
                # Test speeds and format output
                formatted_servers = process_servers(
                    servers, 
                    args.max_servers, 
                    lambda status, detail, percent=None: update_progress(
                        status, detail, 30 + (percent or 0) * 0.7
                    )
                )
            
            # Save server list
            update_progress("Saving server list...", f"Writing to {output_file}", 100)
            
            if save_server_list(formatted_servers, output_file):
                messagebox.showinfo("Success", f"Server list saved to {output_file}")
            else:
                show_error(f"Failed to save server list to {output_file}")
            
            # Close window
            window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
            window.destroy()
    
    # Start fetch process in a separate thread
    threading.Thread(target=run_fetch_process, daemon=True).start()
    
    # Start UI main loop
    window.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

