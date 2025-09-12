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
# Script Name: systems-nordvpn-servers.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible NordVPN server listing tool with advanced filtering
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Advanced server filtering by location and load
#     - Country-based server selection with performance metrics
#     - Feature-based filtering for SSL proxy support
#     - Command-line interface with native macOS integration
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - requests, argparse (auto-installed if missing)
#     - Internet connection
#
# Usage:
#     python systems-nordvpn-servers.py
#
####################################################################################

import os
import sys
import json
import datetime
import subprocess
import argparse
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

def install_and_import(package_name: str, import_name: str = None):
    """Auto-install and import packages with error handling"""
    if import_name is None:
        import_name = package_name
    
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return __import__(import_name)
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return None

# Auto-install and import required packages
requests = install_and_import("requests")
tkinter = install_and_import("tkinter")
if tkinter:
    from tkinter import messagebox, ttk
    import tkinter as tk

class NordVPNServerManager:
    def __init__(self):
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / "nordvpn_servers_log.txt"
        self.servers_cache_file = self.desktop_path / "nordvpn_servers_cache.json"
        
        # NordVPN API endpoints
        self.api_base = "https://api.nordvpn.com"
        self.servers_endpoint = f"{self.api_base}/v1/servers"
        self.countries_endpoint = f"{self.api_base}/v1/servers/countries"
        
        self.servers_data = []
        self.countries_data = []
        self.filtered_servers = []
        
        # Server feature mapping
        self.feature_map = {
            1: "ikev2",
            3: "socks",
            5: "proxy",
            7: "http_proxy_ssl",
            9: "http_cyber_sec",
            11: "pptp",
            13: "l2tp",
            15: "openvpn_udp",
            17: "openvpn_tcp",
            19: "proxy_cyber_sec",
            21: "proxy_ssl",
            23: "proxy_ssl_cyber_sec",
            25: "ikev2_v6",
            27: "openvpn_udp_v6",
            29: "openvpn_tcp_v6",
            31: "proxy_udp",
            33: "proxy_tcp",
            35: "proxy_ssl_tcp",
            37: "proxy_ssl_udp",
            39: "openvpn_udp_tls_crypt",
            41: "openvpn_tcp_tls_crypt",
            43: "openvpn_dedicated_udp",
            45: "openvpn_dedicated_tcp"
        }
        
        self.create_gui()
        
    def create_gui(self):
        """Create the main GUI interface"""
        if not tkinter:
            print("GUI not available, running in console mode...")
            return
            
        self.root = tk.Tk()
        self.root.title("GET SWIFTY - NordVPN Server Manager v1.0.0")
        self.root.geometry("800x700")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="NordVPN Server Manager", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Filters frame
        filters_frame = ttk.LabelFrame(main_frame, text="Server Filters", padding="10")
        filters_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Country filter
        ttk.Label(filters_frame, text="Country:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(filters_frame, textvariable=self.country_var, width=20)
        self.country_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Load filter
        ttk.Label(filters_frame, text="Max Load %:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.load_var = tk.StringVar(value="50")
        self.load_spin = tk.Spinbox(filters_frame, from_=0, to=100, textvariable=self.load_var, width=10)
        self.load_spin.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Feature filter
        ttk.Label(filters_frame, text="Features:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0), padx=(0, 5))
        self.feature_var = tk.StringVar()
        self.feature_combo = ttk.Combobox(filters_frame, textvariable=self.feature_var, width=20)
        self.feature_combo['values'] = ["All"] + list(self.feature_map.values())
        self.feature_combo.set("All")
        self.feature_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0), padx=(0, 20))
        
        # Limit results
        ttk.Label(filters_frame, text="Limit Results:").grid(row=1, column=2, sticky=tk.W, pady=(10, 0), padx=(0, 5))
        self.limit_var = tk.StringVar(value="50")
        self.limit_spin = tk.Spinbox(filters_frame, from_=1, to=1000, textvariable=self.limit_var, width=10)
        self.limit_spin.grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        self.refresh_button = ttk.Button(buttons_frame, text="Refresh Servers", command=self.refresh_servers)
        self.refresh_button.grid(row=0, column=0, padx=5)
        
        self.filter_button = ttk.Button(buttons_frame, text="Apply Filters", command=self.apply_filters)
        self.filter_button.grid(row=0, column=1, padx=5)
        
        self.export_button = ttk.Button(buttons_frame, text="Export to CSV", command=self.export_to_csv)
        self.export_button.grid(row=0, column=2, padx=5)
        
        self.ping_button = ttk.Button(buttons_frame, text="Test Selected", command=self.test_selected_servers)
        self.ping_button.grid(row=0, column=3, padx=5)
        
        ttk.Button(buttons_frame, text="View Log", command=self.view_log).grid(row=0, column=4, padx=5)
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Server list frame
        list_frame = ttk.LabelFrame(main_frame, text="Servers", padding="10")
        list_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Treeview for server list
        columns = ("Name", "Country", "City", "Load", "Features", "IP")
        self.server_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Configure column headings and widths
        self.server_tree.heading("Name", text="Server Name")
        self.server_tree.heading("Country", text="Country")
        self.server_tree.heading("City", text="City")
        self.server_tree.heading("Load", text="Load %")
        self.server_tree.heading("Features", text="Features")
        self.server_tree.heading("IP", text="IP Address")
        
        self.server_tree.column("Name", width=120)
        self.server_tree.column("Country", width=100)
        self.server_tree.column("City", width=100)
        self.server_tree.column("Load", width=80)
        self.server_tree.column("Features", width=200)
        self.server_tree.column("IP", width=120)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.server_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=self.server_tree.xview)
        self.server_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.server_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Server details frame
        details_frame = ttk.LabelFrame(main_frame, text="Server Details", padding="10")
        details_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.details_text = tk.Text(details_frame, height=8, wrap=tk.WORD)
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Bind events
        self.server_tree.bind("<<TreeviewSelect>>", self.on_server_select)
        
        # Load cached data if available
        self.load_cached_data()
        
    def log_message(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Logging error: {e}")
    
    def fetch_servers_data(self) -> bool:
        """Fetch server data from NordVPN API"""
        if not requests:
            self.log_message("Requests library not available")
            return False
            
        try:
            self.log_message("Fetching server data from NordVPN API...")
            
            # Fetch servers
            response = requests.get(self.servers_endpoint, timeout=30)
            response.raise_for_status()
            self.servers_data = response.json()
            
            # Fetch countries
            response = requests.get(self.countries_endpoint, timeout=30)
            response.raise_for_status()
            self.countries_data = response.json()
            
            # Cache the data
            cache_data = {
                'servers': self.servers_data,
                'countries': self.countries_data,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(self.servers_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.log_message(f"Fetched {len(self.servers_data)} servers from {len(self.countries_data)} countries")
            return True
            
        except Exception as e:
            self.log_message(f"Error fetching server data: {e}")
            return False
    
    def load_cached_data(self):
        """Load cached server data if available"""
        try:
            if self.servers_cache_file.exists():
                with open(self.servers_cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is recent (less than 24 hours old)
                cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.datetime.now() - cache_time).total_seconds() < 86400:
                    self.servers_data = cache_data['servers']
                    self.countries_data = cache_data['countries']
                    self.update_country_list()
                    self.apply_filters()
                    self.log_message(f"Loaded cached data: {len(self.servers_data)} servers")
                    return
            
            # Cache is old or doesn't exist, fetch fresh data
            self.refresh_servers()
            
        except Exception as e:
            self.log_message(f"Error loading cached data: {e}")
            self.refresh_servers()
    
    def update_country_list(self):
        """Update the country dropdown with available countries"""
        if hasattr(self, 'country_combo'):
            countries = ["All"] + sorted([country['name'] for country in self.countries_data])
            self.country_combo['values'] = countries
            self.country_combo.set("All")
    
    def refresh_servers(self):
        """Refresh server data from API"""
        def refresh_thread():
            try:
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.start()
                    self.progress_var.set("Fetching server data...")
                    self.refresh_button.config(state='disabled')
                
                if self.fetch_servers_data():
                    self.update_country_list()
                    self.apply_filters()
                    
                    if hasattr(self, 'progress_var'):
                        self.progress_var.set("Server data updated")
                        messagebox.showinfo("Success", f"Fetched {len(self.servers_data)} servers")
                else:
                    if hasattr(self, 'progress_var'):
                        self.progress_var.set("Failed to fetch data")
                        messagebox.showerror("Error", "Failed to fetch server data")
                
            except Exception as e:
                self.log_message(f"Refresh error: {e}")
                if hasattr(self, 'progress_var'):
                    messagebox.showerror("Error", f"Refresh failed: {e}")
            
            finally:
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.stop()
                    self.refresh_button.config(state='normal')
        
        threading.Thread(target=refresh_thread, daemon=True).start()
    
    def apply_filters(self):
        """Apply filters to server list"""
        if not self.servers_data:
            return
        
        try:
            # Get filter values
            country_filter = self.country_var.get() if hasattr(self, 'country_var') else "All"
            max_load = int(self.load_var.get()) if hasattr(self, 'load_var') else 100
            feature_filter = self.feature_var.get() if hasattr(self, 'feature_var') else "All"
            limit = int(self.limit_var.get()) if hasattr(self, 'limit_var') else 50
            
            # Filter servers
            filtered = []
            
            for server in self.servers_data:
                # Skip if no location data
                if 'locations' not in server or not server['locations']:
                    continue
                
                location = server['locations'][0]
                
                # Country filter
                if country_filter != "All" and location['country']['name'] != country_filter:
                    continue
                
                # Load filter
                if server['load'] > max_load:
                    continue
                
                # Feature filter
                if feature_filter != "All":
                    feature_id = None
                    for fid, fname in self.feature_map.items():
                        if fname == feature_filter:
                            feature_id = fid
                            break
                    
                    if feature_id and not any(tech['id'] == feature_id for tech in server.get('technologies', [])):
                        continue
                
                filtered.append(server)
            
            # Sort by load (ascending)
            filtered.sort(key=lambda x: x['load'])
            
            # Apply limit
            self.filtered_servers = filtered[:limit]
            
            # Update display
            self.update_server_list()
            
            self.log_message(f"Applied filters: {len(self.filtered_servers)} servers match criteria")
            
        except Exception as e:
            self.log_message(f"Filter error: {e}")
    
    def update_server_list(self):
        """Update the server list display"""
        if not hasattr(self, 'server_tree'):
            return
        
        # Clear existing items
        for item in self.server_tree.get_children():
            self.server_tree.delete(item)
        
        # Add filtered servers
        for server in self.filtered_servers:
            if not server.get('locations'):
                continue
                
            location = server['locations'][0]
            
            # Get server features
            features = []
            for tech in server.get('technologies', []):
                if tech['id'] in self.feature_map:
                    features.append(self.feature_map[tech['id']])
            
            features_str = ", ".join(features[:3])  # Show first 3 features
            if len(features) > 3:
                features_str += f" (+{len(features) - 3} more)"
            
            # Get IP address
            ip_address = server.get('station', '')
            
            self.server_tree.insert('', 'end', values=(
                server['name'],
                location['country']['name'],
                location['country']['city']['name'],
                f"{server['load']}%",
                features_str,
                ip_address
            ))
    
    def on_server_select(self, event):
        """Handle server selection"""
        selection = self.server_tree.selection()
        if not selection or not hasattr(self, 'details_text'):
            return
        
        # Get selected server name
        item = self.server_tree.item(selection[0])
        server_name = item['values'][0]
        
        # Find server in filtered list
        selected_server = None
        for server in self.filtered_servers:
            if server['name'] == server_name:
                selected_server = server
                break
        
        if not selected_server:
            return
        
        # Display server details
        self.details_text.delete(1.0, tk.END)
        
        location = selected_server['locations'][0] if selected_server.get('locations') else {}
        
        details = f"""Server Name: {selected_server['name']}
Hostname: {selected_server.get('hostname', 'N/A')}
Station IP: {selected_server.get('station', 'N/A')}
Load: {selected_server['load']}%
Status: {selected_server.get('status', 'Unknown')}

Location Information:
Country: {location.get('country', {}).get('name', 'N/A')}
City: {location.get('country', {}).get('city', {}).get('name', 'N/A')}

Supported Technologies:
"""
        
        for tech in selected_server.get('technologies', []):
            tech_name = self.feature_map.get(tech['id'], f"Unknown ({tech['id']})")
            details += f"  • {tech_name}\n"
        
        if selected_server.get('groups'):
            details += f"\nServer Groups:\n"
            for group in selected_server['groups']:
                details += f"  • {group.get('title', 'Unknown')}\n"
        
        self.details_text.insert(1.0, details)
    
    def test_selected_servers(self):
        """Test ping to selected servers"""
        selection = self.server_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select servers to test")
            return
        
        def test_thread():
            try:
                self.progress_bar.start()
                self.progress_var.set("Testing server connectivity...")
                
                results = []
                
                for item_id in selection:
                    item = self.server_tree.item(item_id)
                    server_name = item['values'][0]
                    ip_address = item['values'][5]
                    
                    if ip_address:
                        # Ping test
                        try:
                            result = subprocess.run(['ping', '-c', '3', ip_address], 
                                                  capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                # Extract average ping time
                                lines = result.stdout.split('\n')
                                for line in lines:
                                    if 'avg' in line:
                                        ping_time = line.split('/')[-3]
                                        results.append(f"{server_name}: {ping_time}ms")
                                        break
                                else:
                                    results.append(f"{server_name}: Reachable")
                            else:
                                results.append(f"{server_name}: Unreachable")
                        except subprocess.TimeoutExpired:
                            results.append(f"{server_name}: Timeout")
                        except Exception as e:
                            results.append(f"{server_name}: Error - {e}")
                
                # Show results
                result_text = "\n".join(results)
                messagebox.showinfo("Ping Test Results", result_text)
                self.log_message(f"Ping test completed: {len(results)} servers tested")
                
            except Exception as e:
                self.log_message(f"Ping test error: {e}")
                messagebox.showerror("Error", f"Ping test failed: {e}")
            
            finally:
                self.progress_bar.stop()
                self.progress_var.set("Ready")
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def export_to_csv(self):
        """Export filtered servers to CSV"""
        if not self.filtered_servers:
            messagebox.showwarning("Warning", "No servers to export")
            return
        
        try:
            csv_file = self.desktop_path / f"nordvpn_servers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(csv_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("Name,Hostname,Country,City,Load,IP,Features\n")
                
                # Data
                for server in self.filtered_servers:
                    location = server['locations'][0] if server.get('locations') else {}
                    
                    features = []
                    for tech in server.get('technologies', []):
                        if tech['id'] in self.feature_map:
                            features.append(self.feature_map[tech['id']])
                    
                    f.write(f'"{server["name"]}",'
                           f'"{server.get("hostname", "")}",'
                           f'"{location.get("country", {}).get("name", "")}",'
                           f'"{location.get("country", {}).get("city", {}).get("name", "")}",'
                           f'{server["load"]},'
                           f'"{server.get("station", "")}",'
                           f'"{", ".join(features)}"\n')
            
            self.log_message(f"Exported {len(self.filtered_servers)} servers to {csv_file}")
            messagebox.showinfo("Export Complete", f"Servers exported to:\n{csv_file}")
            
        except Exception as e:
            self.log_message(f"Export error: {e}")
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def view_log(self):
        """Open the log file"""
        try:
            if self.log_file.exists():
                os.system(f'open "{self.log_file}"')
            else:
                messagebox.showinfo("Log", "No log file found")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open log file: {e}")
    
    def run(self):
        """Run the application"""
        if tkinter:
            self.log_message("GET SWIFTY NordVPN Server Manager v1.0.0 started")
            self.root.mainloop()
        else:
            self.console_mode()
    
    def console_mode(self):
        """Run in console mode if GUI is not available"""
        print("Running in console mode...")
        
        # Fetch server data
        if not self.fetch_servers_data():
            print("Failed to fetch server data")
            return
        
        # Simple filter by country
        country = input("Enter country name (or 'all' for all countries): ").strip()
        max_load = input("Enter maximum load percentage (default 50): ").strip()
        
        try:
            max_load = int(max_load) if max_load else 50
        except ValueError:
            max_load = 50
        
        # Filter servers
        filtered = []
        for server in self.servers_data:
            if not server.get('locations'):
                continue
                
            location = server['locations'][0]
            
            if country.lower() != 'all' and location['country']['name'].lower() != country.lower():
                continue
                
            if server['load'] > max_load:
                continue
            
            filtered.append(server)
        
        # Sort and display
        filtered.sort(key=lambda x: x['load'])
        
        print(f"\nFound {len(filtered)} servers:")
        print("-" * 80)
        print(f"{'Name':<20} {'Country':<15} {'City':<15} {'Load':<8} {'IP'}")
        print("-" * 80)
        
        for server in filtered[:50]:  # Show first 50
            location = server['locations'][0]
            print(f"{server['name']:<20} "
                  f"{location['country']['name']:<15} "
                  f"{location['country']['city']['name']:<15} "
                  f"{server['load']}%{'':<5} "
                  f"{server.get('station', '')}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="NordVPN Server Manager")
    parser.add_argument("--country", help="Filter by country")
    parser.add_argument("--max-load", type=int, default=50, help="Maximum load percentage")
    parser.add_argument("--feature", help="Filter by feature")
    parser.add_argument("--console", action="store_true", help="Run in console mode")
    
    args = parser.parse_args()
    
    try:
        manager = NordVPNServerManager()
        
        if args.console or not tkinter:
            manager.console_mode()
        else:
            manager.run()
            
    except Exception as e:
        print(f"Application error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()