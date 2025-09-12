#!/usr/bin/env python3
####################################################################################
#                                                                                  #
#   ██████╗  ██████╗ ██████╗ ████████╗ █████╗ ██╗          ██████╗ ██╗   ██╗███╗   ██╗#
#   ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗██║         ██╔════╝ ██║   ██║████╗  ██║#
#   ██████╔╝██║   ██║██████╔╝   ██║   ███████║██║         ██║  ███╗██║   ██║██╔██╗ ██║#
#   ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██║██║         ██║   ██║██║   ██║██║╚██╗██║#
#   ██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║███████╗    ╚██████╔╝╚██████╔╝██║ ╚████║#
#   ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝#
#                                                                                  #
####################################################################################
#
# Script Name: portal_gun_launcher.py
# 
# Author: @spacewelder314 🚀
# Dimension: C-137 (Production)
#                                              
# Date Created: 2025-01-30
# Last Modified: 2025-01-30
#
# Version: 1.0.0
#
# Description: Universal launcher for Script.Library - Your portal to automation
#              across dimensions. Features both CLI and GUI interfaces for
#              accessing the entire script multiverse.
#
# Usage: python portal_gun_launcher.py [--gui] [--category CATEGORY] [--list]
#
# Dependencies: customtkinter, rich, Pillow
#
# Portal Gun Compatible: ✅
#
# Notes: This is the master control interface. Handle with care.
#        "Science is more art than science, Morty."
#                                                                                
####################################################################################

"""
Portal Gun Launcher - Script.Library Universal Interface
=========================================================

The ultimate control panel for your automation multiverse.
Navigate through dimensions of scripts with Rick-level genius.

*burp* Let's get schwifty with some automation!
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import importlib.util

# Rich console for beautiful CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich.layout import Layout
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint
    from rich.columns import Columns
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("📦 Installing Rich for enhanced display...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich.layout import Layout
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint
    from rich.columns import Columns
    from rich.text import Text
    RICH_AVAILABLE = True

# GUI imports (optional)
try:
    import customtkinter
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

console = Console()

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = Path.home() / 'Desktop' / 'ScriptLibrary_Logs'
LOG_DIR.mkdir(exist_ok=True)

# Category definitions with emojis and colors
CATEGORIES = {
    'AI-ML': {'emoji': '🤖', 'color': '#00FF41', 'desc': 'Artificial Intelligence & Machine Learning'},
    'Audio': {'emoji': '🎵', 'color': '#FF006E', 'desc': 'Audio Processing & Manipulation'},
    'CLIs': {'emoji': '⚡', 'color': '#FFBE0B', 'desc': 'Command Line Interfaces'},
    'Data Scraping': {'emoji': '📊', 'color': '#39FF14', 'desc': 'Web Scraping & Data Collection'},
    'Documents': {'emoji': '📄', 'color': '#00B4D8', 'desc': 'Document Processing & Conversion'},
    'Forensics': {'emoji': '🔍', 'color': '#FF4365', 'desc': 'Digital Forensics & Analysis'},
    'GITHub': {'emoji': '🐙', 'color': '#6A0DAD', 'desc': 'GitHub Integration & Tools'},
    'GPUs': {'emoji': '⚡', 'color': '#FFD700', 'desc': 'GPU Computing & CUDA'},
    'Images': {'emoji': '🖼️', 'color': '#FF1493', 'desc': 'Image Processing & Manipulation'},
    'JSON': {'emoji': '📋', 'color': '#00CED1', 'desc': 'JSON Processing & Validation'},
    'Mobile': {'emoji': '📱', 'color': '#FF69B4', 'desc': 'Mobile Device Integration'},
    'NLP': {'emoji': '🧠', 'color': '#9370DB', 'desc': 'Natural Language Processing'},
    'System': {'emoji': '🖥️', 'color': '#32CD32', 'desc': 'System Administration & Utilities'},
    'Video': {'emoji': '🎬', 'color': '#FF8C00', 'desc': 'Video Processing & Editing'}
}

# Rick-themed messages
RICK_QUOTES = [
    "Wubba Lubba Dub Dub! Let's automate!",
    "I turned myself into a script launcher, Morty! I'm Launcher Rick!",
    "Sometimes science is more art than science. A lot of people don't get that.",
    "Listen, I'm not the nicest guy in the universe, because I'm the smartest.",
    "What about the reality where Hitler cured cancer? The answer is: Don't think about it.",
    "Nobody exists on purpose. Nobody belongs anywhere. We're all going to die. Come use these scripts.",
    "*burp* Science, biatch!",
    "You gotta get schwifty with automation!",
    "This is why we can't have nice things... so I automated them.",
    "Reality is poison. I choose scripts."
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_portal_gun_logs():
    """Initialize multiverse-compatible logging system"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"portal_gun_{timestamp}_dimension_C137.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | [DIMENSION-C137] | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if not RICH_AVAILABLE else logging.NullHandler()
        ]
    )
    
    logging.info("🚀 Portal Gun initialized - Reality stable")
    logging.info(f"📍 Current dimension: {os.getcwd()}")
    return log_file

# ============================================================================
# SCRIPT DISCOVERY ENGINE
# ============================================================================

class ScriptDiscovery:
    """Discover and catalog scripts across dimensions"""
    
    def __init__(self):
        self.scripts = {}
        self.metadata_cache = {}
        
    def scan_dimension(self, category: str) -> List[Dict]:
        """Scan a category directory for scripts"""
        scripts = []
        category_path = SCRIPT_DIR / category
        
        if not category_path.exists():
            return scripts
            
        # Look for Python and Shell scripts
        for ext in ['*.py', '*.sh']:
            for script_path in category_path.glob(ext):
                if script_path.name == '__init__.py':
                    continue
                    
                script_info = self.analyze_script(script_path, category)
                if script_info:
                    scripts.append(script_info)
                    
        return scripts
    
    def analyze_script(self, path: Path, category: str) -> Optional[Dict]:
        """Extract metadata from a script"""
        try:
            # Check for metadata file first
            metadata_file = path.with_suffix('.metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Parse script header for metadata
                metadata = self.parse_script_header(path)
                
            metadata.update({
                'path': str(path),
                'category': category,
                'filename': path.name,
                'extension': path.suffix
            })
            
            return metadata
            
        except Exception as e:
            logging.warning(f"Failed to analyze {path}: {e}")
            return {
                'path': str(path),
                'category': category,
                'filename': path.name,
                'extension': path.suffix,
                'name': path.stem,
                'description': 'No description available',
                'version': 'Unknown'
            }
    
    def parse_script_header(self, path: Path) -> Dict:
        """Parse metadata from script header comments"""
        metadata = {
            'name': path.stem,
            'description': 'No description available',
            'version': 'Unknown',
            'author': 'Unknown',
            'usage': 'See script for usage'
        }
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:50]  # Only check first 50 lines
                
            for line in lines:
                line = line.strip()
                if 'Script Name:' in line:
                    metadata['name'] = line.split(':', 1)[1].strip()
                elif 'Description:' in line:
                    metadata['description'] = line.split(':', 1)[1].strip()
                elif 'Version:' in line:
                    metadata['version'] = line.split(':', 1)[1].strip()
                elif 'Author:' in line:
                    metadata['author'] = line.split(':', 1)[1].strip()
                elif 'Usage:' in line:
                    metadata['usage'] = line.split(':', 1)[1].strip()
                    
        except Exception as e:
            logging.debug(f"Could not parse header for {path}: {e}")
            
        return metadata
    
    def discover_all(self) -> Dict[str, List[Dict]]:
        """Discover all scripts across all categories"""
        console.print("\n🧬 [bold cyan]Scanning multiverse for scripts...[/bold cyan]\n")
        
        all_scripts = {}
        with console.status("[bold green]Discovering scripts across dimensions...") as status:
            for category in CATEGORIES.keys():
                status.update(f"Scanning {category}...")
                scripts = self.scan_dimension(category)
                if scripts:
                    all_scripts[category] = scripts
                    console.print(f"  {CATEGORIES[category]['emoji']} {category}: [green]{len(scripts)} scripts found[/green]")
                    
        return all_scripts

# ============================================================================
# CLI INTERFACE
# ============================================================================

class PortalGunCLI:
    """Command-line interface for the Portal Gun Launcher"""
    
    def __init__(self):
        self.discovery = ScriptDiscovery()
        self.scripts = {}
        
    def run(self):
        """Main CLI entry point"""
        self.show_banner()
        self.scripts = self.discovery.discover_all()
        
        if not self.scripts:
            console.print("[red]❌ No scripts found in any dimension![/red]")
            console.print("[yellow]Check that script directories exist and contain scripts.[/yellow]")
            return
            
        self.main_menu()
    
    def show_banner(self):
        """Display the epic banner"""
        banner = """
[bold cyan]
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗  ██████╗ ██████╗ ████████╗ █████╗ ██╗          ██████╗ ██╗   ██╗███╗   ██║
║   ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗██║         ██╔════╝ ██║   ██║████╗  ██║
║   ██████╔╝██║   ██║██████╔╝   ██║   ███████║██║         ██║  ███╗██║   ██║██╔██╗ ██║
║   ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██║██║         ██║   ██║██║   ██║██║╚██╗██║
║   ██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║███████╗    ╚██████╔╝╚██████╔╝██║ ╚████║
║   ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
║                                                                              ║
║                    [bold green]Script.Library Universal Launcher v1.0[/bold green]                    ║
║                         [italic]Dimension C-137 Edition[/italic]                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
        """
        console.print(banner)
        
        # Random Rick quote
        import random
        quote = random.choice(RICK_QUOTES)
        console.print(f"\n[italic yellow]'{quote}'[/italic yellow]\n")
    
    def main_menu(self):
        """Display main menu and handle navigation"""
        while True:
            console.print("\n[bold cyan]═══ MAIN MENU ═══[/bold cyan]\n")
            
            # Create menu table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            
            table.add_row("1", "📂 Browse by Category")
            table.add_row("2", "🔍 Search Scripts")
            table.add_row("3", "🎲 Random Script (Feeling Lucky)")
            table.add_row("4", "📊 View Statistics")
            table.add_row("5", "🔧 Settings & Configuration")
            table.add_row("6", "❓ Help & Documentation")
            table.add_row("0", "🚪 Exit Portal Gun")
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold green]Enter your choice[/bold green]", choices=["0","1","2","3","4","5","6"])
            
            if choice == "1":
                self.browse_categories()
            elif choice == "2":
                self.search_scripts()
            elif choice == "3":
                self.random_script()
            elif choice == "4":
                self.show_statistics()
            elif choice == "5":
                self.settings_menu()
            elif choice == "6":
                self.show_help()
            elif choice == "0":
                if Confirm.ask("\n[yellow]Exit Portal Gun Launcher?[/yellow]"):
                    console.print("\n[green]👋 Peace out! *burp*[/green]")
                    break
    
    def browse_categories(self):
        """Browse scripts by category"""
        console.print("\n[bold cyan]═══ BROWSE BY CATEGORY ═══[/bold cyan]\n")
        
        # Display categories
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Category", style="white")
        table.add_column("Scripts", style="green")
        table.add_column("Description", style="yellow")
        
        categories = list(self.scripts.keys())
        for i, cat in enumerate(categories, 1):
            count = len(self.scripts[cat])
            emoji = CATEGORIES[cat]['emoji']
            desc = CATEGORIES[cat]['desc']
            table.add_row(str(i), f"{emoji} {cat}", str(count), desc)
        
        table.add_row("0", "↩️  Back", "", "Return to main menu")
        console.print(table)
        
        choice = Prompt.ask("\n[bold green]Select category[/bold green]")
        
        if choice == "0":
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(categories):
                self.list_category_scripts(categories[idx])
        except (ValueError, IndexError):
            console.print("[red]Invalid selection![/red]")
    
    def list_category_scripts(self, category: str):
        """List all scripts in a category"""
        scripts = self.scripts[category]
        emoji = CATEGORIES[category]['emoji']
        
        console.print(f"\n[bold cyan]═══ {emoji} {category} SCRIPTS ═══[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", no_wrap=True, width=3)
        table.add_column("Script Name", style="white", no_wrap=False)
        table.add_column("Version", style="green", width=8)
        table.add_column("Description", style="yellow")
        
        for i, script in enumerate(scripts, 1):
            name = script.get('name', script['filename'])
            version = script.get('version', 'Unknown')
            desc = script.get('description', 'No description')[:50]
            if len(script.get('description', '')) > 50:
                desc += '...'
            table.add_row(str(i), name, version, desc)
        
        table.add_row("0", "↩️  Back", "", "Return to categories")
        console.print(table)
        
        choice = Prompt.ask("\n[bold green]Select script to run (or 0 to go back)[/bold green]")
        
        if choice == "0":
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(scripts):
                self.run_script(scripts[idx])
        except (ValueError, IndexError):
            console.print("[red]Invalid selection![/red]")
    
    def run_script(self, script: Dict):
        """Execute a selected script"""
        console.print(f"\n[bold cyan]═══ LAUNCHING SCRIPT ═══[/bold cyan]\n")
        console.print(f"📜 Script: [bold]{script['filename']}[/bold]")
        console.print(f"📁 Category: {script['category']}")
        console.print(f"📄 Description: {script.get('description', 'N/A')}")
        console.print(f"🔧 Usage: {script.get('usage', 'N/A')}")
        
        if not Confirm.ask("\n[yellow]Launch this script?[/yellow]"):
            return
        
        script_path = Path(script['path'])
        
        console.print("\n[green]🚀 Initiating dimension jump...[/green]")
        console.print("[italic]Reality may shift. This is normal.[/italic]\n")
        
        try:
            if script_path.suffix == '.py':
                subprocess.run([sys.executable, str(script_path)], check=True)
            elif script_path.suffix == '.sh':
                subprocess.run(['bash', str(script_path)], check=True)
            else:
                console.print(f"[red]Unknown script type: {script_path.suffix}[/red]")
                
            console.print("\n[green]✅ Script executed successfully![/green]")
            console.print("[italic]Reality stabilized. Wubba Lubba Dub Dub![/italic]")
            
        except subprocess.CalledProcessError as e:
            console.print(f"\n[red]❌ Script failed with exit code {e.returncode}[/red]")
            console.print("[italic]Reality breach detected. Attempting repairs...[/italic]")
        except Exception as e:
            console.print(f"\n[red]❌ Error launching script: {e}[/red]")
    
    def search_scripts(self):
        """Search for scripts by name or description"""
        console.print("\n[bold cyan]═══ SEARCH SCRIPTS ═══[/bold cyan]\n")
        
        query = Prompt.ask("[bold green]Enter search term[/bold green]").lower()
        
        results = []
        for category, scripts in self.scripts.items():
            for script in scripts:
                name = script.get('name', script['filename']).lower()
                desc = script.get('description', '').lower()
                if query in name or query in desc:
                    script['_category'] = category
                    results.append(script)
        
        if not results:
            console.print(f"[yellow]No scripts found matching '{query}'[/yellow]")
            return
        
        console.print(f"\n[green]Found {len(results)} matching scripts:[/green]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Script", style="white")
        table.add_column("Category", style="green")
        table.add_column("Description", style="yellow")
        
        for i, script in enumerate(results, 1):
            name = script.get('name', script['filename'])
            cat = script['_category']
            desc = script.get('description', 'N/A')[:40] + '...'
            table.add_row(str(i), name, cat, desc)
        
        console.print(table)
        
        if Confirm.ask("\n[yellow]Run one of these scripts?[/yellow]"):
            choice = Prompt.ask("[bold green]Enter script number[/bold green]")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    self.run_script(results[idx])
            except (ValueError, IndexError):
                console.print("[red]Invalid selection![/red]")
    
    def random_script(self):
        """Select and run a random script"""
        import random
        
        console.print("\n[bold cyan]═══ RANDOM SCRIPT SELECTOR ═══[/bold cyan]")
        console.print("[italic]Let the multiverse decide your fate...[/italic]\n")
        
        # Flatten all scripts
        all_scripts = []
        for scripts in self.scripts.values():
            all_scripts.extend(scripts)
        
        if not all_scripts:
            console.print("[red]No scripts available![/red]")
            return
        
        script = random.choice(all_scripts)
        
        console.print("🎲 [bold green]The multiverse has chosen:[/bold green]\n")
        console.print(f"📜 Script: [bold]{script['filename']}[/bold]")
        console.print(f"📁 Category: {script['category']}")
        console.print(f"📄 Description: {script.get('description', 'N/A')}")
        
        if Confirm.ask("\n[yellow]Accept fate and run this script?[/yellow]"):
            self.run_script(script)
    
    def show_statistics(self):
        """Display script library statistics"""
        console.print("\n[bold cyan]═══ MULTIVERSE STATISTICS ═══[/bold cyan]\n")
        
        total_scripts = sum(len(scripts) for scripts in self.scripts.values())
        
        # Overall stats
        stats_panel = Panel(
            f"""
[bold green]Total Scripts:[/bold green] {total_scripts}
[bold blue]Categories:[/bold blue] {len(self.scripts)}
[bold yellow]Dimensions Accessible:[/bold yellow] ∞
[bold magenta]Rick Level:[/bold magenta] Maximum
[bold cyan]Reality Stability:[/bold cyan] 98.7%
            """,
            title="📊 Portal Gun Statistics",
            border_style="green"
        )
        console.print(stats_panel)
        
        # Category breakdown
        console.print("\n[bold]Category Breakdown:[/bold]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Scripts", style="green")
        table.add_column("Percentage", style="yellow")
        table.add_column("Status", style="white")
        
        for cat, scripts in sorted(self.scripts.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(scripts)
            percentage = (count / total_scripts) * 100
            emoji = CATEGORIES[cat]['emoji']
            
            if count > 20:
                status = "🔥 Hot"
            elif count > 10:
                status = "✨ Active"
            elif count > 5:
                status = "📈 Growing"
            else:
                status = "🌱 Emerging"
            
            table.add_row(
                f"{emoji} {cat}",
                str(count),
                f"{percentage:.1f}%",
                status
            )
        
        console.print(table)
        
        input("\nPress Enter to continue...")
    
    def settings_menu(self):
        """Settings and configuration menu"""
        console.print("\n[bold cyan]═══ SETTINGS & CONFIGURATION ═══[/bold cyan]\n")
        
        settings = Panel(
            """
[bold]Current Configuration:[/bold]

📁 Script Directory: /Volumes/Development/Github/Script.Library
📊 Log Directory: ~/Desktop/ScriptLibrary_Logs
🌍 Active Dimension: C-137 (Production)
🔧 Auto-update: Disabled
🎨 Theme: Portal Green
🔊 Verbose Mode: Disabled
⚡ Turbo Mode: Disabled

[italic]Settings are configured in portal_gun_config.json[/italic]
            """,
            title="⚙️ Portal Gun Settings",
            border_style="yellow"
        )
        console.print(settings)
        
        input("\nPress Enter to continue...")
    
    def show_help(self):
        """Display help and documentation"""
        console.print("\n[bold cyan]═══ HELP & DOCUMENTATION ═══[/bold cyan]\n")
        
        help_text = """
[bold]Portal Gun Launcher - Quick Guide[/bold]

[bold green]Navigation:[/bold green]
• Use number keys to select menu options
• Type search terms when prompted
• Press Enter to confirm selections
• Use 0 to go back in most menus

[bold yellow]Script Execution:[/bold yellow]
• Scripts run in their own process
• Output is displayed in real-time
• Logs are saved to ~/Desktop/ScriptLibrary_Logs
• Press Ctrl+C to abort a running script

[bold cyan]Categories:[/bold cyan]
• Each category contains specialized scripts
• Scripts are auto-discovered on launch
• Metadata is extracted from script headers

[bold magenta]Tips:[/bold magenta]
• Try the random script selector for surprises!
• Search works on both names and descriptions
• Check statistics to see most active categories
• Scripts marked "Portal Gun Compatible" have enhanced features

[bold red]Troubleshooting:[/bold red]
• If a script fails, check the logs
• Ensure all dependencies are installed
• Some scripts require sudo access
• Reality breaches are usually temporary

[italic]"Nobody exists on purpose. Nobody belongs anywhere. 
We're all going to die. Come use these scripts."[/italic]
        """
        
        console.print(help_text)
        input("\nPress Enter to continue...")

# ============================================================================
# GUI INTERFACE
# ============================================================================

class PortalGunGUI:
    """CustomTkinter GUI for the Portal Gun Launcher"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            console.print("[red]GUI dependencies not available. Installing...[/red]")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter", "Pillow"])
            console.print("[green]GUI dependencies installed! Please restart the launcher.[/green]")
            sys.exit(0)
            
        self.discovery = ScriptDiscovery()
        self.scripts = {}
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the GUI"""
        # Configure CustomTkinter
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        
        # Create main window
        self.root = customtkinter.CTk()
        self.root.title("Portal Gun Launcher - Script.Library Control Panel")
        self.root.geometry("1400x900")
        
        # Set window icon (if available)
        # self.root.iconbitmap('portal_gun.ico')
        
        # Create main container
        self.create_widgets()
        
        # Load scripts
        self.load_scripts()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Header Frame
        header_frame = customtkinter.CTkFrame(self.root, height=100, corner_radius=0)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = customtkinter.CTkLabel(
            header_frame,
            text="🧬 PORTAL GUN LAUNCHER",
            font=("Courier", 36, "bold")
        )
        title_label.pack(pady=20)
        
        subtitle_label = customtkinter.CTkLabel(
            header_frame,
            text="Script.Library Universal Control Panel - Dimension C-137",
            font=("Courier", 14)
        )
        subtitle_label.pack()
        
        # Main container with sidebar
        main_container = customtkinter.CTkFrame(self.root, corner_radius=0)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Sidebar
        self.create_sidebar(main_container)
        
        # Content area
        self.content_frame = customtkinter.CTkFrame(main_container)
        self.content_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        
        # Initial content
        self.show_welcome()
        
    def create_sidebar(self, parent):
        """Create the sidebar with categories"""
        sidebar = customtkinter.CTkFrame(parent, width=250)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Search box
        search_label = customtkinter.CTkLabel(sidebar, text="🔍 Search Scripts", font=("Courier", 14, "bold"))
        search_label.pack(pady=(20, 5))
        
        self.search_entry = customtkinter.CTkEntry(sidebar, placeholder_text="Enter search term...")
        self.search_entry.pack(padx=20, pady=(0, 10), fill="x")
        self.search_entry.bind("<Return>", lambda e: self.search_scripts())
        
        search_btn = customtkinter.CTkButton(
            sidebar,
            text="Search",
            command=self.search_scripts,
            fg_color="#00FF41",
            text_color="black",
            hover_color="#39FF14"
        )
        search_btn.pack(padx=20, pady=(0, 20))
        
        # Categories
        categories_label = customtkinter.CTkLabel(sidebar, text="📂 Categories", font=("Courier", 14, "bold"))
        categories_label.pack(pady=(10, 10))
        
        # Category buttons
        for category, info in CATEGORIES.items():
            btn = customtkinter.CTkButton(
                sidebar,
                text=f"{info['emoji']} {category}",
                command=lambda c=category: self.show_category(c),
                fg_color=info['color'],
                text_color="white",
                hover_color=info['color'],
                height=35,
                font=("Courier", 12)
            )
            btn.pack(padx=20, pady=5, fill="x")
        
        # Bottom buttons
        stats_btn = customtkinter.CTkButton(
            sidebar,
            text="📊 Statistics",
            command=self.show_statistics,
            fg_color="#6A0DAD",
            hover_color="#8B008B"
        )
        stats_btn.pack(side="bottom", padx=20, pady=10, fill="x")
        
        settings_btn = customtkinter.CTkButton(
            sidebar,
            text="⚙️ Settings",
            command=self.show_settings,
            fg_color="#708090",
            hover_color="#778899"
        )
        settings_btn.pack(side="bottom", padx=20, pady=(0, 10), fill="x")
        
    def show_welcome(self):
        """Display welcome screen"""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        welcome_frame = customtkinter.CTkFrame(self.content_frame)
        welcome_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ASCII art (simplified for GUI)
        ascii_art = customtkinter.CTkTextbox(welcome_frame, height=200, font=("Courier", 10))
        ascii_art.pack(pady=20, padx=20, fill="x")
        ascii_art.insert("1.0", """
    ██████╗  ██████╗ ██████╗ ████████╗ █████╗ ██╗      
    ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗██║      
    ██████╔╝██║   ██║██████╔╝   ██║   ███████║██║      
    ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██║██║      
    ██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║███████╗ 
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ 
                     GUN LAUNCHER v1.0
        """)
        ascii_art.configure(state="disabled")
        
        # Welcome message
        import random
        quote = random.choice(RICK_QUOTES)
        
        welcome_label = customtkinter.CTkLabel(
            welcome_frame,
            text=f"Welcome to the Script Multiverse!",
            font=("Courier", 24, "bold")
        )
        welcome_label.pack(pady=20)
        
        quote_label = customtkinter.CTkLabel(
            welcome_frame,
            text=f'"{quote}"',
            font=("Courier", 14, "italic"),
            text_color="#00FF41"
        )
        quote_label.pack(pady=10)
        
        # Quick stats
        if self.scripts:
            total = sum(len(scripts) for scripts in self.scripts.values())
            stats_label = customtkinter.CTkLabel(
                welcome_frame,
                text=f"🧬 {total} Scripts Available Across {len(self.scripts)} Categories",
                font=("Courier", 16)
            )
            stats_label.pack(pady=20)
        
        # Quick actions
        actions_frame = customtkinter.CTkFrame(welcome_frame)
        actions_frame.pack(pady=30)
        
        random_btn = customtkinter.CTkButton(
            actions_frame,
            text="🎲 Random Script",
            command=self.run_random_script,
            width=200,
            height=50,
            font=("Courier", 14),
            fg_color="#FF006E",
            hover_color="#FF1493"
        )
        random_btn.pack(side="left", padx=10)
        
        recent_btn = customtkinter.CTkButton(
            actions_frame,
            text="🕐 Recent Scripts",
            command=self.show_recent,
            width=200,
            height=50,
            font=("Courier", 14),
            fg_color="#00B4D8",
            hover_color="#00CED1"
        )
        recent_btn.pack(side="left", padx=10)
        
    def show_category(self, category):
        """Display scripts in a category"""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        if category not in self.scripts:
            error_label = customtkinter.CTkLabel(
                self.content_frame,
                text=f"No scripts found in {category}",
                font=("Courier", 16)
            )
            error_label.pack(pady=50)
            return
        
        # Category header
        header_frame = customtkinter.CTkFrame(self.content_frame)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        emoji = CATEGORIES[category]['emoji']
        title = customtkinter.CTkLabel(
            header_frame,
            text=f"{emoji} {category} Scripts",
            font=("Courier", 24, "bold")
        )
        title.pack(side="left")
        
        count_label = customtkinter.CTkLabel(
            header_frame,
            text=f"({len(self.scripts[category])} scripts)",
            font=("Courier", 14),
            text_color="#00FF41"
        )
        count_label.pack(side="left", padx=20)
        
        # Scrollable frame for scripts
        scroll_frame = customtkinter.CTkScrollableFrame(self.content_frame)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Display scripts as cards
        for script in self.scripts[category]:
            self.create_script_card(scroll_frame, script)
    
    def create_script_card(self, parent, script):
        """Create a card widget for a script"""
        card = customtkinter.CTkFrame(parent, height=120)
        card.pack(fill="x", padx=10, pady=5)
        card.pack_propagate(False)
        
        # Script info
        info_frame = customtkinter.CTkFrame(card)
        info_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        name_label = customtkinter.CTkLabel(
            info_frame,
            text=script.get('name', script['filename']),
            font=("Courier", 16, "bold"),
            anchor="w"
        )
        name_label.pack(fill="x")
        
        desc_label = customtkinter.CTkLabel(
            info_frame,
            text=script.get('description', 'No description available'),
            font=("Courier", 11),
            anchor="w",
            text_color="#8B8C9A"
        )
        desc_label.pack(fill="x", pady=(5, 0))
        
        version_label = customtkinter.CTkLabel(
            info_frame,
            text=f"Version: {script.get('version', 'Unknown')} | {script.get('author', 'Unknown')}",
            font=("Courier", 10),
            anchor="w",
            text_color="#708090"
        )
        version_label.pack(fill="x", pady=(5, 0))
        
        # Action buttons
        btn_frame = customtkinter.CTkFrame(card)
        btn_frame.pack(side="right", padx=10, pady=10)
        
        run_btn = customtkinter.CTkButton(
            btn_frame,
            text="▶️ Run",
            command=lambda s=script: self.run_script(s),
            width=80,
            fg_color="#00FF41",
            text_color="black",
            hover_color="#39FF14"
        )
        run_btn.pack(pady=5)
        
        info_btn = customtkinter.CTkButton(
            btn_frame,
            text="ℹ️ Info",
            command=lambda s=script: self.show_script_info(s),
            width=80,
            fg_color="#708090",
            hover_color="#778899"
        )
        info_btn.pack(pady=5)
    
    def run_script(self, script):
        """Execute a script"""
        # Create execution window
        exec_window = customtkinter.CTkToplevel(self.root)
        exec_window.title(f"Executing: {script['filename']}")
        exec_window.geometry("800x600")
        
        # Header
        header = customtkinter.CTkLabel(
            exec_window,
            text=f"🚀 Executing: {script.get('name', script['filename'])}",
            font=("Courier", 18, "bold")
        )
        header.pack(pady=10)
        
        # Output text area
        output_text = customtkinter.CTkTextbox(exec_window, font=("Courier", 11))
        output_text.pack(fill="both", expand=True, padx=20, pady=10)
        
        output_text.insert("1.0", "🧬 Initializing dimension jump...\n")
        output_text.insert("end", f"📜 Script: {script['filename']}\n")
        output_text.insert("end", f"📁 Category: {script['category']}\n")
        output_text.insert("end", "━" * 50 + "\n\n")
        
        # Control buttons
        btn_frame = customtkinter.CTkFrame(exec_window)
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        close_btn = customtkinter.CTkButton(
            btn_frame,
            text="Close",
            command=exec_window.destroy,
            fg_color="#FF006E",
            hover_color="#FF1493"
        )
        close_btn.pack(side="right")
        
        # Execute script in thread to not block GUI
        import threading
        
        def execute():
            script_path = Path(script['path'])
            try:
                output_text.insert("end", "🔄 Executing script...\n\n")
                
                if script_path.suffix == '.py':
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        capture_output=True,
                        text=True
                    )
                elif script_path.suffix == '.sh':
                    result = subprocess.run(
                        ['bash', str(script_path)],
                        capture_output=True,
                        text=True
                    )
                else:
                    output_text.insert("end", f"❌ Unknown script type: {script_path.suffix}\n")
                    return
                
                if result.stdout:
                    output_text.insert("end", result.stdout)
                if result.stderr:
                    output_text.insert("end", f"\n⚠️ Errors:\n{result.stderr}")
                
                if result.returncode == 0:
                    output_text.insert("end", "\n\n✅ Script executed successfully!")
                    output_text.insert("end", "\n🎉 Wubba Lubba Dub Dub!")
                else:
                    output_text.insert("end", f"\n\n❌ Script failed with exit code {result.returncode}")
                    
            except Exception as e:
                output_text.insert("end", f"\n\n❌ Error: {str(e)}")
        
        thread = threading.Thread(target=execute)
        thread.start()
    
    def show_script_info(self, script):
        """Display detailed script information"""
        info_window = customtkinter.CTkToplevel(self.root)
        info_window.title(f"Script Info: {script['filename']}")
        info_window.geometry("600x500")
        
        # Script details
        details_frame = customtkinter.CTkFrame(info_window)
        details_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        info_text = customtkinter.CTkTextbox(details_frame, font=("Courier", 12))
        info_text.pack(fill="both", expand=True)
        
        # Format script information
        info = f"""
{'='*50}
SCRIPT INFORMATION
{'='*50}

📜 Filename: {script['filename']}
📁 Category: {script['category']}
📍 Path: {script['path']}
📝 Extension: {script['extension']}

🏷️ Name: {script.get('name', 'N/A')}
🔢 Version: {script.get('version', 'Unknown')}
👤 Author: {script.get('author', 'Unknown')}

📄 Description:
{script.get('description', 'No description available')}

🔧 Usage:
{script.get('usage', 'See script for usage information')}

{'='*50}
"""
        
        # Check for metadata file
        metadata_file = Path(script['path']).with_suffix('.metadata.json')
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                info += "\n📊 METADATA:\n"
                info += json.dumps(metadata, indent=2)
            except:
                pass
        
        info_text.insert("1.0", info)
        info_text.configure(state="disabled")
        
        # Close button
        close_btn = customtkinter.CTkButton(
            info_window,
            text="Close",
            command=info_window.destroy
        )
        close_btn.pack(pady=10)
    
    def search_scripts(self):
        """Search for scripts"""
        query = self.search_entry.get().lower()
        if not query:
            return
        
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Search header
        header_label = customtkinter.CTkLabel(
            self.content_frame,
            text=f"🔍 Search Results for: '{query}'",
            font=("Courier", 20, "bold")
        )
        header_label.pack(pady=20)
        
        # Find matching scripts
        results = []
        for category, scripts in self.scripts.items():
            for script in scripts:
                name = script.get('name', script['filename']).lower()
                desc = script.get('description', '').lower()
                if query in name or query in desc:
                    script_copy = script.copy()
                    script_copy['category'] = category
                    results.append(script_copy)
        
        if not results:
            no_results = customtkinter.CTkLabel(
                self.content_frame,
                text="No scripts found matching your search.",
                font=("Courier", 14),
                text_color="#FF006E"
            )
            no_results.pack(pady=50)
            return
        
        # Results count
        count_label = customtkinter.CTkLabel(
            self.content_frame,
            text=f"Found {len(results)} matching scripts",
            font=("Courier", 14),
            text_color="#00FF41"
        )
        count_label.pack(pady=(0, 20))
        
        # Scrollable results
        scroll_frame = customtkinter.CTkScrollableFrame(self.content_frame)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        for script in results:
            self.create_script_card(scroll_frame, script)
    
    def show_statistics(self):
        """Display statistics window"""
        stats_window = customtkinter.CTkToplevel(self.root)
        stats_window.title("Portal Gun Statistics")
        stats_window.geometry("700x600")
        
        # Header
        header = customtkinter.CTkLabel(
            stats_window,
            text="📊 Multiverse Statistics",
            font=("Courier", 24, "bold")
        )
        header.pack(pady=20)
        
        # Stats frame
        stats_frame = customtkinter.CTkFrame(stats_window)
        stats_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Calculate statistics
        total_scripts = sum(len(scripts) for scripts in self.scripts.values())
        
        # Overall stats
        overall_frame = customtkinter.CTkFrame(stats_frame)
        overall_frame.pack(fill="x", padx=20, pady=20)
        
        stats_text = f"""
Total Scripts: {total_scripts}
Categories: {len(self.scripts)}
Python Scripts: {sum(1 for cat in self.scripts.values() for s in cat if s['extension'] == '.py')}
Shell Scripts: {sum(1 for cat in self.scripts.values() for s in cat if s['extension'] == '.sh')}
Dimensions Accessible: ∞
Rick Level: Maximum
Reality Stability: 98.7%
        """
        
        overall_label = customtkinter.CTkLabel(
            overall_frame,
            text=stats_text,
            font=("Courier", 14),
            justify="left"
        )
        overall_label.pack()
        
        # Category breakdown
        breakdown_label = customtkinter.CTkLabel(
            stats_frame,
            text="Category Breakdown:",
            font=("Courier", 18, "bold")
        )
        breakdown_label.pack(pady=(20, 10))
        
        # Create breakdown list
        breakdown_frame = customtkinter.CTkScrollableFrame(stats_frame, height=250)
        breakdown_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        for cat, scripts in sorted(self.scripts.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(scripts)
            percentage = (count / total_scripts * 100) if total_scripts > 0 else 0
            emoji = CATEGORIES[cat]['emoji']
            
            row = customtkinter.CTkFrame(breakdown_frame)
            row.pack(fill="x", pady=2)
            
            cat_label = customtkinter.CTkLabel(
                row,
                text=f"{emoji} {cat}:",
                font=("Courier", 12),
                width=200,
                anchor="w"
            )
            cat_label.pack(side="left", padx=10)
            
            count_label = customtkinter.CTkLabel(
                row,
                text=f"{count} scripts ({percentage:.1f}%)",
                font=("Courier", 12),
                text_color="#00FF41"
            )
            count_label.pack(side="left")
        
        # Close button
        close_btn = customtkinter.CTkButton(
            stats_window,
            text="Close",
            command=stats_window.destroy
        )
        close_btn.pack(pady=10)
    
    def show_settings(self):
        """Display settings window"""
        settings_window = customtkinter.CTkToplevel(self.root)
        settings_window.title("Portal Gun Settings")
        settings_window.geometry("600x500")
        
        # Header
        header = customtkinter.CTkLabel(
            settings_window,
            text="⚙️ Settings & Configuration",
            font=("Courier", 24, "bold")
        )
        header.pack(pady=20)
        
        # Settings frame
        settings_frame = customtkinter.CTkFrame(settings_window)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Current configuration
        config_text = f"""
Current Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Script Directory: 
   {SCRIPT_DIR}

📊 Log Directory:
   {LOG_DIR}

🌍 Active Dimension: C-137 (Production)

🔧 Auto-update: Disabled

🎨 Theme: Portal Green

🔊 Verbose Mode: Disabled

⚡ Turbo Mode: Disabled

🧬 Portal Gun Status: ACTIVE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Note: Settings are configured in portal_gun_config.json
        """
        
        config_label = customtkinter.CTkLabel(
            settings_frame,
            text=config_text,
            font=("Courier", 12),
            justify="left"
        )
        config_label.pack(padx=20, pady=20)
        
        # Action buttons
        btn_frame = customtkinter.CTkFrame(settings_frame)
        btn_frame.pack(pady=20)
        
        reload_btn = customtkinter.CTkButton(
            btn_frame,
            text="🔄 Reload Scripts",
            command=self.load_scripts,
            width=150
        )
        reload_btn.pack(side="left", padx=10)
        
        logs_btn = customtkinter.CTkButton(
            btn_frame,
            text="📁 Open Logs",
            command=lambda: subprocess.run(['open', str(LOG_DIR)]),
            width=150
        )
        logs_btn.pack(side="left", padx=10)
        
        # Close button
        close_btn = customtkinter.CTkButton(
            settings_window,
            text="Close",
            command=settings_window.destroy
        )
        close_btn.pack(pady=10)
    
    def show_recent(self):
        """Show recently used scripts (placeholder)"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        label = customtkinter.CTkLabel(
            self.content_frame,
            text="🕐 Recent Scripts (Coming Soon)",
            font=("Courier", 24, "bold")
        )
        label.pack(pady=50)
        
        info = customtkinter.CTkLabel(
            self.content_frame,
            text="This feature will track your recently used scripts across dimensions.",
            font=("Courier", 14),
            text_color="#8B8C9A"
        )
        info.pack()
    
    def run_random_script(self):
        """Select and run a random script"""
        import random
        
        all_scripts = []
        for scripts in self.scripts.values():
            all_scripts.extend(scripts)
        
        if not all_scripts:
            return
        
        script = random.choice(all_scripts)
        
        # Confirmation dialog
        result = messagebox.askyesno(
            "Random Script Selected",
            f"The multiverse has chosen:\n\n{script.get('name', script['filename'])}\n\nRun this script?"
        )
        
        if result:
            self.run_script(script)
    
    def load_scripts(self):
        """Load all scripts"""
        self.scripts = self.discovery.discover_all()
        self.show_welcome()  # Refresh welcome screen with new stats
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Portal Gun Launcher"""
    parser = argparse.ArgumentParser(
        description="🧬 Portal Gun Launcher - Script.Library Universal Interface",
        epilog="*burp* That's all folks! - Rick C-137"
    )
    
    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch GUI interface (requires CustomTkinter)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available scripts'
    )
    
    parser.add_argument(
        '--category', '-c',
        type=str,
        help='Filter by category'
    )
    
    parser.add_argument(
        '--search', '-s',
        type=str,
        help='Search for scripts'
    )
    
    parser.add_argument(
        '--dimension',
        default='C-137',
        help='Target dimension (default: C-137 - Production)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable Rick-level verbosity'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_portal_gun_logs()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Quick operations
    if args.list or args.search or args.category:
        discovery = ScriptDiscovery()
        scripts = discovery.discover_all()
        
        if args.list:
            console.print("\n[bold cyan]Available Scripts:[/bold cyan]\n")
            for cat, scripts_list in scripts.items():
                console.print(f"{CATEGORIES[cat]['emoji']} [bold]{cat}[/bold]: {len(scripts_list)} scripts")
                if args.verbose:
                    for script in scripts_list:
                        console.print(f"  - {script['filename']}")
        
        elif args.search:
            results = []
            for cat, scripts_list in scripts.items():
                for script in scripts_list:
                    if args.search.lower() in script.get('name', '').lower() or \
                       args.search.lower() in script.get('description', '').lower():
                        results.append((cat, script))
            
            if results:
                console.print(f"\n[green]Found {len(results)} matches:[/green]\n")
                for cat, script in results:
                    console.print(f"{CATEGORIES[cat]['emoji']} [{cat}] {script['filename']}")
            else:
                console.print(f"[yellow]No scripts found matching '{args.search}'[/yellow]")
        
        elif args.category:
            if args.category in scripts:
                console.print(f"\n{CATEGORIES[args.category]['emoji']} [bold]{args.category} Scripts:[/bold]\n")
                for script in scripts[args.category]:
                    console.print(f"  - {script['filename']}: {script.get('description', 'N/A')}")
            else:
                console.print(f"[red]Category '{args.category}' not found[/red]")
        
        return
    
    # Launch appropriate interface
    if args.gui:
        if not GUI_AVAILABLE:
            console.print("[yellow]GUI dependencies not installed. Installing now...[/yellow]")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter", "Pillow"])
            console.print("[green]Dependencies installed! Launching GUI...[/green]")
        
        app = PortalGunGUI()
        app.run()
    else:
        # Default to CLI
        cli = PortalGunCLI()
        cli.run()

if __name__ == "__main__":
    main()