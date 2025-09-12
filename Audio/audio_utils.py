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
# Script Name: audio_utils.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive utilities library for audio processing scripts      
#              including dependency management, file operations, API helpers,    
#              and specialized audio processing functions.                       
#
# Usage: Import as module: from audio_utils import *
#
# Dependencies: tkinter, requests, pathlib, sqlite3, hashlib, mutagen             
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Provides universal macOS compatibility with desktop logging and         
#        comprehensive error handling for all audio-related operations.          
#                                                                                
####################################################################################

"""
Audio Scripts Utilities Library
==============================

Comprehensive utilities for audio processing scripts including:
- Dependency management and auto-installation
- Logging setup with desktop file placement
- File and folder selection dialogs (macOS native)
- Multi-threading and parallel processing
- API credential management and storage
- Progress tracking with GUI indicators
- Audio file format detection and validation
- Metadata extraction and manipulation
- Spotify API integration helpers
- Apple Podcasts database utilities
- ElevenLabs API utilities
- Audio processing and conversion helpers

Technical Features:
- Universal macOS path compatibility
- Comprehensive error handling
- Thread-safe operations
- Memory-efficient processing
- GUI dialog integration
- Configuration persistence

Author: sanchez314c@speedheathens.com
Version: 1.0.0
License: MIT
"""

import sys
import os
import subprocess
import importlib
import logging
import argparse
import json
import multiprocessing
import threading
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import time


def check_and_install_dependencies(required_packages: List[str], 
                                  auto_install: bool = False) -> bool:
    """
    Check if required packages are installed and optionally install them.
    
    Args:
        required_packages: List of package names to check
        auto_install: If True, automatically install missing packages without prompting
    
    Returns:
        bool: True if all dependencies are met, False otherwise
    """
    missing_packages = []
    
    for package in required_packages:
        # Strip any version specifiers for the import check
        import_name = package.split('>=')[0].split('==')[0].strip()
        
        # Skip tkinter as it's a special case (built into Python)
        if import_name.lower() == 'tkinter':
            try:
                importlib.import_module('tkinter')
            except ImportError:
                missing_packages.append(package)
            continue
        
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if not missing_packages:
        return True
    
    print(f"Missing required packages: {', '.join(missing_packages)}")
    
    if auto_install or input("Install missing packages? (y/n): ").lower() == 'y':
        try:
            # Use Python executable that ran this script
            python = sys.executable
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call(
                    [python, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL
                )
            print("All dependencies installed successfully.")
            return True
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return False
    else:
        print("Please install required packages to continue.")
        return False


def setup_logging(script_path: Optional[Path] = None) -> Path:
    """
    Initialize logging with console and file handlers.
    
    Args:
        script_path: Path to the script file. If None, uses calling script's path.
    
    Returns:
        Path: Path to the log file
    """
    if script_path is None:
        # Get the calling script's filename
        import inspect
        frame = inspect.stack()[1]
        script_path = Path(frame.filename)
    
    log_file = script_path.with_suffix('.log')
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    
    return log_file


def setup_argument_parser(description: str, epilog: str = "", 
                          args_config: Optional[List[Dict[str, Any]]] = None) -> argparse.ArgumentParser:
    """
    Creates a standardized argument parser for all audio scripts.
    
    Args:
        description: Description of the script
        epilog: Additional text after the help text
        args_config: List of dictionaries specifying custom arguments
                    Each dict should include keys like 'flags', 'help', etc.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add standard arguments all scripts will have
    parser.add_argument('--auto-install', action='store_true',
                        help='Automatically install missing dependencies without prompting')
    parser.add_argument('--version', action='store_true',
                        help='Print version information and exit')
    
    # Add custom arguments
    if args_config:
        for arg_dict in args_config:
            flags = arg_dict.pop('flags')
            parser.add_argument(*flags, **arg_dict)
    
    return parser


def print_script_info(name: str, version: str, author: str, date: str) -> None:
    """
    Prints standardized script information.
    
    Args:
        name: Script name
        version: Version string
        author: Author name
        date: Release date
    """
    print(f"{name} v{version}")
    print(f"By {author} ({date})")
    print("-" * 40)


# File/Folder Selection Dialogs
def select_file(title: str = "Select File", 
                file_types: List[Tuple[str, str]] = None,
                initial_dir: Optional[str] = None) -> Optional[Path]:
    """
    Uses macOS native file dialog to select a file.
    
    Args:
        title: Dialog window title
        file_types: List of tuples with file type descriptions and patterns
        initial_dir: Initial directory to open dialog in
        
    Returns:
        Path object for selected file or None if canceled
    """
    if file_types is None:
        file_types = [("All Files", "*.*")]
    
    root = tk.Tk()
    root.withdraw()
    
    # Make sure the dialog appears on top
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=file_types,
        initialdir=initial_dir
    )
    
    root.destroy()
    return Path(file_path) if file_path else None


def select_files(title: str = "Select Files", 
                file_types: List[Tuple[str, str]] = None,
                initial_dir: Optional[str] = None) -> List[Path]:
    """
    Uses macOS native file dialog to select multiple files.
    
    Args:
        title: Dialog window title
        file_types: List of tuples with file type descriptions and patterns
        initial_dir: Initial directory to open dialog in
        
    Returns:
        List of Path objects for selected files or empty list if canceled
    """
    if file_types is None:
        file_types = [("All Files", "*.*")]
    
    root = tk.Tk()
    root.withdraw()
    
    # Make sure the dialog appears on top
    root.attributes("-topmost", True)
    
    file_paths = filedialog.askopenfilenames(
        title=title,
        filetypes=file_types,
        initialdir=initial_dir
    )
    
    root.destroy()
    return [Path(file_path) for file_path in file_paths] if file_paths else []


def select_directory(title: str = "Select Directory",
                    initial_dir: Optional[str] = None) -> Optional[Path]:
    """
    Uses macOS native file dialog to select a directory.
    
    Args:
        title: Dialog window title
        initial_dir: Initial directory to open dialog in
        
    Returns:
        Path object for selected directory or None if canceled
    """
    root = tk.Tk()
    root.withdraw()
    
    # Make sure the dialog appears on top
    root.attributes("-topmost", True)
    
    directory = filedialog.askdirectory(
        title=title,
        initialdir=initial_dir
    )
    
    root.destroy()
    return Path(directory) if directory else None


def select_save_file(title: str = "Save As", 
                    file_types: List[Tuple[str, str]] = None,
                    initial_dir: Optional[str] = None,
                    default_extension: Optional[str] = None,
                    default_name: Optional[str] = None) -> Optional[Path]:
    """
    Uses macOS native file dialog to select a save location.
    
    Args:
        title: Dialog window title
        file_types: List of tuples with file type descriptions and patterns
        initial_dir: Initial directory to open dialog in
        default_extension: Default file extension
        default_name: Default filename
        
    Returns:
        Path object for save location or None if canceled
    """
    if file_types is None:
        file_types = [("All Files", "*.*")]
    
    root = tk.Tk()
    root.withdraw()
    
    # Make sure the dialog appears on top
    root.attributes("-topmost", True)
    
    initialfile = default_name if default_name else None
    
    file_path = filedialog.asksaveasfilename(
        title=title,
        filetypes=file_types,
        initialdir=initial_dir,
        defaultextension=default_extension,
        initialfile=initialfile
    )
    
    root.destroy()
    return Path(file_path) if file_path else None


# Credential Handling
class CredentialManager:
    """Handles secure storage and retrieval of API credentials"""
    
    @staticmethod
    def get_config_path(app_name: str) -> Path:
        """Get path to the config file for a specific application"""
        return Path.home() / f".{app_name.lower().replace(' ', '_')}_config.json"
    
    @staticmethod
    def load_credentials(app_name: str) -> Dict[str, Any]:
        """Load credentials from config file"""
        config_path = CredentialManager.get_config_path(app_name)
        if not config_path.exists():
            return {}
            
        try:
            with open(config_path, 'r') as f:
                return json.loads(f.read())
        except Exception as e:
            logging.error(f"Error loading credentials: {e}")
            return {}
    
    @staticmethod
    def save_credentials(app_name: str, credentials: Dict[str, Any]) -> bool:
        """Save credentials to config file"""
        config_path = CredentialManager.get_config_path(app_name)
        try:
            with open(config_path, 'w') as f:
                f.write(json.dumps(credentials, indent=2))
            return True
        except Exception as e:
            logging.error(f"Error saving credentials: {e}")
            return False
    
    @staticmethod
    def get_credential(app_name: str, key: str, 
                      prompt: str = None, 
                      password: bool = False) -> Optional[str]:
        """
        Get a credential, prompting the user if not found
        
        Args:
            app_name: Application name for credential storage
            key: Credential key
            prompt: Prompt to show when requesting credential
            password: Whether to hide input (for passwords, tokens, etc)
            
        Returns:
            The credential value or None if canceled
        """
        # Try to load from config first
        credentials = CredentialManager.load_credentials(app_name)
        
        if key in credentials and credentials[key]:
            return credentials[key]
        
        # If not found, prompt user
        if prompt is None:
            prompt = f"Enter {key} for {app_name}"
        
        root = tk.Tk()
        root.withdraw()
        
        # Make sure the dialog appears on top
        root.attributes("-topmost", True)
        
        if password:
            value = simpledialog.askstring(
                f"{app_name} Setup", 
                prompt,
                parent=root,
                show='*'
            )
        else:
            value = simpledialog.askstring(
                f"{app_name} Setup", 
                prompt,
                parent=root
            )
        
        root.destroy()
        
        if value:
            # Save to config
            credentials[key] = value
            CredentialManager.save_credentials(app_name, credentials)
            return value
        
        return None


# Multi-Threading Configuration
def get_optimal_workers(cpu_bound: bool = True) -> int:
    """
    Determine optimal number of worker threads/processes
    
    Args:
        cpu_bound: Whether the task is CPU-bound (True) or IO-bound (False)
        
    Returns:
        Number of worker threads/processes to use
    """
    cpu_count = multiprocessing.cpu_count()
    
    if cpu_bound:
        # For CPU-bound tasks, use N-1 cores (leave one for system)
        return max(1, cpu_count - 1)
    else:
        # For IO-bound tasks, can use more workers
        return cpu_count * 2


class ParallelExecutor:
    """Helper for parallel execution with appropriate executor type"""
    
    def __init__(self, cpu_bound: bool = True, max_workers: Optional[int] = None):
        """
        Initialize parallel executor
        
        Args:
            cpu_bound: Whether task is CPU-bound (True) or IO-bound (False)
            max_workers: Maximum number of workers, or None to use optimal
        """
        self.cpu_bound = cpu_bound
        self.max_workers = max_workers or get_optimal_workers(cpu_bound)
    
    def __enter__(self):
        """Create and return the appropriate executor type"""
        if self.cpu_bound:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self.executor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the executor"""
        self.executor.shutdown()


# Progress Tracking
class ProgressTracker:
    """Manages progress UI and tracking for long-running operations"""
    
    def __init__(self, title: str = "Processing", max_value: int = 100):
        """
        Initialize progress tracker
        
        Args:
            title: Window title
            max_value: Maximum progress value
        """
        self.title = title
        self.max_value = max_value
        self.current = 0
        self.window = None
        self.progress_var = None
        self.progress_bar = None
        self.status_label = None
        self.start_time = None
        self.active = False
    
    def _setup_window(self):
        """Set up the progress window"""
        self.window = tk.Toplevel()
        self.window.title(self.title)
        self.window.geometry("400x150")
        self.window.resizable(False, False)
        
        # Center on screen
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Make sure it's on top
        self.window.attributes("-topmost", True)
        
        # Progress variable and bar
        self.progress_var = tk.DoubleVar(value=0)
        
        # Status label
        self.status_label = ttk.Label(
            self.window, 
            text="Initializing...",
            font=("Helvetica", 12)
        )
        self.status_label.pack(pady=(20, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.window,
            orient="horizontal",
            length=350,
            mode="determinate",
            variable=self.progress_var,
            maximum=self.max_value
        )
        self.progress_bar.pack(pady=10, padx=25)
        
        # Time label
        self.time_label = ttk.Label(
            self.window, 
            text="Time elapsed: 0s",
            font=("Helvetica", 10)
        )
        self.time_label.pack(pady=5)
        
        # Update the window
        self.window.update()
    
    def start(self):
        """Start progress tracking"""
        if self.active:
            return
        
        self.active = True
        self.start_time = time.time()
        self._setup_window()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_time_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _update_time_loop(self):
        """Update the elapsed time display"""
        while self.active and self.window:
            try:
                elapsed = int(time.time() - self.start_time)
                self.time_label.config(text=f"Time elapsed: {elapsed}s")
                time.sleep(1)
                
                # Process events to keep UI responsive
                if self.window:
                    self.window.update()
            except:
                # Window might have been destroyed
                break
    
    def update(self, value: Optional[int] = None, 
              increment: int = 1, 
              status: Optional[str] = None):
        """
        Update progress
        
        Args:
            value: Set progress to this value
            increment: Increment progress by this amount
            status: Update status text
        """
        if not self.active or not self.window:
            self.start()
        
        if value is not None:
            self.current = value
        else:
            self.current += increment
        
        # Update progress bar
        self.progress_var.set(min(self.current, self.max_value))
        
        # Update status if provided
        if status:
            self.status_label.config(text=status)
        
        # Update percentage in window title
        percentage = int((self.current / self.max_value) * 100)
        self.window.title(f"{self.title} - {percentage}%")
        
        # Process events to update UI
        self.window.update()
    
    def finish(self, message: str = "Operation completed successfully", 
              show_time: bool = True, auto_close: bool = True):
        """
        Finish progress tracking
        
        Args:
            message: Completion message
            show_time: Whether to show total time in message
            auto_close: Whether to auto-close after delay
        """
        if not self.active:
            return
        
        # Set progress to 100%
        self.progress_var.set(self.max_value)
        
        # Update status
        if show_time:
            elapsed = int(time.time() - self.start_time)
            message = f"{message} (Total time: {elapsed}s)"
        
        self.status_label.config(text=message)
        
        # Change progress bar style to indicate completion
        self.progress_bar.configure(style="green.Horizontal.TProgressbar")
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", 
                       background="green", 
                       troughcolor="lightgray")
        
        # Stop update thread
        self.active = False
        
        # Auto-close after delay
        if auto_close:
            self.window.after(3000, self.close)
        
        # Process events to update UI
        self.window.update()
    
    def close(self):
        """Close the progress window"""
        if self.window:
            self.window.destroy()
            self.window = None


# Message Dialogs
def show_error(title: str, message: str):
    """Show an error dialog"""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()

def show_warning(title: str, message: str):
    """Show a warning dialog"""
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning(title, message)
    root.destroy()

def show_info(title: str, message: str):
    """Show an info dialog"""
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()

def show_confirmation(title: str, message: str) -> bool:
    """Show a confirmation dialog"""
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(title, message)
    root.destroy()
    return result


# Audio-Specific Utilities
# ========================

# Audio File Format Detection
SUPPORTED_AUDIO_FORMATS = {
    '.mp3': 'MPEG Audio Layer III',
    '.m4a': 'MPEG-4 Audio',
    '.aac': 'Advanced Audio Coding',
    '.wav': 'Waveform Audio File Format',
    '.flac': 'Free Lossless Audio Codec',
    '.ogg': 'Ogg Vorbis',
    '.opus': 'Opus Audio',
    '.aiff': 'Audio Interchange File Format',
    '.wma': 'Windows Media Audio',
    '.mp4': 'MPEG-4 Video (Audio)',
    '.mov': 'QuickTime Movie (Audio)',
    '.avi': 'Audio Video Interleave (Audio)'
}

def is_audio_file(file_path: Path) -> bool:
    """Check if a file is a supported audio format"""
    return file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS

def get_audio_format_description(file_path: Path) -> str:
    """Get human-readable description of audio format"""
    ext = file_path.suffix.lower()
    return SUPPORTED_AUDIO_FORMATS.get(ext, "Unknown Audio Format")

def find_audio_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all audio files in a directory"""
    audio_files = []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
        
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_audio_file(file_path):
            audio_files.append(file_path)
            
    return sorted(audio_files)


# Audio Metadata Utilities
def get_audio_duration(file_path: Path) -> Optional[float]:
    """Get audio file duration in seconds using ffprobe"""
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(file_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"Error getting audio duration: {e}")
    
    return None

def format_duration(seconds: float) -> str:
    """Format duration from seconds to human-readable format"""
    if seconds is None:
        return "Unknown"
        
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for filesystem compatibility"""
    import re
    
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    filename = re.sub(invalid_chars, '_', filename)
    
    # Replace multiple spaces/underscores with single underscore
    filename = re.sub(r'[_\s]+', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        available_length = max_length - len(ext)
        filename = name[:available_length] + ext
        
    return filename


# File Hash Utilities
def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> Optional[str]:
    """Calculate hash of a file"""
    import hashlib
    
    try:
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return None

def find_duplicate_files(file_paths: List[Path]) -> Dict[str, List[Path]]:
    """Find duplicate files based on SHA-256 hash"""
    duplicates = {}
    hash_to_files = {}
    
    for file_path in file_paths:
        file_hash = calculate_file_hash(file_path)
        if file_hash:
            if file_hash in hash_to_files:
                hash_to_files[file_hash].append(file_path)
            else:
                hash_to_files[file_hash] = [file_path]
                
    # Only keep groups with duplicates
    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            duplicates[file_hash] = files
            
    return duplicates


# Spotify API Utilities
class SpotifyHelper:
    """Helper utilities for Spotify API integration"""
    
    @staticmethod
    def get_spotify_config_path() -> Path:
        """Get path to Spotify configuration file"""
        return Path.home() / ".spotify_config.json"
    
    @staticmethod
    def load_spotify_credentials() -> Dict[str, str]:
        """Load Spotify API credentials"""
        config_path = SpotifyHelper.get_spotify_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading Spotify config: {e}")
                
        return {}
    
    @staticmethod
    def save_spotify_credentials(credentials: Dict[str, str]) -> bool:
        """Save Spotify API credentials"""
        config_path = SpotifyHelper.get_spotify_config_path()
        
        try:
            with open(config_path, 'w') as f:
                json.dump(credentials, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving Spotify config: {e}")
            return False
    
    @staticmethod
    def setup_spotify_credentials() -> Optional[Dict[str, str]]:
        """Setup Spotify API credentials through GUI prompts"""
        credentials = SpotifyHelper.load_spotify_credentials()
        
        # Check if we need to prompt for credentials
        required_keys = ['client_id', 'client_secret']
        missing_keys = [key for key in required_keys if not credentials.get(key)]
        
        if missing_keys:
            root = tk.Tk()
            root.withdraw()
            
            for key in missing_keys:
                prompt = f"Enter Spotify {key.replace('_', ' ').title()}:"
                value = simpledialog.askstring(
                    "Spotify Setup",
                    prompt,
                    parent=root,
                    show='*' if 'secret' in key else None
                )
                
                if not value:
                    root.destroy()
                    return None
                    
                credentials[key] = value
                
            root.destroy()
            
            # Save credentials
            if SpotifyHelper.save_spotify_credentials(credentials):
                return credentials
                
        return credentials if all(credentials.get(key) for key in required_keys) else None


# Apple Podcasts Database Utilities
class ApplePodcastsHelper:
    """Helper utilities for Apple Podcasts database access"""
    
    @staticmethod
    def get_podcasts_db_path() -> Optional[Path]:
        """Get path to Apple Podcasts database"""
        possible_paths = [
            Path.home() / "Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Documents/MTLibrary.sqlite",
            Path.home() / "Library/Containers/com.apple.podcasts/Data/Documents/MTLibrary.sqlite"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        return None
    
    @staticmethod
    def query_podcasts_db(query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query on the Apple Podcasts database"""
        import sqlite3
        
        db_path = ApplePodcastsHelper.get_podcasts_db_path()
        if not db_path:
            return []
            
        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Error querying podcasts database: {e}")
            return []
    
    @staticmethod
    def get_podcast_shows() -> List[Dict[str, Any]]:
        """Get list of podcast shows"""
        query = """
        SELECT DISTINCT title, author, feedURL, artworkURL
        FROM ZMTPODCAST 
        WHERE title IS NOT NULL
        ORDER BY title
        """
        
        return ApplePodcastsHelper.query_podcasts_db(query)
    
    @staticmethod
    def get_podcast_episodes(show_title: str = None) -> List[Dict[str, Any]]:
        """Get podcast episodes, optionally filtered by show"""
        if show_title:
            query = """
            SELECT e.title, e.author, e.pubDate, e.duration, e.assetURL, p.title as show_title
            FROM ZMTEPISODE e
            JOIN ZMTPODCAST p ON e.podcast = p.Z_PK
            WHERE p.title = ? AND e.title IS NOT NULL
            ORDER BY e.pubDate DESC
            """
            params = (show_title,)
        else:
            query = """
            SELECT e.title, e.author, e.pubDate, e.duration, e.assetURL, p.title as show_title
            FROM ZMTEPISODE e
            JOIN ZMTPODCAST p ON e.podcast = p.Z_PK
            WHERE e.title IS NOT NULL
            ORDER BY e.pubDate DESC
            LIMIT 100
            """
            params = ()
            
        return ApplePodcastsHelper.query_podcasts_db(query, params)


# ElevenLabs API Utilities
class ElevenLabsHelper:
    """Helper utilities for ElevenLabs API integration"""
    
    @staticmethod
    def get_config_path() -> Path:
        """Get path to ElevenLabs configuration file"""
        return Path.home() / ".elevenlabs_config.json"
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load ElevenLabs configuration"""
        config_path = ElevenLabsHelper.get_config_path()
        
        default_config = {
            "api_key": "",
            "voice_id": "",
            "model": "eleven_monolingual_v1",
            "stability": 0.5,
            "similarity_boost": 0.75,
            "chunk_size": 2000
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logging.error(f"Error loading ElevenLabs config: {e}")
                
        return default_config
    
    @staticmethod
    def save_config(config: Dict[str, Any]) -> bool:
        """Save ElevenLabs configuration"""
        config_path = ElevenLabsHelper.get_config_path()
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving ElevenLabs config: {e}")
            return False
    
    @staticmethod
    def test_api_key(api_key: str) -> bool:
        """Test if an ElevenLabs API key is valid"""
        try:
            import requests
            
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.elevenlabs.io/v1/voices",
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"Error testing ElevenLabs API key: {e}")
            return False
    
    @staticmethod
    def split_text_for_api(text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into chunks suitable for ElevenLabs API"""
        import re
        
        chunks = []
        current_chunk = ""
        
        # Split into sentences
        sentences = re.split('(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks


# Audio Processing Utilities
class AudioProcessor:
    """Utilities for audio file processing and conversion"""
    
    @staticmethod
    def get_ffmpeg_path() -> Optional[str]:
        """Get path to ffmpeg executable"""
        import shutil
        return shutil.which('ffmpeg')
    
    @staticmethod
    def convert_audio_format(input_path: Path, output_path: Path, 
                           format: str = 'mp3', quality: str = 'high') -> bool:
        """Convert audio file to different format using ffmpeg"""
        ffmpeg_path = AudioProcessor.get_ffmpeg_path()
        if not ffmpeg_path:
            logging.error("ffmpeg not found. Please install ffmpeg.")
            return False
            
        try:
            # Quality settings
            quality_settings = {
                'high': ['-b:a', '320k'],
                'medium': ['-b:a', '192k'],
                'low': ['-b:a', '128k']
            }
            
            cmd = [
                ffmpeg_path,
                '-i', str(input_path),
                '-y',  # Overwrite output file
                *quality_settings.get(quality, quality_settings['high']),
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"Error converting audio: {e}")
            return False
    
    @staticmethod
    def extract_audio_from_video(video_path: Path, audio_path: Path, 
                               format: str = 'mp3') -> bool:
        """Extract audio track from video file"""
        ffmpeg_path = AudioProcessor.get_ffmpeg_path()
        if not ffmpeg_path:
            logging.error("ffmpeg not found. Please install ffmpeg.")
            return False
            
        try:
            cmd = [
                ffmpeg_path,
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'libmp3lame' if format == 'mp3' else format,
                '-y',   # Overwrite output file
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"Error extracting audio: {e}")
            return False
    
    @staticmethod
    def normalize_audio_volume(input_path: Path, output_path: Path, 
                             target_level: str = '-16dB') -> bool:
        """Normalize audio volume using ffmpeg"""
        ffmpeg_path = AudioProcessor.get_ffmpeg_path()
        if not ffmpeg_path:
            logging.error("ffmpeg not found. Please install ffmpeg.")
            return False
            
        try:
            cmd = [
                ffmpeg_path,
                '-i', str(input_path),
                '-af', f'loudnorm=I={target_level}',
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"Error normalizing audio: {e}")
            return False


# Desktop Integration Utilities
def setup_desktop_logging(script_name: str) -> Path:
    """Setup logging with file placed on desktop"""
    desktop_path = Path.home() / "Desktop"
    log_file = desktop_path / f"{script_name}.log"
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite existing log
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    logging.info(f"Logging initialized: {log_file}")
    return log_file

def open_file_in_finder(file_path: Path):
    """Open file or folder in macOS Finder"""
    try:
        subprocess.run(['open', '-R', str(file_path)], check=True)
    except Exception as e:
        logging.error(f"Error opening in Finder: {e}")

def play_audio_file(file_path: Path):
    """Play audio file using macOS built-in player"""
    try:
        subprocess.run(['afplay', str(file_path)], check=True)
    except Exception as e:
        logging.error(f"Error playing audio: {e}")


if __name__ == "__main__":
    print("This is a utility module and not meant to be run directly.")
    print("Import it into your audio scripts: from audio_utils import *")
    sys.exit(1)