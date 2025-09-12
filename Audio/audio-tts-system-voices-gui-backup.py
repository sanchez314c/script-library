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
# Script Name: audio-tts-system-voices-gui.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive text-to-speech application supporting both system    
#              voices (via pyttsx3) and ElevenLabs API with intuitive GUI        
#              interface, batch processing, and audio file generation.           
#
# Usage: python audio-tts-system-voices-gui.py [--cli] [--file FILE] [--output OUTPUT] 
#
# Dependencies: pyttsx3, requests, pydub, tkinter, concurrent.futures              
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Supports both CLI and GUI modes, intelligent text chunking for large     
#        documents, multi-core processing, and comprehensive error handling.      
#                                                                                
####################################################################################

"""
Text-to-Speech Universal Application
===================================

A comprehensive text-to-speech solution that combines system voices and cloud-based
speech synthesis into a single, user-friendly application. Features both command-line
and graphical interfaces for maximum flexibility.

Core Features:
- System voice integration via pyttsx3
- ElevenLabs API cloud synthesis
- Intuitive GUI with voice preview
- CLI mode for automation
- Batch text processing
- Audio file generation
- Smart text chunking
- Multi-core processing
- Comprehensive error handling

Technical Architecture:
- Provider-based design for extensibility
- Threaded GUI for responsiveness
- Progress tracking for long operations
- Configuration persistence
- Universal macOS compatibility

Author: sanchez314c@speedheathens.com
Version: 1.0.0
License: MIT
"""

import os
import re
import sys
import json
import time
import logging
import tempfile
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from tkinter import (
    Tk, ttk, StringVar, BooleanVar, IntVar, DoubleVar,
    Frame, Label, Button, Text, Scrollbar, Menu, Toplevel,
    messagebox, filedialog, simpledialog
)
import tkinter as tk
from threading import Thread

# Dependency management and utilities
try:
    from audio_utils import (
        check_and_install_dependencies, setup_logging, 
        setup_argument_parser, print_script_info,
        select_file, select_save_file, show_error, 
        show_info, show_confirmation, CredentialManager,
        ProgressTracker
    )
    
    REQUIRED_PACKAGES = [
        "pyttsx3>=2.90",
        "requests>=2.25.0",
        "pydub>=0.25.0"
    ]
    
    if not check_and_install_dependencies(REQUIRED_PACKAGES):
        sys.exit(1)
        
except ImportError:
    print("Warning: audio_utils module not found, using fallback implementations")
    
    def check_and_install_dependencies(packages):
        return True
    
    def setup_logging(script_path=None):
        log_file = Path(__file__).with_suffix('.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        return log_file

# Import required packages
try:
    import pyttsx3
    import requests
    from pydub import AudioSegment
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install missing packages: pip install pyttsx3 requests pydub")
    sys.exit(1)


class TTSProvider:
    """Base class for TTS providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.error_message = ""
        
    def initialize(self) -> bool:
        """Initialize the provider. Returns True if successful."""
        raise NotImplementedError
        
    def speak_text(self, text: str, **kwargs) -> bool:
        """Speak text immediately. Returns True if successful."""
        raise NotImplementedError
        
    def save_audio(self, text: str, output_path: Path, **kwargs) -> bool:
        """Save text as audio file. Returns True if successful."""
        raise NotImplementedError
        
    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices for this provider."""
        return []
        
    def get_settings_widget(self, parent) -> Optional[Frame]:
        """Get a settings widget for this provider."""
        return None


class SystemVoiceProvider(TTSProvider):
    """System voice provider using pyttsx3"""
    
    def __init__(self):
        super().__init__("System Voices")
        self.engine = None
        self.voices = []
        
    def initialize(self) -> bool:
        """Initialize pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            self.voices = self.get_voices()
            self.available = True
            return True
        except Exception as e:
            self.error_message = f"Failed to initialize system voices: {e}"
            logging.error(self.error_message)
            return False
            
    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available system voices"""
        if not self.engine:
            return []
            
        try:
            voices = self.engine.getProperty('voices')
            return [
                {
                    'id': voice.id,
                    'name': voice.name,
                    'language': getattr(voice, 'languages', ['Unknown'])[0] if hasattr(voice, 'languages') else 'Unknown'
                }
                for voice in voices
            ]
        except Exception as e:
            logging.error(f"Error getting system voices: {e}")
            return []
            
    def speak_text(self, text: str, voice_id: str = None, 
                  rate: int = 150, volume: float = 1.0, **kwargs) -> bool:
        """Speak text using system voice"""
        if not self.engine:
            return False
            
        try:
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logging.error(f"Error speaking text: {e}")
            return False
            
    def save_audio(self, text: str, output_path: Path, 
                   voice_id: str = None, rate: int = 150, 
                   volume: float = 1.0, **kwargs) -> bool:
        """Save text as audio file using system voice"""
        if not self.engine:
            return False
            
        try:
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Save to file
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()
            return True
        except Exception as e:
            logging.error(f"Error saving audio: {e}")
            return False


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs API provider"""
    
    def __init__(self):
        super().__init__("ElevenLabs")
        self.api_key = None
        self.voices = []
        self.config = {}
        self.chunk_size = 2000
        
    def initialize(self) -> bool:
        """Initialize ElevenLabs provider"""
        try:
            # Load configuration
            config_path = Path.home() / ".elevenlabs_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    
            # Get API key
            if 'ELEVENLABS_API_KEY' in os.environ:
                self.api_key = os.environ['ELEVENLABS_API_KEY']
            elif self.config.get('api_key'):
                self.api_key = self.config['api_key']
            else:
                self.api_key = self._prompt_for_api_key()
                
            if not self.api_key:
                self.error_message = "ElevenLabs API key not provided"
                return False
                
            # Test API key by fetching voices
            self.voices = self._fetch_voices()
            if not self.voices:
                self.error_message = "Could not fetch voices from ElevenLabs API"
                return False
                
            self.available = True
            return True
        except Exception as e:
            self.error_message = f"ElevenLabs initialization failed: {e}"
            logging.error(self.error_message)
            return False
            
    def _prompt_for_api_key(self) -> Optional[str]:
        """Prompt user for API key"""
        try:
            if 'CredentialManager' in globals():
                return CredentialManager.get_credential(
                    "ElevenLabs", 
                    "api_key", 
                    "Please enter your ElevenLabs API key:",
                    password=True
                )
            else:
                # Fallback prompt
                root = Tk()
                root.withdraw()
                api_key = simpledialog.askstring(
                    "ElevenLabs Setup",
                    "Please enter your ElevenLabs API key:",
                    show='*'
                )
                root.destroy()
                
                if api_key:
                    # Save to config
                    self.config['api_key'] = api_key
                    self._save_config()
                    
                return api_key
        except Exception as e:
            logging.error(f"Error prompting for API key: {e}")
            return None
            
    def _save_config(self):
        """Save configuration to file"""
        try:
            config_path = Path.home() / ".elevenlabs_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            
    def _fetch_voices(self) -> List[Dict[str, Any]]:
        """Fetch available voices from ElevenLabs API"""
        try:
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.elevenlabs.io/v1/voices",
                headers=headers
            )
            response.raise_for_status()
            
            voices_data = response.json()
            return [
                {
                    'id': voice['voice_id'],
                    'name': voice['name'],
                    'category': voice.get('category', 'Unknown'),
                    'description': voice.get('description', '')
                }
                for voice in voices_data.get('voices', [])
            ]
        except Exception as e:
            logging.error(f"Error fetching ElevenLabs voices: {e}")
            return []
            
    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available ElevenLabs voices"""
        return self.voices
        
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        chunks = []
        current_chunk = ""
        
        # Split into sentences
        sentences = re.split('(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
        
    def _convert_chunk(self, chunk: str, voice_id: str) -> Optional[bytes]:
        """Convert text chunk to audio using ElevenLabs API"""
        try:
            headers = {
                "Accept": "audio/mpeg",
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": chunk,
                "model_id": self.config.get("model", "eleven_monolingual_v1"),
                "voice_settings": {
                    "stability": self.config.get("stability", 0.5),
                    "similarity_boost": self.config.get("similarity_boost", 0.75)
                }
            }
            
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            return response.content
        except Exception as e:
            logging.error(f"Error converting chunk: {e}")
            return None
            
    def speak_text(self, text: str, voice_id: str = None, **kwargs) -> bool:
        """Speak text using ElevenLabs (saves temporarily and plays)"""
        try:
            if not voice_id and self.voices:
                voice_id = self.voices[0]['id']
                
            # Convert and play
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                if self.save_audio(text, Path(tmp_file.name), voice_id=voice_id, **kwargs):
                    # Play the audio file (macOS)
                    os.system(f"afplay '{tmp_file.name}'")
                    os.unlink(tmp_file.name)
                    return True
                else:
                    os.unlink(tmp_file.name)
                    return False
        except Exception as e:
            logging.error(f"Error speaking text: {e}")
            return False
            
    def save_audio(self, text: str, output_path: Path, voice_id: str = None, 
                   progress_callback: callable = None, **kwargs) -> bool:
        """Save text as audio file using ElevenLabs"""
        try:
            if not voice_id and self.voices:
                voice_id = self.voices[0]['id']
                
            chunks = self._split_text_into_chunks(text)
            total_chunks = len(chunks)
            
            if total_chunks == 1:
                # Single chunk - direct conversion
                audio_data = self._convert_chunk(chunks[0], voice_id)
                if audio_data:
                    output_path.write_bytes(audio_data)
                    return True
                return False
            else:
                # Multiple chunks - convert and merge
                temp_files = []
                
                for i, chunk in enumerate(chunks):
                    if progress_callback:
                        progress_callback(i + 1, total_chunks, f"Converting chunk {i + 1}/{total_chunks}")
                        
                    audio_data = self._convert_chunk(chunk, voice_id)
                    if audio_data:
                        temp_file = Path(tempfile.mktemp(suffix='.mp3'))
                        temp_file.write_bytes(audio_data)
                        temp_files.append(temp_file)
                    else:
                        # Clean up on error
                        for temp_file in temp_files:
                            temp_file.unlink()
                        return False
                        
                # Merge audio files
                if progress_callback:
                    progress_callback(total_chunks, total_chunks, "Merging audio chunks...")
                    
                combined = AudioSegment.empty()
                for temp_file in temp_files:
                    segment = AudioSegment.from_mp3(temp_file)
                    combined += segment
                    temp_file.unlink()
                    
                combined.export(output_path, format="mp3")
                return True
                
        except Exception as e:
            logging.error(f"Error saving audio: {e}")
            return False


class TTSApplication:
    """Main TTS application with GUI"""
    
    def __init__(self):
        self.log_file = setup_logging(Path(__file__))
        self.providers = {}
        self.current_provider = None
        self.setup_providers()
        self.setup_gui()
        
    def setup_providers(self):
        """Initialize TTS providers"""
        # System voices provider
        system_provider = SystemVoiceProvider()
        if system_provider.initialize():
            self.providers["system"] = system_provider
            logging.info("System voice provider initialized successfully")
        else:
            logging.warning("System voice provider failed to initialize")
            
        # ElevenLabs provider
        elevenlabs_provider = ElevenLabsProvider()
        if elevenlabs_provider.initialize():
            self.providers["elevenlabs"] = elevenlabs_provider
            logging.info("ElevenLabs provider initialized successfully")
        else:
            logging.warning(f"ElevenLabs provider failed to initialize: {elevenlabs_provider.error_message}")
            
        # Set default provider
        if "system" in self.providers:
            self.current_provider = self.providers["system"]
        elif "elevenlabs" in self.providers:
            self.current_provider = self.providers["elevenlabs"]
            
    def setup_gui(self):
        """Setup the main GUI"""
        self.root = Tk()
        self.root.title("Text-to-Speech Universal Application")
        self.root.geometry("800x700")
        
        # Variables
        self.provider_var = StringVar()
        self.voice_var = StringVar()
        self.rate_var = IntVar(value=150)
        self.volume_var = DoubleVar(value=1.0)
        self.text_var = StringVar()
        
        self.setup_menu()
        self.setup_main_interface()
        
    def setup_menu(self):
        """Setup the application menu"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Text File...", command=self.load_text_file)
        file_menu.add_command(label="Save Audio As...", command=self.save_audio_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="ElevenLabs Configuration...", command=self.configure_elevenlabs)
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_main_interface(self):
        """Setup the main interface"""
        # Provider selection frame
        provider_frame = ttk.LabelFrame(self.root, text="TTS Provider")
        provider_frame.pack(fill="x", padx=10, pady=5)
        
        provider_options = []
        for key, provider in self.providers.items():
            provider_options.append(f"{provider.name}")
            
        if provider_options:
            provider_combo = ttk.Combobox(
                provider_frame, 
                textvariable=self.provider_var,
                values=provider_options,
                state="readonly"
            )
            provider_combo.pack(fill="x", padx=5, pady=5)
            provider_combo.bind("<<ComboboxSelected>>", self.on_provider_changed)
            provider_combo.set(provider_options[0])
            self.on_provider_changed()
        else:
            ttk.Label(provider_frame, text="No TTS providers available").pack(pady=5)
            
        # Voice selection frame
        self.voice_frame = ttk.LabelFrame(self.root, text="Voice Selection")
        self.voice_frame.pack(fill="x", padx=10, pady=5)
        
        self.voice_combo = ttk.Combobox(
            self.voice_frame,
            textvariable=self.voice_var,
            state="readonly"
        )
        self.voice_combo.pack(fill="x", padx=5, pady=5)
        
        # Settings frame
        self.settings_frame = ttk.LabelFrame(self.root, text="Voice Settings")
        self.settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Rate setting (for system voices)
        rate_frame = Frame(self.settings_frame)
        rate_frame.pack(fill="x", padx=5, pady=2)
        Label(rate_frame, text="Speech Rate:").pack(side="left")
        self.rate_scale = tk.Scale(
            rate_frame,
            from_=50, to=300,
            orient="horizontal",
            variable=self.rate_var
        )
        self.rate_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Volume setting (for system voices)
        volume_frame = Frame(self.settings_frame)
        volume_frame.pack(fill="x", padx=5, pady=2)
        Label(volume_frame, text="Volume:").pack(side="left")
        self.volume_scale = tk.Scale(
            volume_frame,
            from_=0.0, to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.volume_var
        )
        self.volume_scale.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Text input frame
        text_frame = ttk.LabelFrame(self.root, text="Text Input")
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Text widget with scrollbar
        text_widget_frame = Frame(text_frame)
        text_widget_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.text_widget = Text(text_widget_frame, wrap="word", height=10)
        scrollbar = Scrollbar(text_widget_frame, orient="vertical", command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Control buttons frame
        control_frame = Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        Button(control_frame, text="Preview Voice", command=self.preview_voice).pack(side="left", padx=5)
        Button(control_frame, text="Speak Text", command=self.speak_text).pack(side="left", padx=5)
        Button(control_frame, text="Save as Audio", command=self.save_audio_file).pack(side="left", padx=5)
        Button(control_frame, text="Clear Text", command=self.clear_text).pack(side="right", padx=5)
        
        # Status bar
        self.status_var = StringVar(value="Ready")
        status_bar = Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(fill="x", side="bottom")
        
    def on_provider_changed(self, event=None):
        """Handle provider selection change"""
        provider_name = self.provider_var.get()
        
        # Find provider by name
        for key, provider in self.providers.items():
            if provider.name == provider_name:
                self.current_provider = provider
                break
                
        if self.current_provider:
            # Update voice list
            voices = self.current_provider.get_voices()
            voice_names = [voice['name'] for voice in voices]
            
            self.voice_combo['values'] = voice_names
            if voice_names:
                self.voice_combo.set(voice_names[0])
                
            # Update settings visibility
            if isinstance(self.current_provider, SystemVoiceProvider):
                self.rate_scale.pack_info()
                self.volume_scale.pack_info()
            else:
                # Hide system-specific settings for other providers
                pass
                
    def get_current_voice_id(self) -> Optional[str]:
        """Get the ID of the currently selected voice"""
        if not self.current_provider:
            return None
            
        voice_name = self.voice_var.get()
        voices = self.current_provider.get_voices()
        
        for voice in voices:
            if voice['name'] == voice_name:
                return voice['id']
                
        return voices[0]['id'] if voices else None
        
    def preview_voice(self):
        """Preview the selected voice"""
        if not self.current_provider:
            messagebox.showerror("Error", "No TTS provider available")
            return
            
        voice_id = self.get_current_voice_id()
        if not voice_id:
            messagebox.showerror("Error", "No voice selected")
            return
            
        preview_text = "This is a preview of the selected voice and settings."
        
        def preview_thread():
            self.status_var.set("Playing preview...")
            try:
                success = self.current_provider.speak_text(
                    preview_text,
                    voice_id=voice_id,
                    rate=self.rate_var.get(),
                    volume=self.volume_var.get()
                )
                if not success:
                    messagebox.showerror("Error", "Failed to preview voice")
            except Exception as e:
                messagebox.showerror("Error", f"Preview failed: {e}")
            finally:
                self.status_var.set("Ready")
                
        Thread(target=preview_thread, daemon=True).start()
        
    def speak_text(self):
        """Speak the text in the text widget"""
        if not self.current_provider:
            messagebox.showerror("Error", "No TTS provider available")
            return
            
        text = self.text_widget.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to speak")
            return
            
        voice_id = self.get_current_voice_id()
        if not voice_id:
            messagebox.showerror("Error", "No voice selected")
            return
            
        def speak_thread():
            self.status_var.set("Speaking...")
            try:
                success = self.current_provider.speak_text(
                    text,
                    voice_id=voice_id,
                    rate=self.rate_var.get(),
                    volume=self.volume_var.get()
                )
                if not success:
                    messagebox.showerror("Error", "Failed to speak text")
            except Exception as e:
                messagebox.showerror("Error", f"Speech failed: {e}")
            finally:
                self.status_var.set("Ready")
                
        Thread(target=speak_thread, daemon=True).start()
        
    def load_text_file(self):
        """Load text from a file"""
        try:
            if 'select_file' in globals():
                file_path = select_file(
                    "Select Text File",
                    [("Text Files", "*.txt"), ("All Files", "*.*")]
                )
            else:
                file_path = filedialog.askopenfilename(
                    title="Select Text File",
                    filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
                )
                file_path = Path(file_path) if file_path else None
                
            if file_path and file_path.exists():
                text = file_path.read_text(encoding='utf-8')
                self.text_widget.delete("1.0", "end")
                self.text_widget.insert("1.0", text)
                self.status_var.set(f"Loaded: {file_path.name}")
            elif file_path:
                messagebox.showerror("Error", f"File not found: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            
    def save_audio_file(self):
        """Save text as audio file"""
        if not self.current_provider:
            messagebox.showerror("Error", "No TTS provider available")
            return
            
        text = self.text_widget.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to convert")
            return
            
        voice_id = self.get_current_voice_id()
        if not voice_id:
            messagebox.showerror("Error", "No voice selected")
            return
            
        try:
            if 'select_save_file' in globals():
                output_path = select_save_file(
                    "Save Audio As",
                    [("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("All Files", "*.*")],
                    default_extension=".mp3"
                )
            else:
                output_path = filedialog.asksaveasfilename(
                    title="Save Audio As",
                    filetypes=[("MP3 Files", "*.mp3"), ("WAV Files", "*.wav"), ("All Files", "*.*")],
                    defaultextension=".mp3"
                )
                output_path = Path(output_path) if output_path else None
                
            if not output_path:
                return
                
            def save_thread():
                self.status_var.set("Converting text to audio...")
                
                # Create progress window for long operations
                if isinstance(self.current_provider, ElevenLabsProvider):
                    progress_window = Toplevel(self.root)
                    progress_window.title("Converting Audio")
                    progress_window.geometry("400x100")
                    progress_window.resizable(False, False)
                    
                    progress_var = DoubleVar()
                    progress_bar = ttk.Progressbar(
                        progress_window,
                        variable=progress_var,
                        maximum=100,
                        length=350
                    )
                    progress_bar.pack(pady=20)
                    
                    status_label = Label(progress_window, text="Initializing...")
                    status_label.pack()
                    
                    def progress_callback(current, total, status):
                        percentage = (current / total) * 100
                        progress_var.set(percentage)
                        status_label.config(text=status)
                        progress_window.update()
                        
                    progress_window.update()
                else:
                    progress_callback = None
                    
                try:
                    success = self.current_provider.save_audio(
                        text,
                        output_path,
                        voice_id=voice_id,
                        rate=self.rate_var.get(),
                        volume=self.volume_var.get(),
                        progress_callback=progress_callback
                    )
                    
                    if success:
                        self.status_var.set(f"Audio saved: {output_path.name}")
                        messagebox.showinfo("Success", f"Audio saved successfully as:\n{output_path}")
                    else:
                        messagebox.showerror("Error", "Failed to save audio")
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Conversion failed: {e}")
                finally:
                    if isinstance(self.current_provider, ElevenLabsProvider):
                        progress_window.destroy()
                    self.status_var.set("Ready")
                    
            Thread(target=save_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save audio: {e}")
            
    def clear_text(self):
        """Clear the text widget"""
        self.text_widget.delete("1.0", "end")
        self.status_var.set("Ready")
        
    def configure_elevenlabs(self):
        """Configure ElevenLabs settings"""
        if "elevenlabs" not in self.providers:
            messagebox.showinfo("Info", "ElevenLabs provider is not available")
            return
            
        config_window = Toplevel(self.root)
        config_window.title("ElevenLabs Configuration")
        config_window.geometry("400x300")
        config_window.resizable(False, False)
        
        # API Key entry
        api_frame = ttk.LabelFrame(config_window, text="API Configuration")
        api_frame.pack(fill="x", padx=10, pady=10)
        
        Label(api_frame, text="API Key:").pack(anchor="w", padx=5, pady=2)
        api_key_var = StringVar(value=self.providers["elevenlabs"].api_key or "")
        api_entry = tk.Entry(api_frame, textvariable=api_key_var, show="*", width=40)
        api_entry.pack(fill="x", padx=5, pady=2)
        
        # Model selection
        Label(api_frame, text="Model:").pack(anchor="w", padx=5, pady=2)
        model_var = StringVar(value=self.providers["elevenlabs"].config.get("model", "eleven_monolingual_v1"))
        model_combo = ttk.Combobox(
            api_frame,
            textvariable=model_var,
            values=["eleven_monolingual_v1", "eleven_multilingual_v1", "eleven_multilingual_v2"],
            state="readonly"
        )
        model_combo.pack(fill="x", padx=5, pady=2)
        
        # Voice settings
        voice_frame = ttk.LabelFrame(config_window, text="Voice Settings")
        voice_frame.pack(fill="x", padx=10, pady=10)
        
        Label(voice_frame, text="Stability:").pack(anchor="w", padx=5, pady=2)
        stability_var = DoubleVar(value=self.providers["elevenlabs"].config.get("stability", 0.5))
        stability_scale = tk.Scale(
            voice_frame,
            from_=0.0, to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=stability_var
        )
        stability_scale.pack(fill="x", padx=5, pady=2)
        
        Label(voice_frame, text="Similarity Boost:").pack(anchor="w", padx=5, pady=2)
        similarity_var = DoubleVar(value=self.providers["elevenlabs"].config.get("similarity_boost", 0.75))
        similarity_scale = tk.Scale(
            voice_frame,
            from_=0.0, to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=similarity_var
        )
        similarity_scale.pack(fill="x", padx=5, pady=2)
        
        # Buttons
        button_frame = Frame(config_window)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        def save_config():
            try:
                provider = self.providers["elevenlabs"]
                provider.api_key = api_key_var.get()
                provider.config.update({
                    "api_key": api_key_var.get(),
                    "model": model_var.get(),
                    "stability": stability_var.get(),
                    "similarity_boost": similarity_var.get()
                })
                provider._save_config()
                
                # Refresh voices
                provider.voices = provider._fetch_voices()
                self.on_provider_changed()
                
                messagebox.showinfo("Success", "Configuration saved successfully")
                config_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
                
        Button(button_frame, text="Save", command=save_config).pack(side="right", padx=5)
        Button(button_frame, text="Cancel", command=config_window.destroy).pack(side="right")
        
    def show_about(self):
        """Show about dialog"""
        about_text = """Text-to-Speech Universal Application v1.0.0

A comprehensive text-to-speech solution supporting:
• System voices (via pyttsx3)
• ElevenLabs cloud synthesis
• Batch text processing
• Audio file generation

Author: sanchez314c@speedheathens.com
License: MIT"""
        
        messagebox.showinfo("About", about_text)
        
    def run(self):
        """Run the application"""
        if not self.providers:
            messagebox.showerror(
                "Error", 
                "No TTS providers available. Please check your configuration."
            )
            return
            
        self.root.mainloop()


def run_cli_mode(args):
    """Run in CLI mode"""
    print("Text-to-Speech CLI Mode")
    print("-" * 30)
    
    # Initialize providers
    providers = {}
    
    # System provider
    system_provider = SystemVoiceProvider()
    if system_provider.initialize():
        providers["system"] = system_provider
        print("✓ System voices available")
    
    # ElevenLabs provider
    elevenlabs_provider = ElevenLabsProvider()
    if elevenlabs_provider.initialize():
        providers["elevenlabs"] = elevenlabs_provider
        print("✓ ElevenLabs available")
    
    if not providers:
        print("✗ No TTS providers available")
        return 1
        
    # Select provider
    if len(providers) == 1:
        provider_key = list(providers.keys())[0]
        provider = providers[provider_key]
        print(f"Using {provider.name}")
    else:
        print("\nAvailable providers:")
        provider_keys = list(providers.keys())
        for i, key in enumerate(provider_keys):
            print(f"{i + 1}. {providers[key].name}")
            
        try:
            choice = int(input("Select provider (1-{}): ".format(len(provider_keys))))
            provider_key = provider_keys[choice - 1]
            provider = providers[provider_key]
        except (ValueError, IndexError):
            print("Invalid selection")
            return 1
            
    # Get text input
    if args.file:
        try:
            text = Path(args.file).read_text()
            print(f"Loaded text from: {args.file}")
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    else:
        print("\nEnter text to speak (press Ctrl+D when done):")
        try:
            text_lines = []
            while True:
                line = input()
                text_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            text = "\n".join(text_lines)
            
    if not text.strip():
        print("No text provided")
        return 1
        
    # Select voice
    voices = provider.get_voices()
    if not voices:
        print("No voices available")
        return 1
        
    if len(voices) == 1:
        voice = voices[0]
    else:
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"{i + 1}. {voice['name']}")
            
        try:
            choice = int(input("Select voice (1-{}): ".format(len(voices))))
            voice = voices[choice - 1]
        except (ValueError, IndexError):
            voice = voices[0]
            print(f"Using default voice: {voice['name']}")
            
    # Process text
    if args.output:
        print(f"Saving audio to: {args.output}")
        success = provider.save_audio(text, Path(args.output), voice_id=voice['id'])
        if success:
            print("Audio saved successfully")
            return 0
        else:
            print("Failed to save audio")
            return 1
    else:
        print("Speaking text...")
        success = provider.speak_text(text, voice_id=voice['id'])
        if success:
            print("Speech completed")
            return 0
        else:
            print("Failed to speak text")
            return 1


def main():
    """Main entry point"""
    # Script metadata
    SCRIPT_NAME = "Text-to-Speech Universal Application"
    SCRIPT_VERSION = "1.0.0"
    SCRIPT_AUTHOR = "sanchez314c@speedheathens.com"
    SCRIPT_DATE = "2025-01-23"
    
    try:
        # Set up argument parser
        if 'setup_argument_parser' in globals():
            parser = setup_argument_parser(
                description=__doc__,
                args_config=[
                    {
                        'flags': ['--cli'],
                        'help': 'Run in command-line mode',
                        'action': 'store_true'
                    },
                    {
                        'flags': ['-f', '--file'],
                        'help': 'Text file to process',
                        'type': str,
                        'default': None
                    },
                    {
                        'flags': ['-o', '--output'],
                        'help': 'Output audio file path',
                        'type': str,
                        'default': None
                    }
                ]
            )
            
            args = parser.parse_args()
            
            if args.version:
                if 'print_script_info' in globals():
                    print_script_info(SCRIPT_NAME, SCRIPT_VERSION, SCRIPT_AUTHOR, SCRIPT_DATE)
                else:
                    print(f"{SCRIPT_NAME} v{SCRIPT_VERSION}")
                    print(f"By {SCRIPT_AUTHOR} ({SCRIPT_DATE})")
                return 0
                
        else:
            # Fallback argument parsing
            parser = argparse.ArgumentParser(description="Text-to-Speech Universal Application")
            parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
            parser.add_argument('-f', '--file', help='Text file to process')
            parser.add_argument('-o', '--output', help='Output audio file path')
            parser.add_argument('--version', action='store_true', help='Show version info')
            
            args = parser.parse_args()
            
            if args.version:
                print(f"{SCRIPT_NAME} v{SCRIPT_VERSION}")
                return 0
        
        # Run in appropriate mode
        if args.cli:
            return run_cli_mode(args)
        else:
            app = TTSApplication()
            app.run()
            return 0
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())