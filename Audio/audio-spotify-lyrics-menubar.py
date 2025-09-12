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
# Script Name: audio-spotify-lyrics-menubar.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: macOS menu bar application that fetches and displays lyrics for    
#              currently playing Spotify tracks. Integrates with Spotify API      
#              and multiple lyrics sources with real-time track monitoring        
#              and notification system.                                               
#
# Usage: python audio-spotify-lyrics-menubar.py [--reset-credentials] 
#
# Dependencies: spotipy, rumps, requests, beautifulsoup4                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Requires Spotify API credentials (Client ID/Secret) and optional Genius       
#        API credentials for enhanced lyrics fetching. Runs as persistent menubar   
#        application with background track monitoring.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
Spotify Lyrics Menu Bar App

macOS menu bar application that fetches and displays lyrics for currently playing
Spotify tracks with real-time monitoring and multiple lyrics sources.
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

# Third-party imports
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    import requests
    from bs4 import BeautifulSoup
    import rumps
    import tkinter as tk
    from tkinter import simpledialog, messagebox
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using:")
    print("pip install spotipy rumps requests beautifulsoup4")
    sys.exit(1)

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'audio-spotify-lyrics-menubar.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('audio-spotify-lyrics-menubar')

class CredentialManager:
    """Manages secure storage and retrieval of API credentials"""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.config_file = Path.home() / f'.{app_name.lower().replace(" ", "_")}_config.json'
    
    def load_credentials(self) -> Dict[str, Any]:
        """Load credentials from config file"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.loads(f.read())
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}
    
    def save_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Save credentials to config file"""
        try:
            with open(self.config_file, 'w') as f:
                f.write(json.dumps(credentials, indent=2))
            return True
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False
    
    def get_credential(self, key: str, prompt: str, is_secret: bool = False) -> Optional[str]:
        """Get credential with GUI prompt if not found"""
        credentials = self.load_credentials()
        
        if key in credentials and credentials[key]:
            return credentials[key]
        
        # Prompt user for credential
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        if is_secret:
            value = simpledialog.askstring(
                f"{self.app_name} Setup",
                prompt,
                parent=root,
                show='*'
            )
        else:
            value = simpledialog.askstring(
                f"{self.app_name} Setup",
                prompt,
                parent=root
            )
        
        root.destroy()
        
        if value:
            credentials[key] = value
            self.save_credentials(credentials)
            return value
        
        return None
    
    def reset_credentials(self):
        """Reset all stored credentials"""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
            logger.info("Credentials reset successfully")
        except Exception as e:
            logger.error(f"Error resetting credentials: {e}")

class LyricsProvider:
    """Handles lyrics fetching from multiple sources"""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
    
    def fetch_from_genius(self, song_title: str, artist_name: str) -> Optional[str]:
        """Fetch lyrics from Genius using web scraping"""
        try:
            # Search for song on Genius
            search_url = "https://genius.com/api/search"
            params = {
                'q': f"{artist_name} {song_title}"
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            # Find the best match
            hits = search_data.get('response', {}).get('hits', [])
            if not hits:
                return None
            
            # Get the first result's URL
            song_url = hits[0]['result']['url']
            
            # Fetch the lyrics page
            page_response = requests.get(song_url, timeout=10)
            page_response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(page_response.text, 'html.parser')
            
            # Find lyrics container (Genius uses various class names)
            lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})
            
            if lyrics_containers:
                lyrics_text = []
                for container in lyrics_containers:
                    lyrics_text.append(container.get_text(separator='\n'))
                
                lyrics = '\n'.join(lyrics_text).strip()
                if lyrics:
                    logger.info(f"Successfully fetched lyrics from Genius for: {song_title}")
                    return lyrics
            
            # Fallback: try alternative selectors
            lyrics_div = soup.find('div', class_=lambda x: x and 'lyrics' in x.lower())
            if lyrics_div:
                lyrics = lyrics_div.get_text(separator='\n').strip()
                if lyrics:
                    logger.info(f"Successfully fetched lyrics (fallback) for: {song_title}")
                    return lyrics
            
            logger.warning(f"No lyrics found on Genius for: {song_title}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching lyrics from Genius: {e}")
            return None
    
    def fetch_from_azlyrics(self, song_title: str, artist_name: str) -> Optional[str]:
        """Fetch lyrics from AZLyrics as fallback"""
        try:
            # Normalize names for AZLyrics URL format
            artist_clean = ''.join(c for c in artist_name.lower() if c.isalnum())
            song_clean = ''.join(c for c in song_title.lower() if c.isalnum())
            
            url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{song_clean}.html"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # AZLyrics stores lyrics in a div without class after comment "Sorry about that..."
            lyrics_div = soup.find('div', {'class': ''})
            if lyrics_div:
                lyrics = lyrics_div.get_text(separator='\n').strip()
                if lyrics and len(lyrics) > 50:  # Basic validation
                    logger.info(f"Successfully fetched lyrics from AZLyrics for: {song_title}")
                    return lyrics
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching lyrics from AZLyrics: {e}")
            return None

class SpotifyLyricsFetcher:
    """Main Spotify integration and lyrics fetching"""
    
    def __init__(self):
        self.credential_manager = CredentialManager("Spotify Lyrics Fetcher")
        self.lyrics_provider = LyricsProvider(self.credential_manager)
        self.spotify_client = None
        self.current_track = None
        self.last_fetch_time = None
        
        # Spotify OAuth configuration
        self.redirect_uri = "http://localhost:8888/callback"
        self.scope = "user-read-playback-state user-read-currently-playing"
        self.cache_path = "/tmp/.spotify_lyrics_cache"
        
        self.setup_spotify_client()
    
    def setup_spotify_client(self) -> bool:
        """Initialize Spotify client with OAuth"""
        try:
            # Get Spotify credentials
            client_id = self.credential_manager.get_credential(
                "SPOTIFY_CLIENT_ID",
                "Enter your Spotify Client ID:",
                False
            )
            
            client_secret = self.credential_manager.get_credential(
                "SPOTIFY_CLIENT_SECRET", 
                "Enter your Spotify Client Secret:",
                True
            )
            
            if not client_id or not client_secret:
                logger.error("Spotify credentials not provided")
                return False
            
            # Setup OAuth
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                cache_path=self.cache_path
            )
            
            self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test the connection
            user_info = self.spotify_client.current_user()
            logger.info(f"Successfully connected to Spotify for user: {user_info['display_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Spotify client: {e}")
            return False
    
    def get_current_track(self) -> Optional[Dict[str, Any]]:
        """Get currently playing track from Spotify"""
        try:
            if not self.spotify_client:
                return None
            
            playback = self.spotify_client.current_playback()
            
            if not playback or not playback.get('is_playing'):
                return None
            
            track = playback.get('item')
            if not track:
                return None
            
            track_info = {
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'duration_ms': track['duration_ms'],
                'progress_ms': playback.get('progress_ms', 0)
            }
            
            return track_info
            
        except Exception as e:
            logger.error(f"Error getting current track: {e}")
            return None
    
    def fetch_lyrics(self, song_title: str, artist_name: str) -> Optional[str]:
        """Fetch lyrics from available sources"""
        # Try Genius first
        lyrics = self.lyrics_provider.fetch_from_genius(song_title, artist_name)
        
        if not lyrics:
            # Try AZLyrics as fallback
            lyrics = self.lyrics_provider.fetch_from_azlyrics(song_title, artist_name)
        
        return lyrics
    
    def should_fetch_lyrics(self, track_info: Dict[str, Any]) -> bool:
        """Determine if we should fetch lyrics for current track"""
        if not track_info:
            return False
        
        # Check if track changed
        if not self.current_track or self.current_track['id'] != track_info['id']:
            return True
        
        # Check if enough time has passed since last fetch
        if self.last_fetch_time:
            time_since_fetch = datetime.now() - self.last_fetch_time
            if time_since_fetch < timedelta(minutes=1):
                return False
        
        return True

class SpotifyLyricsApp(rumps.App):
    """Main menu bar application"""
    
    def __init__(self):
        super(SpotifyLyricsApp, self).__init__("🎵 Spotify Lyrics")
        self.fetcher = SpotifyLyricsFetcher()
        self.current_lyrics = None
        
        # Setup menu
        self.menu = [
            "Show Current Lyrics",
            "Refresh Track Info",
            None,  # Separator
            "Reset Credentials",
            "Quit"
        ]
        
        # Start background monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background thread for monitoring Spotify"""
        self.monitoring_thread = threading.Thread(target=self.monitor_spotify, daemon=True)
        self.monitoring_thread.start()
    
    def monitor_spotify(self):
        """Background monitoring loop"""
        while True:
            try:
                track_info = self.fetcher.get_current_track()
                
                if track_info:
                    # Update menu bar title with current track
                    track_display = f"🎵 {track_info['name'][:25]}"
                    if len(track_info['name']) > 25:
                        track_display += "..."
                    self.title = track_display
                    
                    # Check if we should fetch lyrics
                    if self.fetcher.should_fetch_lyrics(track_info):
                        self.fetcher.current_track = track_info
                        self.fetcher.last_fetch_time = datetime.now()
                        
                        # Fetch lyrics in background
                        lyrics_thread = threading.Thread(
                            target=self.fetch_lyrics_background,
                            args=(track_info['name'], track_info['artist']),
                            daemon=True
                        )
                        lyrics_thread.start()
                else:
                    self.title = "🎵 No Track Playing"
                    self.current_lyrics = None
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait longer if there's an error
    
    def fetch_lyrics_background(self, song_title: str, artist_name: str):
        """Fetch lyrics in background thread"""
        try:
            lyrics = self.fetcher.fetch_lyrics(song_title, artist_name)
            self.current_lyrics = lyrics
        except Exception as e:
            logger.error(f"Error fetching lyrics in background: {e}")
    
    @rumps.clicked("Show Current Lyrics")
    def show_lyrics(self, _):
        """Display lyrics for current track"""
        if not self.fetcher.current_track:
            rumps.notification(
                "No Track Playing",
                "Please play a song in Spotify",
                ""
            )
            return
        
        track = self.fetcher.current_track
        
        if self.current_lyrics:
            # Show first part of lyrics in notification
            preview = self.current_lyrics[:200]
            if len(self.current_lyrics) > 200:
                preview += "..."
            
            rumps.notification(
                f"Lyrics: {track['name']}",
                f"by {track['artist']}",
                preview
            )
        else:
            # Try to fetch lyrics now if not available
            rumps.notification(
                "Fetching Lyrics...",
                f"Looking up lyrics for {track['name']}",
                "This may take a moment"
            )
            
            # Fetch in background
            lyrics_thread = threading.Thread(
                target=self.fetch_and_show_lyrics,
                args=(track['name'], track['artist']),
                daemon=True
            )
            lyrics_thread.start()
    
    def fetch_and_show_lyrics(self, song_title: str, artist_name: str):
        """Fetch lyrics and show result"""
        try:
            lyrics = self.fetcher.fetch_lyrics(song_title, artist_name)
            
            if lyrics:
                self.current_lyrics = lyrics
                preview = lyrics[:200]
                if len(lyrics) > 200:
                    preview += "..."
                
                rumps.notification(
                    f"Lyrics: {song_title}",
                    f"by {artist_name}",
                    preview
                )
            else:
                rumps.notification(
                    "Lyrics Not Found",
                    f"{song_title} by {artist_name}",
                    "Could not find lyrics for this track"
                )
        except Exception as e:
            logger.error(f"Error fetching lyrics: {e}")
            rumps.notification(
                "Error",
                "Failed to fetch lyrics",
                str(e)
            )
    
    @rumps.clicked("Refresh Track Info")
    def refresh_track(self, _):
        """Manually refresh current track info"""
        try:
            track_info = self.fetcher.get_current_track()
            
            if track_info:
                self.fetcher.current_track = track_info
                self.current_lyrics = None  # Clear cached lyrics
                
                rumps.notification(
                    "Track Refreshed",
                    f"Now playing: {track_info['name']}",
                    f"by {track_info['artist']}"
                )
            else:
                rumps.notification(
                    "No Track Playing",
                    "Spotify is not currently playing",
                    ""
                )
        except Exception as e:
            logger.error(f"Error refreshing track: {e}")
            rumps.notification(
                "Error",
                "Failed to refresh track info",
                str(e)
            )
    
    @rumps.clicked("Reset Credentials")
    def reset_credentials(self, _):
        """Reset stored API credentials"""
        try:
            self.fetcher.credential_manager.reset_credentials()
            
            rumps.notification(
                "Credentials Reset",
                "All stored credentials have been cleared",
                "You'll be prompted for new credentials on next use"
            )
            
            logger.info("Credentials reset by user")
            
        except Exception as e:
            logger.error(f"Error resetting credentials: {e}")
            rumps.notification(
                "Error",
                "Failed to reset credentials",
                str(e)
            )
    
    @rumps.clicked("Quit")
    def quit_app(self, _):
        """Quit the application"""
        logger.info("Application shutting down")
        rumps.quit_application()

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Spotify Lyrics Menu Bar App"
    )
    
    parser.add_argument(
        '--reset-credentials',
        action='store_true',
        help='Reset stored API credentials and exit'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Handle credential reset
        if args.reset_credentials:
            credential_manager = CredentialManager("Spotify Lyrics Fetcher")
            credential_manager.reset_credentials()
            print("Credentials reset successfully")
            return 0
        
        # Create and run the app
        app = SpotifyLyricsApp()
        
        # Check if Spotify client setup was successful
        if not app.fetcher.spotify_client:
            logger.error("Failed to setup Spotify client. Exiting.")
            messagebox.showerror(
                "Setup Failed",
                "Failed to setup Spotify client. Please check your credentials and try again."
            )
            return 1
        
        logger.info("Starting Spotify Lyrics menu bar app")
        app.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())