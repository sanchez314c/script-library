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
# Script Name: audio-apple-podcasts-exporter.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Exports downloaded Apple Podcasts episodes to organized directory    
#              structure with preserved metadata and descriptive naming. Integrates      
#              with Apple Podcasts SQLite database and maintains ID3 tags for        
#              maximum compatibility.                                               
#
# Usage: python audio-apple-podcasts-exporter.py [--output DIR] 
#
# Dependencies: mutagen, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Accesses Apple Podcasts database in user's Library folder. Creates       
#        organized podcast directory structure with proper metadata preservation.   
#        Handles corrupted files and filename sanitization automatically.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
Apple Podcasts Exporter

Exports downloaded Apple Podcasts episodes to organized directory structure with
preserved metadata, descriptive naming, and comprehensive error handling.
"""

import os
import sys
import shutil
import urllib.parse
import sqlite3
import datetime
import logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple, Dict, Any

# Third-party imports
try:
    from mutagen.mp3 import MP3, HeaderNotFoundError
    from mutagen.easyid3 import EasyID3
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using:")
    print("pip install mutagen")
    sys.exit(1)

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'audio-apple-podcasts-exporter.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('audio-apple-podcasts-exporter')

class ApplePodcastsExporter:
    """Apple Podcasts episode exporter with metadata preservation"""
    
    def __init__(self):
        # Apple Podcasts database path
        self.db_path = Path.home() / "Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Documents/MTLibrary.sqlite"
        
        # Progress tracking
        self.progress_window = None
        self.progress_var = None
        self.status_var = None
        
        logger.info("Initialized Apple Podcasts exporter")
    
    def check_database_access(self) -> bool:
        """Check if Apple Podcasts database is accessible"""
        if not self.db_path.exists():
            logger.error(f"Apple Podcasts database not found at: {self.db_path}")
            return False
        
        try:
            # Test database connection
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                table_count = cursor.fetchone()[0]
                logger.info(f"Found {table_count} tables in Apple Podcasts database")
                return True
        except sqlite3.Error as e:
            logger.error(f"Cannot access Apple Podcasts database: {e}")
            return False
    
    def select_directory(self, title: str) -> Optional[Path]:
        """Select directory using native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        directory = filedialog.askdirectory(title=title)
        root.destroy()
        
        return Path(directory) if directory else None
    
    def create_progress_window(self, title: str, max_value: int):
        """Create progress tracking window"""
        self.progress_window = tk.Toplevel()
        self.progress_window.title(title)
        self.progress_window.geometry("450x150")
        self.progress_window.resizable(False, False)
        self.progress_window.attributes('-topmost', True)
        
        # Center window
        self.progress_window.update_idletasks()
        width = self.progress_window.winfo_width()
        height = self.progress_window.winfo_height()
        x = (self.progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.progress_window.winfo_screenheight() // 2) - (height // 2)
        self.progress_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(self.progress_window, textvariable=self.status_var)
        status_label.pack(pady=20)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            self.progress_window,
            variable=self.progress_var,
            maximum=max_value,
            length=400
        )
        progress_bar.pack(pady=10)
        
        self.progress_window.update()
    
    def update_progress(self, current: int, status: str = None):
        """Update progress window"""
        if self.progress_window and self.progress_var:
            self.progress_var.set(current)
            if status and self.status_var:
                self.status_var.set(status)
            self.progress_window.update()
    
    def close_progress_window(self):
        """Close progress window"""
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
    
    def get_downloaded_episodes(self) -> List[Tuple]:
        """Fetch downloaded episodes from Apple Podcasts database"""
        query = """
        SELECT 
            p.ZAUTHOR as author,
            p.ZTITLE as podcast_title,
            e.ZTITLE as episode_title,
            e.ZASSETURL as asset_url,
            e.ZPUBDATE as pub_date,
            e.ZDURATION as duration,
            e.ZEPISODEDESCRIPTION as description
        FROM ZMTEPISODE e 
        JOIN ZMTPODCAST p ON e.ZPODCASTUUID = p.ZUUID 
        WHERE e.ZASSETURL IS NOT NULL
        AND e.ZASSETURL != ''
        ORDER BY p.ZTITLE, e.ZPUBDATE DESC;
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                episodes = cursor.fetchall()
                logger.info(f"Found {len(episodes)} downloaded episodes")
                return episodes
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def sanitize_filename(self, text: str, max_length: int = 100) -> str:
        """Sanitize text for safe filename usage"""
        if not text:
            return "Unknown"
        
        # Replace problematic characters
        replacements = {
            '/': '|',
            ':': ',',
            '<': '(',
            '>': ')',
            '"': "'",
            '\\': '|',
            '|': '-',
            '?': '',
            '*': '',
            '\n': ' ',
            '\r': ' ',
            '\t': ' '
        }
        
        sanitized = text
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        
        # Remove multiple spaces and strip
        sanitized = ' '.join(sanitized.split())
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rsplit(' ', 1)[0] + '...'
        
        return sanitized.strip()
    
    def convert_apple_timestamp(self, apple_timestamp: float) -> datetime.datetime:
        """Convert Apple's timestamp format to Python datetime"""
        # Apple uses seconds since January 1, 2001 (Mac epoch)
        mac_epoch = datetime.datetime(2001, 1, 1)
        return mac_epoch + datetime.timedelta(seconds=apple_timestamp)
    
    def export_episode(self, episode_data: Tuple, output_dir: Path) -> Tuple[bool, str]:
        """Export a single podcast episode with metadata"""
        try:
            author, podcast_title, episode_title, asset_url, pub_date, duration, description = episode_data
            
            # Sanitize names for filesystem
            safe_author = self.sanitize_filename(author or "Unknown Author", 50)
            safe_podcast = self.sanitize_filename(podcast_title or "Unknown Podcast", 80)
            safe_episode = self.sanitize_filename(episode_title or "Unknown Episode", 120)
            
            # Convert publication date
            if pub_date:
                pub_datetime = self.convert_apple_timestamp(pub_date)
                date_prefix = pub_datetime.strftime("%Y.%m.%d")
            else:
                date_prefix = "Unknown.Date"
            
            # Create podcast directory
            podcast_dir = output_dir / safe_podcast
            podcast_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with date prefix
            filename = f"{date_prefix} - {safe_episode} - ({safe_author}).mp3"
            dest_path = podcast_dir / filename
            
            # Handle filename collisions
            counter = 1
            while dest_path.exists():
                base_filename = f"{date_prefix} - {safe_episode} - ({safe_author})_{counter}.mp3"
                dest_path = podcast_dir / base_filename
                counter += 1
            
            # Get source file path from URL
            if not asset_url or not asset_url.startswith('file://'):
                return False, f"Invalid asset URL for episode: {episode_title}"
            
            source_path = Path(urllib.parse.unquote(asset_url[7:]))  # Remove 'file://' prefix
            
            if not source_path.exists():
                return False, f"Source file not found: {source_path}"
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            # Update metadata with ID3 tags
            try:
                audio_file = MP3(dest_path, ID3=EasyID3)
                if audio_file.tags is None:
                    audio_file.add_tags()
                
                # Set basic tags
                audio_file.tags['artist'] = author or "Unknown Author"
                audio_file.tags['album'] = podcast_title or "Unknown Podcast"
                audio_file.tags['title'] = episode_title or "Unknown Episode"
                audio_file.tags['genre'] = "Podcast"
                
                # Add date if available
                if pub_date:
                    audio_file.tags['date'] = pub_datetime.strftime("%Y-%m-%d")
                
                # Add album artist (podcast author)
                audio_file.tags['albumartist'] = author or "Unknown Author"
                
                audio_file.save()
                
                logger.info(f"Exported: {safe_podcast} - {safe_episode}")
                return True, f"Successfully exported: {episode_title}"
                
            except HeaderNotFoundError:
                # File might be corrupted or not a valid MP3
                logger.warning(f"Corrupted or invalid MP3 file: {episode_title}")
                return False, f"Corrupted file (kept copy): {episode_title}"
            except Exception as e:
                logger.warning(f"Metadata update failed for {episode_title}: {e}")
                return True, f"Exported without metadata: {episode_title}"
            
        except Exception as e:
            error_msg = f"Failed to export {episode_title}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def export_all_episodes(self, output_dir: Path) -> Dict[str, Any]:
        """Export all downloaded episodes"""
        # Get episodes from database
        episodes = self.get_downloaded_episodes()
        
        if not episodes:
            logger.warning("No downloaded episodes found")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'errors': []
            }
        
        logger.info(f"Starting export of {len(episodes)} episodes to {output_dir}")
        
        # Create progress window
        self.create_progress_window("Exporting Podcasts", len(episodes))
        
        successful = 0
        failed = 0
        errors = []
        
        # Export episodes one by one
        for i, episode in enumerate(episodes, 1):
            episode_title = episode[2] if episode[2] else "Unknown Episode"
            
            # Update progress
            self.update_progress(i, f"Exporting: {episode_title[:50]}...")
            
            # Export episode
            success, message = self.export_episode(episode, output_dir)
            
            if success:
                successful += 1
            else:
                failed += 1
                errors.append(message)
        
        self.close_progress_window()
        
        return {
            'total': len(episodes),
            'successful': successful,
            'failed': failed,
            'errors': errors
        }
    
    def run_interactive(self):
        """Run interactive mode with GUI dialogs"""
        logger.info("Starting interactive Apple Podcasts export...")
        
        # Check database access
        if not self.check_database_access():
            messagebox.showerror(
                "Database Not Found",
                "Apple Podcasts database not found or not accessible.\n\n"
                "Please ensure:\n"
                "1. Apple Podcasts app is installed\n"
                "2. You have downloaded some episodes\n"
                "3. This script has necessary permissions"
            )
            return
        
        # Get episode count for user information
        episodes = self.get_downloaded_episodes()
        if not episodes:
            messagebox.showinfo(
                "No Episodes Found",
                "No downloaded podcast episodes were found in Apple Podcasts."
            )
            return
        
        # Show episode count and get confirmation
        if not messagebox.askyesno(
            "Export Confirmation",
            f"Found {len(episodes)} downloaded episodes.\n\n"
            f"Export all episodes to organized folders?"
        ):
            logger.info("User cancelled export")
            return
        
        # Select output directory
        output_dir = self.select_directory("Select Export Directory (for organized podcasts)")
        if not output_dir:
            logger.info("No output directory selected")
            return
        
        try:
            # Export all episodes
            results = self.export_all_episodes(output_dir)
            
            # Show summary
            summary_msg = (
                f"Export completed!\n\n"
                f"Total episodes: {results['total']}\n"
                f"Successfully exported: {results['successful']}\n"
                f"Failed exports: {results['failed']}\n\n"
                f"Episodes exported to: {output_dir}"
            )
            
            if results['failed'] > 0:
                summary_msg += f"\n\nSome episodes failed to export. Check log for details."
                messagebox.showwarning("Export Completed with Errors", summary_msg)
            else:
                messagebox.showinfo("Export Completed", summary_msg)
            
            # Log final summary
            logger.info(f"""
Export Summary:
- Output directory: {output_dir}
- Total episodes: {results['total']}
- Successfully exported: {results['successful']}
- Failed exports: {results['failed']}
""")
            
        except Exception as e:
            error_msg = f"Export process failed: {e}"
            logger.error(error_msg)
            messagebox.showerror("Export Error", error_msg)
        finally:
            self.close_progress_window()

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export Apple Podcasts episodes to organized directories"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for exported podcasts'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List available episodes without exporting'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        exporter = ApplePodcastsExporter()
        
        # Check database access first
        if not exporter.check_database_access():
            print("Error: Cannot access Apple Podcasts database.")
            print("Please ensure Apple Podcasts is installed and you have downloaded episodes.")
            return 1
        
        # List mode
        if args.list_only:
            episodes = exporter.get_downloaded_episodes()
            
            if not episodes:
                print("No downloaded episodes found.")
                return 0
            
            print(f"\nFound {len(episodes)} downloaded episodes:\n")
            
            current_podcast = None
            for episode in episodes:
                author, podcast_title, episode_title, _, pub_date, duration, _ = episode
                
                # Group by podcast
                if podcast_title != current_podcast:
                    current_podcast = podcast_title
                    print(f"\n{podcast_title} (by {author}):")
                    print("-" * 50)
                
                # Format publication date
                if pub_date:
                    pub_datetime = exporter.convert_apple_timestamp(pub_date)
                    date_str = pub_datetime.strftime("%Y-%m-%d")
                else:
                    date_str = "Unknown Date"
                
                print(f"  {date_str}: {episode_title}")
            
            return 0
        
        # Command line export mode
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = exporter.export_all_episodes(output_dir)
            
            print(f"\nExport Summary:")
            print(f"Total episodes: {results['total']}")
            print(f"Successfully exported: {results['successful']}")
            print(f"Failed exports: {results['failed']}")
            
            if results['errors']:
                print(f"\nErrors:")
                for error in results['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(results['errors']) > 5:
                    print(f"  ... and {len(results['errors']) - 5} more errors")
            
            return 0 if results['failed'] == 0 else 1
        
        # Interactive mode
        else:
            exporter.run_interactive()
            return 0
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())