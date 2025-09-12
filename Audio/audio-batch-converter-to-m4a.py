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
# Script Name: audio-batch-converter-to-m4a.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-performance batch audio file converter optimized for M4A output.    
#              Converts various audio formats (AIFF, MP3, CAFF, WAV, AMR, OGG, FLAC)      
#              to M4A while preserving metadata using FFmpeg. Features multi-core        
#              processing and automatic year-based organization.                                               
#
# Usage: python audio-batch-converter-to-m4a.py [--source DIR] [--destination DIR] 
#
# Dependencies: ffmpeg, mutagen, tkinter                                           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Requires FFmpeg to be installed on the system. Uses all available CPU       
#        cores for optimal performance. Automatically organizes output files by   
#        year based on filename pattern detection.                                                    
#                                                                                
####################################################################################
#
#
#
#

"""
Audio Batch Converter to M4A

High-performance audio file converter specifically optimized for M4A output with
multi-core processing, metadata preservation, and automatic organization features.
"""

import os
import sys
import logging
import multiprocessing
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from mutagen import File
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Please install required packages using:")
    print("pip install mutagen")
    sys.exit(1)

# Configure logging
desktop_path = Path.home() / 'Desktop'
log_file = desktop_path / 'audio-batch-converter-to-m4a.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('audio-batch-converter-to-m4a')

class AudioConverter:
    """High-performance M4A audio converter with multi-core processing"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.worker_count = max(1, self.cpu_count - 1)  # Leave 1 core free
        
        # Supported audio formats
        self.supported_formats = {
            '.aiff', '.m4a', '.mp3', '.caff', '.wav', 
            '.amr', '.ogg', '.flac', '.aac', '.wma'
        }
        
        logger.info(f"Initialized converter with {self.worker_count} workers")
        
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available on the system"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def select_directory(self, title: str) -> Optional[Path]:
        """Select directory using native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        directory = filedialog.askdirectory(title=title)
        root.destroy()
        
        return Path(directory) if directory else None
    
    def gather_audio_files(self, source_dir: Path) -> List[Path]:
        """Recursively gather all supported audio files"""
        audio_files = []
        
        for file_path in source_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                # Skip files that are already M4A
                if file_path.suffix.lower() != '.m4a':
                    audio_files.append(file_path)
        
        logger.info(f"Found {len(audio_files)} audio files to convert")
        return audio_files
    
    def extract_year_from_filename(self, file_path: Path) -> str:
        """Extract year from filename or use current year"""
        try:
            # Try to extract 4-digit year from beginning of filename
            year = file_path.stem[:4]
            if year.isdigit() and 1900 <= int(year) <= datetime.now().year + 1:
                return year
        except (ValueError, IndexError):
            pass
        
        # Fallback to current year
        return str(datetime.now().year)
    
    def convert_single_file(self, args: Tuple[Path, Path]) -> Tuple[bool, str]:
        """Convert a single audio file to M4A"""
        input_file, base_output_dir = args
        
        try:
            # Determine output directory based on year
            year = self.extract_year_from_filename(input_file)
            output_dir = base_output_dir / year
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output path
            output_path = output_dir / f"{input_file.stem}.m4a"
            
            # Handle duplicates with counter
            counter = 1
            while output_path.exists():
                base_name = input_file.stem.split('_')[0] if '_' in input_file.stem else input_file.stem
                output_path = output_dir / f"{base_name}_{counter}.m4a"
                counter += 1
            
            # FFmpeg command for high-quality M4A conversion
            cmd = [
                'ffmpeg',
                '-i', str(input_file),
                '-map', '0:a:0',  # First audio stream
                '-map_metadata', '0',  # Copy metadata
                '-c:a', 'aac',  # AAC codec
                '-b:a', '256k',  # 256k bitrate
                '-ar', '48000',  # 48kHz sample rate
                '-af', 'aresample=async=1000',  # Audio filtering
                '-threads', str(self.worker_count),
                '-y',  # Overwrite output files
                str(output_path)
            ]
            
            # Execute conversion
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully converted: {input_file.name}")
                return True, f"Converted: {input_file.name}"
            else:
                error_msg = f"FFmpeg error for {input_file.name}: {result.stderr}"
                logger.error(error_msg)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout converting {input_file.name}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to convert {input_file.name}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_files(self, source_dir: Path, dest_dir: Path) -> Dict[str, Any]:
        """Process all audio files using multi-core conversion"""
        # Gather files
        audio_files = self.gather_audio_files(source_dir)
        
        if not audio_files:
            logger.warning("No audio files found for conversion")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'errors': []
            }
        
        logger.info(f"Starting conversion of {len(audio_files)} files")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Destination: {dest_dir}")
        logger.info(f"Using {self.worker_count} worker processes")
        
        # Prepare arguments for parallel processing
        conversion_args = [(file_path, dest_dir) for file_path in audio_files]
        
        successful = 0
        failed = 0
        errors = []
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
            results = executor.map(self.convert_single_file, conversion_args)
            
            for i, (success, message) in enumerate(results, 1):
                if success:
                    successful += 1
                else:
                    failed += 1
                    errors.append(message)
                
                # Log progress every 10 files
                if i % 10 == 0 or i == len(audio_files):
                    logger.info(f"Progress: {i}/{len(audio_files)} files processed")
        
        return {
            'total': len(audio_files),
            'successful': successful,
            'failed': failed,
            'errors': errors
        }
    
    def run_interactive(self):
        """Run interactive mode with GUI dialogs"""
        logger.info("Starting interactive audio conversion...")
        
        # Check FFmpeg availability
        if not self.check_ffmpeg():
            error_msg = "FFmpeg not found. Please install FFmpeg to use this converter."
            logger.error(error_msg)
            messagebox.showerror("FFmpeg Required", error_msg)
            return
        
        # Select source directory
        source_dir = self.select_directory("Select Source Directory (containing audio files)")
        if not source_dir or not source_dir.exists():
            logger.info("No source directory selected")
            return
        
        # Select destination directory
        dest_dir = self.select_directory("Select Destination Directory (for converted M4A files)")
        if not dest_dir:
            logger.info("No destination directory selected")
            return
        
        # Ensure destination directory exists
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        try:
            results = self.process_files(source_dir, dest_dir)
            
            # Log summary
            summary = f"""
Conversion Summary:
- Total files: {results['total']}
- Successfully converted: {results['successful']}
- Failed conversions: {results['failed']}
- Source: {source_dir}
- Destination: {dest_dir}
"""
            logger.info(summary)
            
            # Show completion dialog
            if results['failed'] > 0:
                messagebox.showwarning(
                    "Conversion Completed with Errors",
                    f"Converted {results['successful']} of {results['total']} files.\n"
                    f"{results['failed']} files failed. Check log for details."
                )
            else:
                messagebox.showinfo(
                    "Conversion Completed",
                    f"Successfully converted all {results['total']} files!"
                )
                
        except Exception as e:
            error_msg = f"Conversion process failed: {e}"
            logger.error(error_msg)
            messagebox.showerror("Conversion Error", error_msg)

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="High-performance batch audio converter to M4A format"
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        help='Source directory containing audio files'
    )
    
    parser.add_argument(
        '--destination', '-d',
        type=str,
        help='Destination directory for converted M4A files'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of worker processes (default: CPU cores - 1)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        converter = AudioConverter()
        
        # Override worker count if specified
        if args.workers:
            converter.worker_count = max(1, min(args.workers, converter.cpu_count))
            logger.info(f"Using {converter.worker_count} workers (user specified)")
        
        # Command line mode
        if args.source and args.destination:
            source_dir = Path(args.source)
            dest_dir = Path(args.destination)
            
            # Validate paths
            if not source_dir.exists() or not source_dir.is_dir():
                logger.error(f"Source directory does not exist: {source_dir}")
                return 1
            
            # Ensure destination exists
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Check FFmpeg
            if not converter.check_ffmpeg():
                logger.error("FFmpeg not found. Please install FFmpeg.")
                return 1
            
            # Process files
            results = converter.process_files(source_dir, dest_dir)
            
            # Print summary
            print(f"\nConversion Summary:")
            print(f"Total files: {results['total']}")
            print(f"Successfully converted: {results['successful']}")
            print(f"Failed conversions: {results['failed']}")
            
            if results['errors']:
                print(f"\nErrors:")
                for error in results['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(results['errors']) > 5:
                    print(f"  ... and {len(results['errors']) - 5} more errors")
            
            return 0 if results['failed'] == 0 else 1
        
        # Interactive mode
        else:
            converter.run_interactive()
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