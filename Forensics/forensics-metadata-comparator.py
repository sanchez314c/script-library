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
# Script Name: forensics-metadata-comparator.py                                  
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced forensic metadata comparison and analysis tool for      
#              detecting file manipulation, timeline inconsistencies, and       
#              evidence tampering through comprehensive metadata examination.   
#
# Usage: python forensics-metadata-comparator.py [--input PATH] [--output DIR]  
#
# Dependencies: exifread, python-magic, pillow, dateutil, pandas              
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional metadata forensics with timeline analysis, anomaly       
#        detection, and evidence integrity verification for investigations.    
#                                                                                
####################################################################################

"""
Advanced Forensic Metadata Comparator
====================================

Comprehensive metadata analysis tool for digital forensics investigations.
Performs metadata extraction, comparison, timeline analysis, and anomaly
detection across multiple file types with evidence integrity preservation.
"""

import os
import sys
import logging
import multiprocessing
import subprocess
import argparse
import time
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import statistics
from datetime import datetime, timedelta

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "exifread>=2.3.2",
        "python-magic>=0.4.24",
        "Pillow>=8.0.0",
        "python-dateutil>=2.8.0",
        "pandas>=1.3.0"
    ]
    
    missing = []
    for pkg in required_packages:
        name = pkg.split('>=')[0].replace('-', '_')
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

check_and_install_dependencies()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ExifTags
import exifread
from dateutil import parser as date_parser
import pandas as pd

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

IS_MACOS = sys.platform == "darwin"

class ForensicMetadataComparator:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        # Metadata extraction handlers for different file types
        self.extractors = {
            'image': self.extract_image_metadata,
            'document': self.extract_document_metadata,
            'video': self.extract_video_metadata,
            'audio': self.extract_audio_metadata,
            'archive': self.extract_archive_metadata
        }
        
        # File type categories
        self.file_categories = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.raw'},
            'document': {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.rtf'},
            'video': {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v'},
            'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'},
            'archive': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'}
        }
        
        self.analysis_stats = {
            'total_files': 0,
            'processed_files': 0,
            'metadata_extracted': 0,
            'anomalies_found': 0,
            'timeline_inconsistencies': 0,
            'start_time': None
        }
        
        self.metadata_results = []
        self.timeline_data = []
        self.anomalies = []
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "forensics-metadata-comparator.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
    def select_input_path(self):
        """Select input directory via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        directory = filedialog.askdirectory(
            title="Select Directory for Metadata Analysis"
        )
        
        root.destroy()
        return Path(directory) if directory else None
        
    def select_output_directory(self):
        """Select output directory via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        directory = filedialog.askdirectory(
            title="Select Output Directory for Analysis Reports"
        )
        
        root.destroy()
        return Path(directory) if directory else None
        
    def create_progress_window(self, total_files):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Forensic Metadata Analysis")
        self.progress_root.geometry("500x150")
        self.progress_root.resizable(False, False)
        
        # Center window
        x = (self.progress_root.winfo_screenwidth() // 2) - 250
        y = (self.progress_root.winfo_screenheight() // 2) - 75
        self.progress_root.geometry(f"+{x}+{y}")
        
        if IS_MACOS:
            self.progress_root.attributes("-topmost", True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_root,
            variable=self.progress_var,
            maximum=total_files,
            length=450
        )
        self.progress_bar.pack(pady=20)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing metadata analysis...")
        self.status_label = tk.Label(
            self.progress_root,
            textvariable=self.status_var,
            font=("Helvetica", 11)
        )
        self.status_label.pack(pady=(0, 20))
        
        self.progress_root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.progress_root.update()
        
    def update_progress(self, current, total, filename=""):
        """Update progress bar and status"""
        if hasattr(self, 'progress_var'):
            self.progress_var.set(current)
            
            percentage = int((current / total) * 100)
            status = f"Analyzing {current}/{total} ({percentage}%)"
            if filename:
                status += f" - {filename}"
                
            self.status_var.set(status)
            self.progress_root.update()
            
    def get_file_category(self, file_path):
        """Determine file category based on extension"""
        ext = file_path.suffix.lower()
        
        for category, extensions in self.file_categories.items():
            if ext in extensions:
                return category
                
        return 'other'
        
    def extract_basic_metadata(self, file_path):
        """Extract basic file system metadata"""
        try:
            stat = file_path.stat()
            
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'extension': file_path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created_time': datetime.fromtimestamp(stat.st_ctime),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'accessed_time': datetime.fromtimestamp(stat.st_atime),
                'permissions': oct(stat.st_mode)[-3:],
                'file_hash': self.calculate_file_hash(file_path)
            }
            
            # Add MIME type if available
            if MAGIC_AVAILABLE:
                try:
                    metadata['mime_type'] = magic.from_file(str(file_path), mime=True)
                    metadata['file_type'] = magic.from_file(str(file_path))
                except Exception:
                    pass
                    
            return metadata
            
        except Exception as e:
            self.logger.error(f"Basic metadata extraction error for {file_path}: {e}")
            return None
            
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
            
    def extract_image_metadata(self, file_path):
        """Extract comprehensive image metadata"""
        metadata = {}
        
        try:
            # PIL metadata
            with Image.open(file_path) as img:
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                metadata['size'] = img.size
                metadata['width'] = img.width
                metadata['height'] = img.height
                
                # EXIF data via PIL
                exifdata = img.getexif()
                if exifdata:
                    pil_exif = {}
                    for tag_id, value in exifdata.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        pil_exif[tag] = str(value)
                    metadata['pil_exif'] = pil_exif
                    
            # Detailed EXIF via exifread
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                
            exif_data = {}
            timestamps = {}
            
            for tag in tags.keys():
                if tag not in ('JPEGThumbnail', 'TIFFThumbnail'):
                    try:
                        value = str(tags[tag])
                        exif_data[tag] = value
                        
                        # Extract timestamp information
                        if 'datetime' in tag.lower() or 'date' in tag.lower():
                            try:
                                parsed_date = date_parser.parse(value, fuzzy=True)
                                timestamps[tag] = parsed_date.isoformat()
                            except:
                                timestamps[tag] = value
                                
                    except:
                        exif_data[tag] = 'Unable to read'
                        
            if exif_data:
                metadata['exif'] = exif_data
            if timestamps:
                metadata['timestamps'] = timestamps
                
        except Exception as e:
            self.logger.error(f"Image metadata extraction error for {file_path}: {e}")
            metadata['error'] = str(e)
            
        return metadata
        
    def extract_document_metadata(self, file_path):
        """Extract document metadata (basic implementation)"""
        metadata = {}
        
        try:
            # For PDFs and Office documents, this would require additional libraries
            # This is a placeholder for basic document analysis
            
            if file_path.suffix.lower() == '.pdf':
                # Would use PyPDF2 or similar for PDF metadata
                metadata['document_type'] = 'PDF'
                
            elif file_path.suffix.lower() in ['.doc', '.docx']:
                # Would use python-docx for Word documents
                metadata['document_type'] = 'Microsoft Word'
                
            elif file_path.suffix.lower() in ['.xls', '.xlsx']:
                # Would use openpyxl for Excel documents
                metadata['document_type'] = 'Microsoft Excel'
                
            else:
                metadata['document_type'] = 'Text/Other'
                
        except Exception as e:
            self.logger.error(f"Document metadata extraction error for {file_path}: {e}")
            metadata['error'] = str(e)
            
        return metadata
        
    def extract_video_metadata(self, file_path):
        """Extract video metadata (placeholder)"""
        metadata = {}
        
        try:
            # Would use ffprobe or similar for video metadata
            metadata['media_type'] = 'Video'
            metadata['format'] = file_path.suffix.lower()
            
        except Exception as e:
            self.logger.error(f"Video metadata extraction error for {file_path}: {e}")
            metadata['error'] = str(e)
            
        return metadata
        
    def extract_audio_metadata(self, file_path):
        """Extract audio metadata (placeholder)"""
        metadata = {}
        
        try:
            # Would use mutagen or similar for audio metadata
            metadata['media_type'] = 'Audio'
            metadata['format'] = file_path.suffix.lower()
            
        except Exception as e:
            self.logger.error(f"Audio metadata extraction error for {file_path}: {e}")
            metadata['error'] = str(e)
            
        return metadata
        
    def extract_archive_metadata(self, file_path):
        """Extract archive metadata (placeholder)"""
        metadata = {}
        
        try:
            # Would analyze archive contents and structure
            metadata['archive_type'] = 'Archive'
            metadata['format'] = file_path.suffix.lower()
            
        except Exception as e:
            self.logger.error(f"Archive metadata extraction error for {file_path}: {e}")
            metadata['error'] = str(e)
            
        return metadata
        
    def analyze_metadata(self, file_path):
        """Perform comprehensive metadata analysis"""
        try:
            analysis = {
                'file_path': str(file_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'basic_metadata': {},
                'specific_metadata': {},
                'anomalies': [],
                'timeline_data': {},
                'risk_score': 0
            }
            
            # Extract basic metadata
            basic_meta = self.extract_basic_metadata(file_path)
            if basic_meta:
                analysis['basic_metadata'] = basic_meta
                
                # Add to timeline data
                timeline_entry = {
                    'file_path': str(file_path),
                    'created': basic_meta['created_time'],
                    'modified': basic_meta['modified_time'],
                    'accessed': basic_meta['accessed_time']
                }
                analysis['timeline_data'] = timeline_entry
                self.timeline_data.append(timeline_entry)
                
            # Extract format-specific metadata
            file_category = self.get_file_category(file_path)
            
            if file_category in self.extractors:
                specific_meta = self.extractors[file_category](file_path)
                if specific_meta:
                    analysis['specific_metadata'] = specific_meta
                    
            # Detect anomalies
            anomalies = self.detect_metadata_anomalies(analysis)
            analysis['anomalies'] = anomalies
            
            # Calculate risk score
            risk_score = len(anomalies) * 20
            analysis['risk_score'] = min(100, risk_score)
            
            # Update stats
            self.analysis_stats['processed_files'] += 1
            if basic_meta or analysis['specific_metadata']:
                self.analysis_stats['metadata_extracted'] += 1
            if anomalies:
                self.analysis_stats['anomalies_found'] += len(anomalies)
                
            self.logger.info(f"✓ Analyzed: {file_path.name} (Risk: {risk_score}/100)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"✗ Metadata analysis failed for {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
            
    def detect_metadata_anomalies(self, analysis):
        """Detect anomalies in metadata"""
        anomalies = []
        
        try:
            basic = analysis.get('basic_metadata', {})
            specific = analysis.get('specific_metadata', {})
            
            # Timeline anomalies
            if basic:
                created = basic.get('created_time')
                modified = basic.get('modified_time')
                accessed = basic.get('accessed_time')
                
                if created and modified:
                    if modified < created:
                        anomalies.append("Modified time before created time")
                        
                if accessed and created:
                    if accessed < created:
                        anomalies.append("Accessed time before created time")
                        
            # EXIF timestamp anomalies for images
            if 'timestamps' in specific:
                timestamps = specific['timestamps']
                exif_times = []
                
                for key, value in timestamps.items():
                    try:
                        if isinstance(value, str) and value != 'Unable to read':
                            parsed_time = date_parser.parse(value, fuzzy=True)
                            exif_times.append(parsed_time)
                    except:
                        continue
                        
                # Check for inconsistent EXIF timestamps
                if len(exif_times) > 1:
                    time_diffs = []
                    for i in range(len(exif_times)):
                        for j in range(i + 1, len(exif_times)):
                            diff = abs((exif_times[i] - exif_times[j]).total_seconds())
                            time_diffs.append(diff)
                            
                    if time_diffs and max(time_diffs) > 86400:  # More than 1 day difference
                        anomalies.append("Large discrepancies in EXIF timestamps")
                        
                # Compare EXIF timestamps with file system timestamps
                if exif_times and basic:
                    fs_created = basic.get('created_time')
                    fs_modified = basic.get('modified_time')
                    
                    for exif_time in exif_times:
                        if fs_created and abs((exif_time - fs_created).total_seconds()) > 3600:
                            anomalies.append("EXIF timestamp differs significantly from file system")
                            break
                            
            # Size anomalies
            if basic and basic.get('size_bytes') == 0:
                anomalies.append("Zero-byte file")
                
            # Extension mismatch
            if basic and MAGIC_AVAILABLE:
                file_ext = basic.get('extension', '').lower()
                mime_type = basic.get('mime_type', '').lower()
                
                extension_mime_map = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.pdf': 'application/pdf',
                    '.txt': 'text/plain'
                }
                
                if file_ext in extension_mime_map:
                    expected_mime = extension_mime_map[file_ext]
                    if mime_type and expected_mime not in mime_type:
                        anomalies.append(f"File extension/MIME type mismatch: {file_ext} vs {mime_type}")
                        
        except Exception as e:
            self.logger.warning(f"Anomaly detection error: {e}")
            
        return anomalies
        
    def analyze_timeline_patterns(self):
        """Analyze timeline patterns across all files"""
        if not self.timeline_data:
            return []
            
        timeline_anomalies = []
        
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(self.timeline_data)
            df['created'] = pd.to_datetime(df['created'])
            df['modified'] = pd.to_datetime(df['modified'])
            df['accessed'] = pd.to_datetime(df['accessed'])
            
            # Detect bulk operations (many files with same timestamp)
            created_counts = df['created'].value_counts()
            modified_counts = df['modified'].value_counts()
            
            # If more than 10 files have the exact same creation time
            bulk_created = created_counts[created_counts > 10]
            if not bulk_created.empty:
                for timestamp, count in bulk_created.items():
                    timeline_anomalies.append({
                        'type': 'bulk_creation',
                        'timestamp': timestamp.isoformat(),
                        'file_count': count,
                        'description': f"{count} files created at exactly the same time"
                    })
                    
            # Similar check for modification times
            bulk_modified = modified_counts[modified_counts > 10]
            if not bulk_modified.empty:
                for timestamp, count in bulk_modified.items():
                    timeline_anomalies.append({
                        'type': 'bulk_modification',
                        'timestamp': timestamp.isoformat(),
                        'file_count': count,
                        'description': f"{count} files modified at exactly the same time"
                    })
                    
            # Detect unusual time patterns (e.g., files created in the future)
            now = datetime.now()
            future_files = df[df['created'] > now]
            if not future_files.empty:
                timeline_anomalies.append({
                    'type': 'future_timestamps',
                    'file_count': len(future_files),
                    'description': f"{len(future_files)} files have creation times in the future"
                })
                
            # Detect very old access times (possibly indicating file recovery)
            very_old_access = df[df['accessed'] < (now - timedelta(days=365 * 5))]  # 5 years ago
            if not very_old_access.empty:
                timeline_anomalies.append({
                    'type': 'very_old_access',
                    'file_count': len(very_old_access),
                    'description': f"{len(very_old_access)} files have very old access times"
                })
                
            self.analysis_stats['timeline_inconsistencies'] = len(timeline_anomalies)
            
        except Exception as e:
            self.logger.error(f"Timeline analysis error: {e}")
            
        return timeline_anomalies
        
    def generate_analysis_report(self, output_dir):
        """Generate comprehensive metadata analysis report"""
        try:
            # Create reports directory
            reports_dir = output_dir / "metadata_analysis_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Analyze timeline patterns
            timeline_anomalies = self.analyze_timeline_patterns()
            
            # Generate JSON report
            json_report = reports_dir / f"metadata_analysis_{timestamp}.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                report_data = {
                    'analysis_summary': dict(self.analysis_stats),
                    'timeline_anomalies': timeline_anomalies,
                    'individual_analyses': self.metadata_results
                }
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate HTML report
            html_report = reports_dir / f"metadata_analysis_{timestamp}.html"
            self.generate_html_report(html_report, timeline_anomalies)
            
            # Generate CSV timeline data
            if self.timeline_data:
                csv_report = reports_dir / f"timeline_data_{timestamp}.csv"
                self.generate_csv_timeline(csv_report)
                
            self.logger.info(f"✓ Reports generated in: {reports_dir}")
            return reports_dir
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
            
    def generate_html_report(self, html_path, timeline_anomalies):
        """Generate HTML metadata analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Forensic Metadata Analysis Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 40px; background: #0a0a0a; color: #00ff00; }}
        .header {{ background: #1a1a1a; padding: 20px; border: 2px solid #00ff00; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #1a1a1a; padding: 15px; border: 1px solid #00ff00; text-align: center; }}
        .file-analysis {{ margin: 20px 0; padding: 15px; border: 1px solid #333; background: #1a1a1a; }}
        .high-risk {{ border-left: 5px solid #ff0000; }}
        .medium-risk {{ border-left: 5px solid #ffaa00; }}
        .low-risk {{ border-left: 5px solid #00ff00; }}
        .anomalies {{ color: #ff4444; background: #2a1a1a; padding: 10px; margin: 10px 0; }}
        .timeline-anomaly {{ background: #2a2a1a; padding: 15px; margin: 10px 0; border: 1px solid #ffaa00; }}
        .metadata {{ color: #aaaaaa; font-size: 12px; }}
        h1, h2, h3 {{ color: #00ff00; }}
        .hash {{ font-family: 'Courier New', monospace; font-size: 10px; word-break: break-all; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 FORENSIC METADATA ANALYSIS REPORT</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analyzer: Forensic Metadata Comparator v1.0.0</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{self.analysis_stats['processed_files']}</h3>
            <p>Files Processed</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['metadata_extracted']}</h3>
            <p>Metadata Extracted</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['anomalies_found']}</h3>
            <p>Anomalies Found</p>
        </div>
        <div class="metric">
            <h3>{len(timeline_anomalies)}</h3>
            <p>Timeline Issues</p>
        </div>
    </div>
"""
        
        # Add timeline anomalies section
        if timeline_anomalies:
            html_content += """
    <h2>⏰ Timeline Anomalies</h2>
"""
            for anomaly in timeline_anomalies:
                html_content += f"""
    <div class="timeline-anomaly">
        <h3>{anomaly.get('type', 'Unknown').replace('_', ' ').title()}</h3>
        <p>{anomaly.get('description', 'No description')}</p>
        <p><strong>Affected Files:</strong> {anomaly.get('file_count', 0)}</p>
    </div>
"""
        
        # Add individual file analyses
        html_content += "\n    <h2>📋 Individual File Analysis</h2>\n"
        
        for analysis in self.metadata_results:
            if 'error' in analysis:
                continue
                
            risk_score = analysis.get('risk_score', 0)
            risk_class = 'high-risk' if risk_score > 60 else 'medium-risk' if risk_score > 30 else 'low-risk'
            
            html_content += f"""
    <div class="file-analysis {risk_class}">
        <h3>📄 {Path(analysis['file_path']).name}</h3>
        <p><strong>Risk Score:</strong> {risk_score}/100</p>
        
        <div class="metadata">
"""
            
            if 'basic_metadata' in analysis:
                basic = analysis['basic_metadata']
                html_content += f"""
            <p><strong>Size:</strong> {basic.get('size_mb', 0)} MB</p>
            <p><strong>Created:</strong> {basic.get('created_time', 'Unknown')}</p>
            <p><strong>Modified:</strong> {basic.get('modified_time', 'Unknown')}</p>
            <p><strong>Hash:</strong> <span class="hash">{basic.get('file_hash', 'N/A')}</span></p>
"""
            
            html_content += "        </div>\n"
            
            if analysis.get('anomalies'):
                html_content += """
        <div class="anomalies">
            <h4>⚠️ Anomalies Detected:</h4>
            <ul>
"""
                for anomaly in analysis['anomalies']:
                    html_content += f"                <li>{anomaly}</li>\n"
                html_content += "            </ul>\n        </div>\n"
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def generate_csv_timeline(self, csv_path):
        """Generate CSV timeline data"""
        try:
            df = pd.DataFrame(self.timeline_data)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            self.logger.error(f"CSV generation error: {e}")
            
    def show_completion_summary(self, reports_dir):
        """Show analysis completion summary"""
        try:
            elapsed = time.time() - self.analysis_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""Forensic Metadata Analysis Completed!

Files Processed: {self.analysis_stats['processed_files']}
Metadata Extracted: {self.analysis_stats['metadata_extracted']}
Anomalies Found: {self.analysis_stats['anomalies_found']}
Timeline Issues: {self.analysis_stats['timeline_inconsistencies']}

Time Elapsed: {int(minutes)}m {int(seconds)}s

Reports saved to: {reports_dir.name}

Check the log file for detailed information."""

            root = tk.Tk()
            root.withdraw()
            
            if IS_MACOS:
                root.attributes("-topmost", True)
                
            messagebox.showinfo("Analysis Complete", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Advanced Forensic Metadata Comparator")
        parser.add_argument('--input', help='Input directory for analysis')
        parser.add_argument('--output', help='Output directory for reports')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Forensic Metadata Comparator v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting forensic metadata analysis...")
        
        try:
            # Get input directory
            if args.input:
                input_dir = Path(args.input)
                if not input_dir.exists() or not input_dir.is_dir():
                    self.logger.error(f"Input directory not found: {args.input}")
                    return
            else:
                input_dir = self.select_input_path()
                
            if not input_dir:
                self.logger.warning("No input directory selected")
                return
                
            # Find all files
            all_files = []
            for file_path in input_dir.rglob('*'):
                if file_path.is_file():
                    all_files.append(file_path)
                    
            if not all_files:
                self.logger.warning("No files found in directory")
                return
                
            # Get output directory
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = self.select_output_directory()
                
            if not output_dir:
                self.logger.error("No output directory selected")
                return
                
            self.analysis_stats['total_files'] = len(all_files)
            self.analysis_stats['start_time'] = time.time()
            
            self.logger.info(f"Analyzing metadata for {len(all_files)} files...")
            
            # Create progress window
            self.create_progress_window(len(all_files))
            
            # Process files with thread pool
            with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
                future_to_file = {
                    executor.submit(self.analyze_metadata, file_path): file_path
                    for file_path in all_files
                }
                
                completed = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed += 1
                    
                    try:
                        analysis = future.result()
                        self.metadata_results.append(analysis)
                        self.update_progress(completed, len(all_files), file_path.name)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {file_path}: {e}")
                        
            # Close progress window
            if hasattr(self, 'progress_root'):
                self.progress_root.destroy()
                
            # Generate reports
            reports_dir = self.generate_analysis_report(output_dir)
            
            if reports_dir:
                # Show completion summary
                self.show_completion_summary(reports_dir)
                
                # Open reports directory if on macOS
                if IS_MACOS:
                    subprocess.run(['open', str(reports_dir)], check=False)
            
        except KeyboardInterrupt:
            self.logger.info("Analysis interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Open log file
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    comparator = ForensicMetadataComparator()
    comparator.run()

if __name__ == "__main__":
    main()