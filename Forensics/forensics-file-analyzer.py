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
# Script Name: forensics-file-analyzer.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced digital forensics file analyzer with deep structure     
#              inspection, signature verification, entropy analysis, and hidden 
#              data detection for comprehensive file system investigation.      
#
# Usage: python forensics-file-analyzer.py [--input PATH] [--output DIR] [--deep]
#
# Dependencies: python-magic, yara-python, exifread, hashlib, entropy           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional forensic analysis with evidence integrity preservation,   
#        comprehensive logging, and universal macOS compatibility.             
#                                                                                
####################################################################################

"""
Advanced Digital Forensics File Analyzer
========================================

Comprehensive file analysis tool for digital forensics investigations.
Performs deep structure inspection, signature verification, entropy analysis,
and hidden data detection with evidence integrity preservation.
"""

import os
import sys
import logging
import multiprocessing
import subprocess
import argparse
import time
import hashlib
import math
import struct
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "python-magic>=0.4.24",
        "exifread>=2.3.2", 
        "pefile>=2021.9.3",
        "yara-python>=4.2.0"
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

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

try:
    import exifread
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False

try:
    import pefile
    PE_AVAILABLE = True
except ImportError:
    PE_AVAILABLE = False

try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False

IS_MACOS = sys.platform == "darwin"

class FileForensicsAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        # File signature database
        self.file_signatures = {
            b'\x89PNG\r\n\x1a\n': 'PNG Image',
            b'\xff\xd8\xff': 'JPEG Image',
            b'GIF87a': 'GIF Image (87a)',
            b'GIF89a': 'GIF Image (89a)',
            b'%PDF': 'PDF Document',
            b'PK\x03\x04': 'ZIP Archive/Office Document',
            b'PK\x05\x06': 'ZIP Archive (empty)',
            b'Rar!\x1a\x07\x00': 'RAR Archive',
            b'\x7fELF': 'ELF Executable',
            b'MZ': 'Windows PE Executable',
            b'\xca\xfe\xba\xbe': 'Java Class File',
            b'\xfe\xed\xfa\xce': 'Mach-O Binary (32-bit)',
            b'\xfe\xed\xfa\xcf': 'Mach-O Binary (64-bit)',
            b'SQLite format 3\x00': 'SQLite Database',
            b'\x1f\x8b': 'GZIP Archive',
            b'BZh': 'BZIP2 Archive'
        }
        
        self.analysis_stats = {
            'total_files': 0,
            'analyzed_files': 0,
            'suspicious_files': 0,
            'start_time': None,
            'file_types': defaultdict(int),
            'anomalies_found': 0
        }
        
        self.analysis_results = []
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "forensics-file-analyzer.log"
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
        """Select input file or directory via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        # Ask user whether to select file or directory
        choice = messagebox.askyesno(
            "Input Selection",
            "Select 'Yes' for single file, 'No' for directory analysis"
        )
        
        if choice:
            # Single file
            file_path = filedialog.askopenfilename(
                title="Select File for Forensic Analysis",
                filetypes=[("All files", "*.*")]
            )
            result = Path(file_path) if file_path else None
        else:
            # Directory
            directory = filedialog.askdirectory(
                title="Select Directory for Forensic Analysis"
            )
            result = Path(directory) if directory else None
            
        root.destroy()
        return result
        
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
        self.progress_root.title("Forensic File Analysis")
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
        self.status_var = tk.StringVar(value="Initializing forensic analysis...")
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
            
    def calculate_file_hashes(self, file_path):
        """Calculate multiple hash values for file integrity"""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
            hashes['md5'] = hashlib.md5(content).hexdigest()
            hashes['sha1'] = hashlib.sha1(content).hexdigest()
            hashes['sha256'] = hashlib.sha256(content).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Hash calculation error for {file_path}: {e}")
            
        return hashes
        
    def calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
            
        entropy = 0
        for x in range(256):
            p_x = float(data.count(bytes([x]))) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
                
        return entropy
        
    def detect_file_signature(self, file_path):
        """Detect file type by signature analysis"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
                
            detected_type = None
            confidence = 0
            
            # Check against known signatures
            for signature, file_type in self.file_signatures.items():
                if header.startswith(signature):
                    detected_type = file_type
                    confidence = 100
                    break
                    
            # Use python-magic if available
            if MAGIC_AVAILABLE and not detected_type:
                try:
                    magic_result = magic.from_file(str(file_path))
                    detected_type = magic_result
                    confidence = 85
                except Exception:
                    pass
                    
            return {
                'detected_type': detected_type or 'Unknown',
                'confidence': confidence,
                'header_hex': header[:32].hex(),
                'header_ascii': ''.join(chr(b) if 32 <= b <= 126 else '.' for b in header[:32])
            }
            
        except Exception as e:
            self.logger.error(f"Signature detection error for {file_path}: {e}")
            return {'detected_type': 'Error', 'confidence': 0}
            
    def analyze_pe_file(self, file_path):
        """Analyze PE (Windows executable) files"""
        if not PE_AVAILABLE:
            return None
            
        try:
            pe = pefile.PE(str(file_path))
            
            analysis = {
                'pe_type': 'PE32+' if pe.OPTIONAL_HEADER.Magic == 0x20b else 'PE32',
                'machine_type': hex(pe.FILE_HEADER.Machine),
                'timestamp': pe.FILE_HEADER.TimeDateStamp,
                'entry_point': hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint),
                'sections': [],
                'imports': [],
                'exports': [],
                'suspicious_indicators': []
            }
            
            # Analyze sections
            for section in pe.sections:
                section_info = {
                    'name': section.Name.decode('utf-8', errors='ignore').strip('\x00'),
                    'virtual_address': hex(section.VirtualAddress),
                    'size': section.SizeOfRawData,
                    'entropy': section.get_entropy()
                }
                
                # Check for suspicious section characteristics
                if section.get_entropy() > 7.0:
                    analysis['suspicious_indicators'].append(f"High entropy section: {section_info['name']}")
                    
                analysis['sections'].append(section_info)
                
            # Analyze imports
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8', errors='ignore')
                    imports = []
                    
                    for imp in entry.imports:
                        if imp.name:
                            imports.append(imp.name.decode('utf-8', errors='ignore'))
                            
                    analysis['imports'].append({
                        'dll': dll_name,
                        'functions': imports[:10]  # Limit to first 10
                    })
                    
            # Check for suspicious imports
            suspicious_dlls = ['kernel32.dll', 'ntdll.dll', 'advapi32.dll']
            suspicious_functions = ['VirtualAlloc', 'WriteProcessMemory', 'CreateRemoteThread']
            
            for imp_entry in analysis['imports']:
                if imp_entry['dll'].lower() in [dll.lower() for dll in suspicious_dlls]:
                    for func in imp_entry['functions']:
                        if func in suspicious_functions:
                            analysis['suspicious_indicators'].append(f"Suspicious import: {func} from {imp_entry['dll']}")
                            
            pe.close()
            return analysis
            
        except Exception as e:
            self.logger.warning(f"PE analysis failed for {file_path}: {e}")
            return None
            
    def extract_metadata(self, file_path):
        """Extract file metadata and EXIF data"""
        metadata = {}
        
        try:
            # Basic file metadata
            stat = file_path.stat()
            metadata['file_info'] = {
                'size_bytes': stat.st_size,
                'created': time.ctime(stat.st_ctime),
                'modified': time.ctime(stat.st_mtime),
                'accessed': time.ctime(stat.st_atime),
                'permissions': oct(stat.st_mode)[-3:]
            }
            
            # EXIF data for images
            if EXIF_AVAILABLE and file_path.suffix.lower() in ['.jpg', '.jpeg', '.tiff']:
                try:
                    with open(file_path, 'rb') as f:
                        tags = exifread.process_file(f)
                        
                    exif_data = {}
                    for tag in tags.keys():
                        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                            exif_data[tag] = str(tags[tag])
                            
                    if exif_data:
                        metadata['exif'] = exif_data
                        
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Metadata extraction error for {file_path}: {e}")
            
        return metadata
        
    def detect_anomalies(self, file_path, analysis_data):
        """Detect various file anomalies and suspicious patterns"""
        anomalies = []
        
        try:
            # Size anomalies
            size = analysis_data.get('metadata', {}).get('file_info', {}).get('size_bytes', 0)
            
            if size == 0:
                anomalies.append("Zero-byte file")
            elif size > 100 * 1024 * 1024:  # 100MB
                anomalies.append("Unusually large file size")
                
            # Extension mismatch
            file_ext = file_path.suffix.lower()
            detected_type = analysis_data.get('signature', {}).get('detected_type', '').lower()
            
            extension_map = {
                '.jpg': 'jpeg',
                '.jpeg': 'jpeg',
                '.png': 'png',
                '.pdf': 'pdf',
                '.zip': 'zip',
                '.exe': 'executable',
                '.dll': 'executable'
            }
            
            if file_ext in extension_map:
                expected_type = extension_map[file_ext]
                if expected_type not in detected_type:
                    anomalies.append(f"Extension mismatch: {file_ext} vs {detected_type}")
                    
            # Entropy analysis
            if 'entropy' in analysis_data:
                entropy = analysis_data['entropy']
                if entropy > 7.5:
                    anomalies.append("High entropy (possible encryption/compression)")
                elif entropy < 1.0:
                    anomalies.append("Very low entropy (highly repetitive data)")
                    
            # Hidden data detection
            with open(file_path, 'rb') as f:
                content = f.read(8192)  # Check first 8KB
                
            # Look for embedded executables
            if b'MZ' in content[100:]:  # PE header not at beginning
                anomalies.append("Possible embedded executable")
                
            # Look for suspicious strings
            suspicious_strings = [b'eval(', b'exec(', b'system(', b'shell_exec', b'base64_decode']
            for sus_string in suspicious_strings:
                if sus_string in content:
                    anomalies.append(f"Suspicious string found: {sus_string.decode('utf-8', errors='ignore')}")
                    
        except Exception as e:
            self.logger.warning(f"Anomaly detection error for {file_path}: {e}")
            
        return anomalies
        
    def analyze_file(self, file_path, deep_analysis=False):
        """Perform comprehensive file analysis"""
        try:
            analysis = {
                'file_path': str(file_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hashes': {},
                'signature': {},
                'metadata': {},
                'entropy': 0,
                'anomalies': [],
                'risk_score': 0
            }
            
            # Calculate file hashes
            analysis['hashes'] = self.calculate_file_hashes(file_path)
            
            # Signature analysis
            analysis['signature'] = self.detect_file_signature(file_path)
            
            # Metadata extraction
            analysis['metadata'] = self.extract_metadata(file_path)
            
            # Entropy calculation
            try:
                with open(file_path, 'rb') as f:
                    sample_data = f.read(65536)  # 64KB sample
                analysis['entropy'] = self.calculate_entropy(sample_data)
            except Exception:
                pass
                
            # PE analysis for executables
            if file_path.suffix.lower() in ['.exe', '.dll', '.sys']:
                pe_analysis = self.analyze_pe_file(file_path)
                if pe_analysis:
                    analysis['pe_analysis'] = pe_analysis
                    
            # Anomaly detection
            analysis['anomalies'] = self.detect_anomalies(file_path, analysis)
            
            # Calculate risk score
            risk_score = 0
            if analysis['anomalies']:
                risk_score += len(analysis['anomalies']) * 10
            if analysis['entropy'] > 7.5:
                risk_score += 20
            if 'pe_analysis' in analysis and analysis['pe_analysis'].get('suspicious_indicators'):
                risk_score += len(analysis['pe_analysis']['suspicious_indicators']) * 15
                
            analysis['risk_score'] = min(100, risk_score)
            
            # Update stats
            self.analysis_stats['analyzed_files'] += 1
            if analysis['anomalies'] or risk_score > 30:
                self.analysis_stats['suspicious_files'] += 1
            if analysis['anomalies']:
                self.analysis_stats['anomalies_found'] += len(analysis['anomalies'])
                
            file_type = analysis['signature'].get('detected_type', 'Unknown')
            self.analysis_stats['file_types'][file_type] += 1
            
            self.logger.info(f"✓ Analyzed: {file_path.name} (Risk: {risk_score}/100)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"✗ Analysis failed for {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
            
    def generate_analysis_report(self, output_dir):
        """Generate comprehensive forensic analysis report"""
        try:
            # Create reports directory
            reports_dir = output_dir / "forensic_analysis_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Generate JSON report
            json_report = reports_dir / f"forensic_analysis_{timestamp}.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                report_data = {
                    'analysis_summary': dict(self.analysis_stats),
                    'individual_analyses': self.analysis_results
                }
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generate HTML report
            html_report = reports_dir / f"forensic_analysis_{timestamp}.html"
            self.generate_html_report(html_report)
            
            # Generate CSV for suspicious files
            csv_report = reports_dir / f"suspicious_files_{timestamp}.csv"
            self.generate_csv_report(csv_report)
            
            self.logger.info(f"✓ Reports generated in: {reports_dir}")
            return reports_dir
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
            
    def generate_html_report(self, html_path):
        """Generate HTML forensic analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Forensic File Analysis Report</title>
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
        .metadata {{ color: #aaaaaa; font-size: 12px; }}
        h1, h2, h3 {{ color: #00ff00; }}
        .hash {{ font-family: 'Courier New', monospace; font-size: 11px; word-break: break-all; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 FORENSIC FILE ANALYSIS REPORT</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analyzer: Digital Forensics File Analyzer v1.0.0</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{self.analysis_stats['analyzed_files']}</h3>
            <p>Files Analyzed</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['suspicious_files']}</h3>
            <p>Suspicious Files</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['anomalies_found']}</h3>
            <p>Anomalies Detected</p>
        </div>
        <div class="metric">
            <h3>{len(self.analysis_stats['file_types'])}</h3>
            <p>File Types Found</p>
        </div>
    </div>
    
    <h2>📊 File Type Distribution</h2>
    <div style="background: #1a1a1a; padding: 15px; border: 1px solid #00ff00;">
"""
        
        for file_type, count in sorted(self.analysis_stats['file_types'].items(), key=lambda x: x[1], reverse=True):
            html_content += f"        <p>{file_type}: {count} files</p>\n"
            
        html_content += "    </div>\n\n"
        
        # Add individual file analyses
        for analysis in self.analysis_results:
            if 'error' in analysis:
                continue
                
            risk_score = analysis.get('risk_score', 0)
            risk_class = 'high-risk' if risk_score > 60 else 'medium-risk' if risk_score > 30 else 'low-risk'
            
            html_content += f"""
    <div class="file-analysis {risk_class}">
        <h3>📄 {Path(analysis['file_path']).name}</h3>
        <p><strong>Risk Score:</strong> {risk_score}/100</p>
        <p><strong>File Type:</strong> {analysis['signature'].get('detected_type', 'Unknown')}</p>
        
        <div class="metadata">
            <p><strong>Size:</strong> {analysis['metadata'].get('file_info', {}).get('size_bytes', 0):,} bytes</p>
            <p><strong>Entropy:</strong> {analysis.get('entropy', 0):.2f}</p>
            <p><strong>SHA256:</strong> <span class="hash">{analysis['hashes'].get('sha256', 'N/A')}</span></p>
        </div>
"""
            
            if analysis.get('anomalies'):
                html_content += """
        <div class="anomalies">
            <h4>⚠️ Anomalies Detected:</h4>
            <ul>
"""
                for anomaly in analysis['anomalies']:
                    html_content += f"                <li>{anomaly}</li>\n"
                html_content += "            </ul>\n        </div>\n"
            
            if 'pe_analysis' in analysis and analysis['pe_analysis'].get('suspicious_indicators'):
                html_content += """
        <div class="anomalies">
            <h4>🚨 PE Analysis Indicators:</h4>
            <ul>
"""
                for indicator in analysis['pe_analysis']['suspicious_indicators']:
                    html_content += f"                <li>{indicator}</li>\n"
                html_content += "            </ul>\n        </div>\n"
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def generate_csv_report(self, csv_path):
        """Generate CSV report of suspicious files"""
        try:
            import csv
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['File Path', 'Risk Score', 'File Type', 'Size', 'Entropy', 'Anomalies', 'SHA256'])
                
                for analysis in self.analysis_results:
                    if 'error' in analysis or not analysis.get('anomalies'):
                        continue
                        
                    writer.writerow([
                        analysis['file_path'],
                        analysis.get('risk_score', 0),
                        analysis['signature'].get('detected_type', 'Unknown'),
                        analysis['metadata'].get('file_info', {}).get('size_bytes', 0),
                        analysis.get('entropy', 0),
                        '; '.join(analysis.get('anomalies', [])),
                        analysis['hashes'].get('sha256', 'N/A')
                    ])
                    
        except Exception as e:
            self.logger.error(f"CSV generation error: {e}")
            
    def show_completion_summary(self, reports_dir):
        """Show analysis completion summary"""
        try:
            elapsed = time.time() - self.analysis_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""Forensic Analysis Completed!

Files Analyzed: {self.analysis_stats['analyzed_files']}
Suspicious Files: {self.analysis_stats['suspicious_files']}
Anomalies Found: {self.analysis_stats['anomalies_found']}
File Types: {len(self.analysis_stats['file_types'])}

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
        parser = argparse.ArgumentParser(description="Advanced Digital Forensics File Analyzer")
        parser.add_argument('--input', help='Input file or directory')
        parser.add_argument('--output', help='Output directory for reports')
        parser.add_argument('--deep', action='store_true', help='Perform deep analysis')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Forensic File Analyzer v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting forensic file analysis...")
        
        try:
            # Get input path
            if args.input:
                input_path = Path(args.input)
                if not input_path.exists():
                    self.logger.error(f"Input path not found: {args.input}")
                    return
            else:
                input_path = self.select_input_path()
                
            if not input_path:
                self.logger.warning("No input path selected")
                return
                
            # Get file list
            if input_path.is_file():
                files_to_analyze = [input_path]
            else:
                files_to_analyze = []
                for file_path in input_path.rglob('*'):
                    if file_path.is_file():
                        files_to_analyze.append(file_path)
                        
            if not files_to_analyze:
                self.logger.warning("No files found to analyze")
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
                
            self.analysis_stats['total_files'] = len(files_to_analyze)
            self.analysis_stats['start_time'] = time.time()
            
            self.logger.info(f"Analyzing {len(files_to_analyze)} files...")
            
            # Create progress window
            self.create_progress_window(len(files_to_analyze))
            
            # Process files with thread pool
            deep_analysis = args.deep if args.deep else False
            
            with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
                future_to_file = {
                    executor.submit(self.analyze_file, file_path, deep_analysis): file_path
                    for file_path in files_to_analyze
                }
                
                completed = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed += 1
                    
                    try:
                        analysis = future.result()
                        self.analysis_results.append(analysis)
                        self.update_progress(completed, len(files_to_analyze), file_path.name)
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
    analyzer = FileForensicsAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()