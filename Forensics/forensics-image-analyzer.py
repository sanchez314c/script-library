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
# Script Name: forensics-image-analyzer.py                                       
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced forensic image analysis tool with EXIF extraction,      
#              duplicate detection, steganography scanning, and comprehensive   
#              visual investigation capabilities for digital evidence processing.
#
# Usage: python forensics-image-analyzer.py [--input PATH] [--output DIR] [--steg]
#
# Dependencies: Pillow, exifread, imagehash, opencv-python, numpy              
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional image forensics with metadata preservation, duplicate     
#        detection, and steganography analysis with universal macOS support.   
#                                                                                
####################################################################################

"""
Advanced Forensic Image Analyzer
===============================

Comprehensive image analysis tool for digital forensics investigations.
Performs EXIF analysis, duplicate detection, steganography scanning,
and visual investigation with evidence integrity preservation.
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
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import statistics

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "Pillow>=8.0.0",
        "exifread>=2.3.2",
        "imagehash>=4.2.1",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0"
    ]
    
    missing = []
    for pkg in required_packages:
        name = pkg.split('>=')[0].replace('-', '_')
        if name == 'opencv_python':
            name = 'cv2'
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
from PIL import Image, ImageStat, ImageFilter, ExifTags
import exifread
import imagehash
import cv2
import numpy as np

IS_MACOS = sys.platform == "darwin"

class ForensicImageAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        
        self.analysis_stats = {
            'total_images': 0,
            'analyzed_images': 0,
            'duplicate_groups': 0,
            'metadata_extracted': 0,
            'suspicious_images': 0,
            'start_time': None
        }
        
        self.analysis_results = []
        self.image_hashes = {}
        self.duplicate_groups = []
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "forensics-image-analyzer.log"
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
            title="Select Directory with Images for Forensic Analysis"
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
        
    def create_progress_window(self, total_images):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Forensic Image Analysis")
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
            maximum=total_images,
            length=450
        )
        self.progress_bar.pack(pady=20)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing image analysis...")
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
            
    def extract_comprehensive_exif(self, image_path):
        """Extract comprehensive EXIF and metadata"""
        metadata = {}
        
        try:
            # PIL EXIF extraction
            with Image.open(image_path) as img:
                basic_info = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height
                }
                
                exifdata = img.getexif()
                if exifdata:
                    pil_exif = {}
                    for tag_id, value in exifdata.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        pil_exif[tag] = str(value)
                    basic_info['pil_exif'] = pil_exif
                    
                metadata['basic_info'] = basic_info
                
            # ExifRead for detailed extraction
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                
            detailed_exif = {}
            for tag in tags.keys():
                if tag not in ('JPEGThumbnail', 'TIFFThumbnail'):
                    try:
                        detailed_exif[tag] = str(tags[tag])
                    except:
                        detailed_exif[tag] = 'Unable to read'
                        
            if detailed_exif:
                metadata['detailed_exif'] = detailed_exif
                
            # Extract specific forensic-relevant fields
            forensic_fields = {}
            
            # Camera information
            if 'Image Make' in detailed_exif:
                forensic_fields['camera_make'] = detailed_exif['Image Make']
            if 'Image Model' in detailed_exif:
                forensic_fields['camera_model'] = detailed_exif['Image Model']
            if 'EXIF LensModel' in detailed_exif:
                forensic_fields['lens_model'] = detailed_exif['EXIF LensModel']
                
            # GPS coordinates
            gps_fields = ['GPS GPSLatitude', 'GPS GPSLongitude', 'GPS GPSAltitude']
            gps_data = {}
            for field in gps_fields:
                if field in detailed_exif:
                    gps_data[field] = detailed_exif[field]
            if gps_data:
                forensic_fields['gps_data'] = gps_data
                
            # Timestamps
            timestamp_fields = ['DateTime', 'EXIF DateTimeOriginal', 'EXIF DateTimeDigitized']
            timestamps = {}
            for field in timestamp_fields:
                if field in detailed_exif:
                    timestamps[field] = detailed_exif[field]
            if timestamps:
                forensic_fields['timestamps'] = timestamps
                
            # Software information
            software_fields = ['Image Software', 'EXIF Software']
            for field in software_fields:
                if field in detailed_exif:
                    forensic_fields['software'] = detailed_exif[field]
                    break
                    
            if forensic_fields:
                metadata['forensic_data'] = forensic_fields
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"EXIF extraction error for {image_path}: {e}")
            return {'error': str(e)}
            
    def calculate_image_hashes(self, image_path):
        """Calculate multiple image hashes for duplicate detection"""
        hashes = {}
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Calculate different hash types
                hashes['average_hash'] = str(imagehash.average_hash(img))
                hashes['perception_hash'] = str(imagehash.phash(img))
                hashes['difference_hash'] = str(imagehash.dhash(img))
                hashes['wavelet_hash'] = str(imagehash.whash(img))
                
                # File hash for exact duplicates
                with open(image_path, 'rb') as f:
                    file_content = f.read()
                    hashes['file_md5'] = hashlib.md5(file_content).hexdigest()
                    hashes['file_sha256'] = hashlib.sha256(file_content).hexdigest()
                    
        except Exception as e:
            self.logger.error(f"Hash calculation error for {image_path}: {e}")
            
        return hashes
        
    def analyze_image_quality(self, image_path):
        """Analyze image quality and detect anomalies"""
        quality_data = {}
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB for analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Basic quality metrics
                stat = ImageStat.Stat(img)
                
                quality_data['brightness'] = {
                    'mean': stat.mean,
                    'median': stat.median,
                    'stddev': stat.stddev
                }
                
                # Calculate image entropy (information content)
                entropy = img.entropy()
                quality_data['entropy'] = entropy
                
                # Detect potential compression artifacts
                # Save as JPEG with high quality and compare
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    img.save(tmp.name, 'JPEG', quality=95)
                    
                    with Image.open(tmp.name) as recompressed:
                        # Calculate difference
                        diff = ImageStat.Stat(Image.blend(img, recompressed, 0.5))
                        quality_data['compression_artifacts'] = sum(diff.stddev) / len(diff.stddev)
                        
                    os.unlink(tmp.name)
                    
                # Blur detection using OpenCV
                cv_img = cv2.imread(str(image_path))
                if cv_img is not None:
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    quality_data['blur_score'] = laplacian_var
                    
                    # Noise detection
                    noise_score = np.std(gray)
                    quality_data['noise_score'] = noise_score
                    
        except Exception as e:
            self.logger.error(f"Quality analysis error for {image_path}: {e}")
            quality_data['error'] = str(e)
            
        return quality_data
        
    def detect_steganography_indicators(self, image_path):
        """Detect potential steganography indicators"""
        steg_indicators = {}
        
        try:
            with Image.open(image_path) as img:
                # LSB analysis for potential hidden data
                if img.mode == 'RGB':
                    r, g, b = img.split()
                    
                    # Analyze LSB patterns in each channel
                    for channel_name, channel in [('red', r), ('green', g), ('blue', b)]:
                        pixels = list(channel.getdata())
                        lsb_bits = [pixel & 1 for pixel in pixels]
                        
                        # Calculate randomness of LSB
                        lsb_entropy = self.calculate_bit_entropy(lsb_bits)
                        steg_indicators[f'{channel_name}_lsb_entropy'] = lsb_entropy
                        
                        # Chi-square test for randomness
                        chi_square = self.chi_square_test(lsb_bits)
                        steg_indicators[f'{channel_name}_chi_square'] = chi_square
                        
                # File size anomaly detection
                expected_size = img.width * img.height * 3  # RGB
                actual_size = os.path.getsize(image_path)
                size_ratio = actual_size / expected_size
                steg_indicators['size_anomaly_ratio'] = size_ratio
                
                # Detect unusual file structure patterns
                with open(image_path, 'rb') as f:
                    content = f.read()
                    
                # Look for embedded file signatures
                embedded_signatures = []
                signatures = [b'PK\x03\x04', b'%PDF', b'Rar!', b'\x89PNG']
                for sig in signatures:
                    if content.find(sig, 100) != -1:  # Not at beginning
                        embedded_signatures.append(sig.hex())
                        
                if embedded_signatures:
                    steg_indicators['embedded_signatures'] = embedded_signatures
                    
        except Exception as e:
            self.logger.error(f"Steganography detection error for {image_path}: {e}")
            steg_indicators['error'] = str(e)
            
        return steg_indicators
        
    def calculate_bit_entropy(self, bits):
        """Calculate entropy of bit sequence"""
        if not bits:
            return 0
            
        ones = sum(bits)
        zeros = len(bits) - ones
        
        if ones == 0 or zeros == 0:
            return 0
            
        p_ones = ones / len(bits)
        p_zeros = zeros / len(bits)
        
        entropy = -p_ones * math.log2(p_ones) - p_zeros * math.log2(p_zeros)
        return entropy
        
    def chi_square_test(self, bits):
        """Perform chi-square test on bit sequence"""
        if len(bits) < 100:
            return 0
            
        ones = sum(bits)
        zeros = len(bits) - ones
        expected = len(bits) / 2
        
        chi_square = ((ones - expected) ** 2 / expected) + ((zeros - expected) ** 2 / expected)
        return chi_square
        
    def find_duplicate_images(self):
        """Find duplicate and similar images using perceptual hashing"""
        self.logger.info("Analyzing image duplicates...")
        
        # Group images by perceptual hash
        hash_groups = defaultdict(list)
        
        for result in self.analysis_results:
            if 'hashes' in result and 'perception_hash' in result['hashes']:
                phash = result['hashes']['perception_hash']
                hash_groups[phash].append(result)
                
        # Find similar images (allowing small hash differences)
        similar_groups = []
        processed_hashes = set()
        
        for result in self.analysis_results:
            if 'hashes' not in result or 'perception_hash' not in result['hashes']:
                continue
                
            current_hash = result['hashes']['perception_hash']
            if current_hash in processed_hashes:
                continue
                
            similar_group = [result]
            processed_hashes.add(current_hash)
            
            # Find similar images (Hamming distance <= 5)
            for other_result in self.analysis_results:
                if other_result == result:
                    continue
                    
                if 'hashes' not in other_result or 'perception_hash' not in other_result['hashes']:
                    continue
                    
                other_hash = other_result['hashes']['perception_hash']
                if other_hash in processed_hashes:
                    continue
                    
                # Calculate Hamming distance
                try:
                    hash1 = imagehash.hex_to_hash(current_hash)
                    hash2 = imagehash.hex_to_hash(other_hash)
                    distance = hash1 - hash2
                    
                    if distance <= 5:  # Similar images
                        similar_group.append(other_result)
                        processed_hashes.add(other_hash)
                        
                except Exception:
                    continue
                    
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
                
        self.duplicate_groups = similar_groups
        self.analysis_stats['duplicate_groups'] = len(similar_groups)
        
        return similar_groups
        
    def analyze_image(self, image_path, check_steganography=False):
        """Perform comprehensive image analysis"""
        try:
            analysis = {
                'file_path': str(image_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': {},
                'hashes': {},
                'quality': {},
                'anomalies': [],
                'risk_score': 0
            }
            
            # Extract metadata and EXIF
            analysis['metadata'] = self.extract_comprehensive_exif(image_path)
            if analysis['metadata'] and 'error' not in analysis['metadata']:
                self.analysis_stats['metadata_extracted'] += 1
                
            # Calculate hashes
            analysis['hashes'] = self.calculate_image_hashes(image_path)
            
            # Quality analysis
            analysis['quality'] = self.analyze_image_quality(image_path)
            
            # Steganography detection if requested
            if check_steganography:
                steg_data = self.detect_steganography_indicators(image_path)
                analysis['steganography'] = steg_data
                
                # Check for steganography indicators
                if 'red_lsb_entropy' in steg_data:
                    avg_entropy = (steg_data.get('red_lsb_entropy', 0) + 
                                 steg_data.get('green_lsb_entropy', 0) + 
                                 steg_data.get('blue_lsb_entropy', 0)) / 3
                    if avg_entropy > 0.9:
                        analysis['anomalies'].append("High LSB entropy (possible steganography)")
                        
                if steg_data.get('embedded_signatures'):
                    analysis['anomalies'].append("Embedded file signatures detected")
                    
                if steg_data.get('size_anomaly_ratio', 1) > 2:
                    analysis['anomalies'].append("Unusual file size for image dimensions")
                    
            # Detect other anomalies
            if 'quality' in analysis and 'error' not in analysis['quality']:
                quality = analysis['quality']
                
                if quality.get('blur_score', 1000) < 100:
                    analysis['anomalies'].append("Image appears blurred")
                    
                if quality.get('compression_artifacts', 0) > 50:
                    analysis['anomalies'].append("Heavy compression artifacts detected")
                    
            # Check for metadata anomalies
            if 'metadata' in analysis and 'forensic_data' in analysis['metadata']:
                forensic = analysis['metadata']['forensic_data']
                
                # Check for missing expected metadata
                if 'timestamps' not in forensic:
                    analysis['anomalies'].append("Missing timestamp metadata")
                    
                # Check for software editing indicators
                software = forensic.get('software', '').lower()
                editing_software = ['photoshop', 'gimp', 'paint', 'editor']
                if any(edit_soft in software for edit_soft in editing_software):
                    analysis['anomalies'].append(f"Edited with: {forensic.get('software', 'Unknown')}")
                    
            # Calculate risk score
            risk_score = 0
            if analysis['anomalies']:
                risk_score += len(analysis['anomalies']) * 15
            if check_steganography and 'steganography' in analysis:
                steg = analysis['steganography']
                if steg.get('embedded_signatures'):
                    risk_score += 30
                avg_entropy = (steg.get('red_lsb_entropy', 0) + 
                             steg.get('green_lsb_entropy', 0) + 
                             steg.get('blue_lsb_entropy', 0)) / 3
                if avg_entropy > 0.9:
                    risk_score += 25
                    
            analysis['risk_score'] = min(100, risk_score)
            
            # Update stats
            self.analysis_stats['analyzed_images'] += 1
            if analysis['anomalies'] or risk_score > 30:
                self.analysis_stats['suspicious_images'] += 1
                
            self.logger.info(f"✓ Analyzed: {image_path.name} (Risk: {risk_score}/100)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"✗ Analysis failed for {image_path}: {e}")
            return {
                'file_path': str(image_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
            
    def generate_analysis_report(self, output_dir):
        """Generate comprehensive forensic image analysis report"""
        try:
            # Create reports directory
            reports_dir = output_dir / "forensic_image_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Find duplicates
            duplicate_groups = self.find_duplicate_images()
            
            # Generate JSON report
            json_report = reports_dir / f"image_analysis_{timestamp}.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                report_data = {
                    'analysis_summary': dict(self.analysis_stats),
                    'duplicate_groups': [[r['file_path'] for r in group] for group in duplicate_groups],
                    'individual_analyses': self.analysis_results
                }
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generate HTML report
            html_report = reports_dir / f"image_analysis_{timestamp}.html"
            self.generate_html_report(html_report, duplicate_groups)
            
            # Generate duplicate images report
            if duplicate_groups:
                dup_report = reports_dir / f"duplicate_images_{timestamp}.txt"
                self.generate_duplicate_report(dup_report, duplicate_groups)
                
            self.logger.info(f"✓ Reports generated in: {reports_dir}")
            return reports_dir
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
            
    def generate_html_report(self, html_path, duplicate_groups):
        """Generate HTML forensic image analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Forensic Image Analysis Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 40px; background: #0a0a0a; color: #00ff00; }}
        .header {{ background: #1a1a1a; padding: 20px; border: 2px solid #00ff00; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #1a1a1a; padding: 15px; border: 1px solid #00ff00; text-align: center; }}
        .image-analysis {{ margin: 20px 0; padding: 15px; border: 1px solid #333; background: #1a1a1a; }}
        .high-risk {{ border-left: 5px solid #ff0000; }}
        .medium-risk {{ border-left: 5px solid #ffaa00; }}
        .low-risk {{ border-left: 5px solid #00ff00; }}
        .anomalies {{ color: #ff4444; background: #2a1a1a; padding: 10px; margin: 10px 0; }}
        .metadata {{ color: #aaaaaa; font-size: 12px; }}
        .duplicates {{ background: #2a2a1a; padding: 15px; margin: 20px 0; border: 1px solid #ffaa00; }}
        h1, h2, h3 {{ color: #00ff00; }}
        .hash {{ font-family: 'Courier New', monospace; font-size: 10px; word-break: break-all; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📸 FORENSIC IMAGE ANALYSIS REPORT</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analyzer: Forensic Image Analyzer v1.0.0</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{self.analysis_stats['analyzed_images']}</h3>
            <p>Images Analyzed</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['suspicious_images']}</h3>
            <p>Suspicious Images</p>
        </div>
        <div class="metric">
            <h3>{self.analysis_stats['metadata_extracted']}</h3>
            <p>Metadata Extracted</p>
        </div>
        <div class="metric">
            <h3>{len(duplicate_groups)}</h3>
            <p>Duplicate Groups</p>
        </div>
    </div>
"""
        
        # Add duplicate groups section
        if duplicate_groups:
            html_content += """
    <h2>🔄 Duplicate Image Groups</h2>
"""
            for i, group in enumerate(duplicate_groups, 1):
                html_content += f"""
    <div class="duplicates">
        <h3>Group {i} ({len(group)} images)</h3>
        <ul>
"""
                for image in group:
                    path = Path(image['file_path'])
                    html_content += f"            <li>{path.name}</li>\n"
                html_content += "        </ul>\n    </div>\n"
        
        # Add individual image analyses
        html_content += "\n    <h2>📋 Individual Image Analysis</h2>\n"
        
        for analysis in self.analysis_results:
            if 'error' in analysis:
                continue
                
            risk_score = analysis.get('risk_score', 0)
            risk_class = 'high-risk' if risk_score > 60 else 'medium-risk' if risk_score > 30 else 'low-risk'
            
            html_content += f"""
    <div class="image-analysis {risk_class}">
        <h3>🖼️ {Path(analysis['file_path']).name}</h3>
        <p><strong>Risk Score:</strong> {risk_score}/100</p>
        
        <div class="metadata">
"""
            
            if 'metadata' in analysis and 'basic_info' in analysis['metadata']:
                basic = analysis['metadata']['basic_info']
                html_content += f"""
            <p><strong>Format:</strong> {basic.get('format', 'Unknown')}</p>
            <p><strong>Dimensions:</strong> {basic.get('width', 0)} x {basic.get('height', 0)}</p>
            <p><strong>Mode:</strong> {basic.get('mode', 'Unknown')}</p>
"""
            
            if 'hashes' in analysis:
                hashes = analysis['hashes']
                html_content += f"""
            <p><strong>SHA256:</strong> <span class="hash">{hashes.get('file_sha256', 'N/A')}</span></p>
            <p><strong>Perceptual Hash:</strong> <span class="hash">{hashes.get('perception_hash', 'N/A')}</span></p>
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
            
    def generate_duplicate_report(self, txt_path, duplicate_groups):
        """Generate text report of duplicate images"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("DUPLICATE IMAGES REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Groups: {len(duplicate_groups)}\n\n")
            
            for i, group in enumerate(duplicate_groups, 1):
                f.write(f"GROUP {i} ({len(group)} images):\n")
                f.write("-" * 30 + "\n")
                
                for image in group:
                    path = Path(image['file_path'])
                    f.write(f"  {path.name}\n")
                    
                    # Add hash information
                    if 'hashes' in image:
                        hashes = image['hashes']
                        f.write(f"    Perceptual Hash: {hashes.get('perception_hash', 'N/A')}\n")
                        f.write(f"    MD5: {hashes.get('file_md5', 'N/A')}\n")
                        
                f.write("\n")
                
    def show_completion_summary(self, reports_dir):
        """Show analysis completion summary"""
        try:
            elapsed = time.time() - self.analysis_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""Forensic Image Analysis Completed!

Images Analyzed: {self.analysis_stats['analyzed_images']}
Suspicious Images: {self.analysis_stats['suspicious_images']}
Metadata Extracted: {self.analysis_stats['metadata_extracted']}
Duplicate Groups: {self.analysis_stats['duplicate_groups']}

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
        parser = argparse.ArgumentParser(description="Advanced Forensic Image Analyzer")
        parser.add_argument('--input', help='Input directory with images')
        parser.add_argument('--output', help='Output directory for reports')
        parser.add_argument('--steg', action='store_true', help='Include steganography analysis')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Forensic Image Analyzer v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting forensic image analysis...")
        
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
                
            # Find image files
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(input_dir.rglob(f"*{ext}"))
                image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
                
            if not image_files:
                self.logger.warning("No image files found in directory")
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
                
            self.analysis_stats['total_images'] = len(image_files)
            self.analysis_stats['start_time'] = time.time()
            
            self.logger.info(f"Analyzing {len(image_files)} images...")
            
            # Create progress window
            self.create_progress_window(len(image_files))
            
            # Process images with thread pool
            check_steganography = args.steg if args.steg else False
            
            with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
                future_to_image = {
                    executor.submit(self.analyze_image, image_file, check_steganography): image_file
                    for image_file in image_files
                }
                
                completed = 0
                for future in as_completed(future_to_image):
                    image_file = future_to_image[future]
                    completed += 1
                    
                    try:
                        analysis = future.result()
                        self.analysis_results.append(analysis)
                        self.update_progress(completed, len(image_files), image_file.name)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {image_file}: {e}")
                        
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
    analyzer = ForensicImageAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()