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
# Script Name: forensics-steganography-detector.py                               
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced steganography detection tool with LSB analysis, entropy 
#              calculation, statistical testing, and pattern recognition for    
#              identifying hidden data in digital images and multimedia files.  
#
# Usage: python forensics-steganography-detector.py [--input PATH] [--output DIR]
#
# Dependencies: Pillow, numpy, scipy, matplotlib, opencv-python               
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Professional steganography detection with statistical analysis,        
#        entropy calculation, and comprehensive reporting for investigations.   
#                                                                                
####################################################################################

"""
Advanced Forensic Steganography Detector
=======================================

Comprehensive steganography detection tool for digital forensics investigations.
Performs LSB analysis, entropy calculation, statistical testing, and pattern
recognition to identify hidden data in digital media files.
"""

import os
import sys
import logging
import multiprocessing
import subprocess
import argparse
import time
import json
import math
import base64
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import statistics

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "Pillow>=8.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0"
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
from PIL import Image
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cv2

IS_MACOS = sys.platform == "darwin"

class SteganographyDetector:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        # Supported file formats for analysis
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Detection thresholds
        self.thresholds = {
            'lsb_entropy_threshold': 0.9,      # High entropy in LSB suggests hidden data
            'chi_square_threshold': 3.84,      # Chi-square critical value (p=0.05, df=1)
            'pattern_anomaly_threshold': 0.1,   # Pattern deviation threshold
            'frequency_anomaly_threshold': 0.05 # Frequency distribution anomaly
        }
        
        self.analysis_stats = {
            'total_files': 0,
            'analyzed_files': 0,
            'suspicious_files': 0,
            'hidden_data_detected': 0,
            'start_time': None
        }
        
        self.analysis_results = []
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "forensics-steganography-detector.log"
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
            title="Select Directory with Images for Steganography Analysis"
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
        self.progress_root.title("Steganography Detection Analysis")
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
        self.status_var = tk.StringVar(value="Initializing steganography analysis...")
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
            
    def calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
            
        # Count frequency of each value
        value_counts = defaultdict(int)
        for value in data:
            value_counts[value] += 1
            
        # Calculate probabilities and entropy
        length = len(data)
        entropy = 0.0
        
        for count in value_counts.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
                
        return entropy
        
    def extract_lsb_bits(self, image_array, channel=0):
        """Extract least significant bits from image channel"""
        if len(image_array.shape) == 3:
            channel_data = image_array[:, :, channel]
        else:
            channel_data = image_array
            
        # Extract LSB from each pixel
        lsb_bits = []
        for row in channel_data:
            for pixel in row:
                lsb_bits.append(pixel & 1)
                
        return lsb_bits
        
    def analyze_lsb_entropy(self, image_path):
        """Analyze LSB entropy for steganography detection"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img_array = np.array(img)
                
                lsb_analysis = {}
                
                # Analyze each color channel
                for channel, channel_name in enumerate(['red', 'green', 'blue']):
                    lsb_bits = self.extract_lsb_bits(img_array, channel)
                    
                    # Calculate entropy of LSB
                    lsb_entropy = self.calculate_entropy(lsb_bits)
                    lsb_analysis[f'{channel_name}_lsb_entropy'] = lsb_entropy
                    
                    # Chi-square test for randomness
                    ones = sum(lsb_bits)
                    zeros = len(lsb_bits) - ones
                    expected = len(lsb_bits) / 2
                    
                    if expected > 0:
                        chi_square = ((ones - expected) ** 2 / expected) + ((zeros - expected) ** 2 / expected)
                        lsb_analysis[f'{channel_name}_chi_square'] = chi_square
                    else:
                        lsb_analysis[f'{channel_name}_chi_square'] = 0
                        
                    # Pattern analysis - check for sequential patterns
                    pattern_score = self.analyze_bit_patterns(lsb_bits)
                    lsb_analysis[f'{channel_name}_pattern_score'] = pattern_score
                    
                # Overall assessment
                avg_entropy = statistics.mean([
                    lsb_analysis['red_lsb_entropy'],
                    lsb_analysis['green_lsb_entropy'],
                    lsb_analysis['blue_lsb_entropy']
                ])
                
                lsb_analysis['average_lsb_entropy'] = avg_entropy
                lsb_analysis['high_entropy_suspicious'] = avg_entropy > self.thresholds['lsb_entropy_threshold']
                
                return lsb_analysis
                
        except Exception as e:
            self.logger.error(f"LSB analysis error for {image_path}: {e}")
            return {'error': str(e)}
            
    def analyze_bit_patterns(self, bits):
        """Analyze bit patterns for anomalies"""
        if len(bits) < 100:
            return 0
            
        # Look for repetitive patterns
        pattern_lengths = [2, 3, 4, 8]
        pattern_scores = []
        
        for length in pattern_lengths:
            patterns = defaultdict(int)
            
            for i in range(len(bits) - length + 1):
                pattern = tuple(bits[i:i+length])
                patterns[pattern] += 1
                
            # Calculate pattern distribution entropy
            total_patterns = len(bits) - length + 1
            pattern_probs = [count / total_patterns for count in patterns.values()]
            pattern_entropy = -sum(p * math.log2(p) for p in pattern_probs if p > 0)
            
            # Expected entropy for random patterns
            expected_entropy = length  # Maximum entropy for length-bit patterns
            
            # Score based on deviation from expected randomness
            if expected_entropy > 0:
                pattern_scores.append(pattern_entropy / expected_entropy)
                
        return statistics.mean(pattern_scores) if pattern_scores else 0
        
    def analyze_frequency_distribution(self, image_path):
        """Analyze pixel frequency distributions for anomalies"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img_array = np.array(img)
                
                freq_analysis = {}
                
                # Analyze each color channel
                for channel, channel_name in enumerate(['red', 'green', 'blue']):
                    channel_data = img_array[:, :, channel].flatten()
                    
                    # Calculate histogram
                    hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
                    
                    # Analyze even/odd value distribution
                    even_values = sum(hist[::2])  # Even indices (0, 2, 4, ...)
                    odd_values = sum(hist[1::2])  # Odd indices (1, 3, 5, ...)
                    total_pixels = len(channel_data)
                    
                    if total_pixels > 0:
                        even_ratio = even_values / total_pixels
                        odd_ratio = odd_values / total_pixels
                        
                        # Expected ratio for natural images is roughly 50/50
                        ratio_deviation = abs(even_ratio - odd_ratio)
                        freq_analysis[f'{channel_name}_even_odd_deviation'] = ratio_deviation
                        
                        # Check for unusual spikes in specific values
                        hist_std = np.std(hist)
                        hist_mean = np.mean(hist)
                        
                        # Find outlier frequencies
                        outliers = np.where(hist > hist_mean + 3 * hist_std)[0]
                        freq_analysis[f'{channel_name}_frequency_outliers'] = len(outliers)
                        
                        # Calculate distribution entropy
                        hist_probs = hist / total_pixels
                        hist_entropy = -sum(p * math.log2(p) for p in hist_probs if p > 0)
                        freq_analysis[f'{channel_name}_distribution_entropy'] = hist_entropy
                        
                return freq_analysis
                
        except Exception as e:
            self.logger.error(f"Frequency analysis error for {image_path}: {e}")
            return {'error': str(e)}
            
    def detect_embedded_files(self, image_path):
        """Detect potential embedded files in image"""
        embedded_analysis = {}
        
        try:
            # Read raw image data
            with open(image_path, 'rb') as f:
                raw_data = f.read()
                
            # Look for file signatures after image header
            signatures = {
                b'PK\x03\x04': 'ZIP/Office Document',
                b'%PDF': 'PDF Document',
                b'Rar!\x1a\x07\x00': 'RAR Archive',
                b'\x89PNG\r\n\x1a\n': 'PNG Image',
                b'\xff\xd8\xff': 'JPEG Image',
                b'GIF87a': 'GIF Image',
                b'GIF89a': 'GIF Image'
            }
            
            found_signatures = []
            
            # Skip the first 1KB to avoid image header
            search_data = raw_data[1024:]
            
            for signature, file_type in signatures.items():
                if signature in search_data:
                    offset = search_data.find(signature) + 1024
                    found_signatures.append({
                        'type': file_type,
                        'offset': offset,
                        'signature': signature.hex()
                    })
                    
            embedded_analysis['embedded_signatures'] = found_signatures
            
            # Look for Base64 encoded data patterns
            try:
                # Convert binary data to text and look for base64 patterns
                text_data = raw_data.decode('latin-1', errors='ignore')
                
                # Simple base64 detection - look for patterns of valid base64 characters
                import re
                base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
                base64_matches = re.findall(base64_pattern, text_data)
                
                valid_base64 = []
                for match in base64_matches[:10]:  # Limit to first 10 matches
                    try:
                        decoded = base64.b64decode(match)
                        if len(decoded) > 10:  # Only consider substantial data
                            valid_base64.append({
                                'length': len(match),
                                'decoded_length': len(decoded),
                                'sample': match[:50] + '...' if len(match) > 50 else match
                            })
                    except:
                        continue
                        
                embedded_analysis['base64_patterns'] = valid_base64
                
            except Exception:
                pass
                
            # Check for unusual file size ratio
            with Image.open(image_path) as img:
                expected_size = img.width * img.height * 3  # Rough estimate for RGB
                actual_size = len(raw_data)
                size_ratio = actual_size / expected_size if expected_size > 0 else 0
                
                embedded_analysis['size_ratio'] = size_ratio
                embedded_analysis['size_anomaly'] = size_ratio > 2.0  # Unusually large
                
        except Exception as e:
            self.logger.error(f"Embedded file detection error for {image_path}: {e}")
            embedded_analysis['error'] = str(e)
            
        return embedded_analysis
        
    def analyze_steganography(self, image_path):
        """Perform comprehensive steganography analysis"""
        try:
            analysis = {
                'file_path': str(image_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'lsb_analysis': {},
                'frequency_analysis': {},
                'embedded_analysis': {},
                'detection_indicators': [],
                'confidence_score': 0,
                'risk_level': 'low'
            }
            
            # LSB entropy analysis
            lsb_data = self.analyze_lsb_entropy(image_path)
            analysis['lsb_analysis'] = lsb_data
            
            # Frequency distribution analysis
            freq_data = self.analyze_frequency_distribution(image_path)
            analysis['frequency_analysis'] = freq_data
            
            # Embedded file detection
            embedded_data = self.detect_embedded_files(image_path)
            analysis['embedded_analysis'] = embedded_data
            
            # Evaluate detection indicators
            indicators = []
            confidence = 0
            
            # Check LSB entropy indicators
            if 'average_lsb_entropy' in lsb_data:
                avg_entropy = lsb_data['average_lsb_entropy']
                if avg_entropy > self.thresholds['lsb_entropy_threshold']:
                    indicators.append(f"High LSB entropy: {avg_entropy:.3f}")
                    confidence += 30
                    
            # Check chi-square test results
            for channel in ['red', 'green', 'blue']:
                chi_key = f'{channel}_chi_square'
                if chi_key in lsb_data:
                    chi_value = lsb_data[chi_key]
                    if chi_value > self.thresholds['chi_square_threshold']:
                        indicators.append(f"Anomalous {channel} channel chi-square: {chi_value:.3f}")
                        confidence += 15
                        
            # Check frequency distribution anomalies
            if 'error' not in freq_data:
                for channel in ['red', 'green', 'blue']:
                    deviation_key = f'{channel}_even_odd_deviation'
                    if deviation_key in freq_data:
                        deviation = freq_data[deviation_key]
                        if deviation > self.thresholds['frequency_anomaly_threshold']:
                            indicators.append(f"Unusual {channel} frequency distribution")
                            confidence += 10
                            
            # Check embedded file signatures
            if 'embedded_signatures' in embedded_data and embedded_data['embedded_signatures']:
                for sig in embedded_data['embedded_signatures']:
                    indicators.append(f"Embedded {sig['type']} detected at offset {sig['offset']}")
                    confidence += 40
                    
            # Check Base64 patterns
            if 'base64_patterns' in embedded_data and embedded_data['base64_patterns']:
                indicators.append(f"Potential Base64 encoded data found ({len(embedded_data['base64_patterns'])} patterns)")
                confidence += 20
                
            # Check size anomalies
            if 'size_anomaly' in embedded_data and embedded_data['size_anomaly']:
                indicators.append("Unusual file size for image dimensions")
                confidence += 15
                
            analysis['detection_indicators'] = indicators
            analysis['confidence_score'] = min(100, confidence)
            
            # Determine risk level
            if confidence >= 70:
                analysis['risk_level'] = 'high'
            elif confidence >= 40:
                analysis['risk_level'] = 'medium'
            else:
                analysis['risk_level'] = 'low'
                
            # Update statistics
            self.analysis_stats['analyzed_files'] += 1
            if indicators:
                self.analysis_stats['suspicious_files'] += 1
            if confidence >= 50:
                self.analysis_stats['hidden_data_detected'] += 1
                
            self.logger.info(f"✓ Analyzed: {image_path.name} (Confidence: {confidence}%, Risk: {analysis['risk_level']})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"✗ Steganography analysis failed for {image_path}: {e}")
            return {
                'file_path': str(image_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
            
    def generate_analysis_report(self, output_dir):
        """Generate comprehensive steganography analysis report"""
        try:
            # Create reports directory
            reports_dir = output_dir / "steganography_analysis_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Generate JSON report
            json_report = reports_dir / f"steganography_analysis_{timestamp}.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                report_data = {
                    'analysis_summary': dict(self.analysis_stats),
                    'detection_thresholds': self.thresholds,
                    'individual_analyses': self.analysis_results
                }
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generate HTML report
            html_report = reports_dir / f"steganography_analysis_{timestamp}.html"
            self.generate_html_report(html_report)
            
            # Generate suspicious files summary
            suspicious_files = [r for r in self.analysis_results if r.get('confidence_score', 0) >= 30]
            if suspicious_files:
                suspicious_report = reports_dir / f"suspicious_files_{timestamp}.txt"
                self.generate_suspicious_files_report(suspicious_report, suspicious_files)
                
            self.logger.info(f"✓ Reports generated in: {reports_dir}")
            return reports_dir
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
            
    def generate_html_report(self, html_path):
        """Generate HTML steganography analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Steganography Detection Analysis Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 40px; background: #0a0a0a; color: #00ff00; }}
        .header {{ background: #1a1a1a; padding: 20px; border: 2px solid #00ff00; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #1a1a1a; padding: 15px; border: 1px solid #00ff00; text-align: center; }}
        .file-analysis {{ margin: 20px 0; padding: 15px; border: 1px solid #333; background: #1a1a1a; }}
        .high-risk {{ border-left: 5px solid #ff0000; }}
        .medium-risk {{ border-left: 5px solid #ffaa00; }}
        .low-risk {{ border-left: 5px solid #00ff00; }}
        .indicators {{ color: #ff4444; background: #2a1a1a; padding: 10px; margin: 10px 0; }}
        .metadata {{ color: #aaaaaa; font-size: 12px; }}
        h1, h2, h3 {{ color: #00ff00; }}
        .confidence {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 STEGANOGRAPHY DETECTION ANALYSIS REPORT</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Analyzer: Forensic Steganography Detector v1.0.0</p>
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
            <h3>{self.analysis_stats['hidden_data_detected']}</h3>
            <p>Likely Hidden Data</p>
        </div>
        <div class="metric">
            <h3>{(self.analysis_stats['suspicious_files'] / max(1, self.analysis_stats['analyzed_files']) * 100):.1f}%</h3>
            <p>Detection Rate</p>
        </div>
    </div>
    
    <h2>📊 Detection Thresholds</h2>
    <div style="background: #1a1a1a; padding: 15px; border: 1px solid #00ff00;">
        <p>LSB Entropy Threshold: {self.thresholds['lsb_entropy_threshold']}</p>
        <p>Chi-Square Threshold: {self.thresholds['chi_square_threshold']}</p>
        <p>Pattern Anomaly Threshold: {self.thresholds['pattern_anomaly_threshold']}</p>
        <p>Frequency Anomaly Threshold: {self.thresholds['frequency_anomaly_threshold']}</p>
    </div>
    
    <h2>📋 Individual File Analysis</h2>
"""
        
        # Sort results by confidence score (highest first)
        sorted_results = sorted(self.analysis_results, 
                              key=lambda x: x.get('confidence_score', 0), 
                              reverse=True)
        
        for analysis in sorted_results:
            if 'error' in analysis:
                continue
                
            confidence = analysis.get('confidence_score', 0)
            risk_level = analysis.get('risk_level', 'low')
            risk_class = f'{risk_level}-risk'
            
            html_content += f"""
    <div class="file-analysis {risk_class}">
        <h3>🖼️ {Path(analysis['file_path']).name}</h3>
        <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
        <p class="confidence"><strong>Confidence Score:</strong> {confidence}/100</p>
        
        <div class="metadata">
"""
            
            # LSB analysis summary
            if 'lsb_analysis' in analysis and 'average_lsb_entropy' in analysis['lsb_analysis']:
                lsb = analysis['lsb_analysis']
                html_content += f"""
            <p><strong>Average LSB Entropy:</strong> {lsb['average_lsb_entropy']:.3f}</p>
            <p><strong>High Entropy Detected:</strong> {lsb.get('high_entropy_suspicious', False)}</p>
"""
            
            # Embedded files summary
            if 'embedded_analysis' in analysis:
                embedded = analysis['embedded_analysis']
                if embedded.get('embedded_signatures'):
                    html_content += f"            <p><strong>Embedded Files:</strong> {len(embedded['embedded_signatures'])} detected</p>\n"
                if embedded.get('base64_patterns'):
                    html_content += f"            <p><strong>Base64 Patterns:</strong> {len(embedded['base64_patterns'])} found</p>\n"
                    
            html_content += "        </div>\n"
            
            # Detection indicators
            if analysis.get('detection_indicators'):
                html_content += """
        <div class="indicators">
            <h4>🚨 Detection Indicators:</h4>
            <ul>
"""
                for indicator in analysis['detection_indicators']:
                    html_content += f"                <li>{indicator}</li>\n"
                html_content += "            </ul>\n        </div>\n"
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def generate_suspicious_files_report(self, txt_path, suspicious_files):
        """Generate text report of suspicious files"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("SUSPICIOUS FILES REPORT - STEGANOGRAPHY DETECTION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Suspicious Files Found: {len(suspicious_files)}\n\n")
            
            # Sort by confidence score
            sorted_files = sorted(suspicious_files, 
                                key=lambda x: x.get('confidence_score', 0), 
                                reverse=True)
            
            for i, analysis in enumerate(sorted_files, 1):
                f.write(f"FILE {i}: {Path(analysis['file_path']).name}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Confidence Score: {analysis.get('confidence_score', 0)}/100\n")
                f.write(f"Risk Level: {analysis.get('risk_level', 'unknown').upper()}\n")
                
                if analysis.get('detection_indicators'):
                    f.write("Detection Indicators:\n")
                    for indicator in analysis['detection_indicators']:
                        f.write(f"  • {indicator}\n")
                        
                f.write("\n")
                
    def show_completion_summary(self, reports_dir):
        """Show analysis completion summary"""
        try:
            elapsed = time.time() - self.analysis_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            detection_rate = (self.analysis_stats['suspicious_files'] / 
                            max(1, self.analysis_stats['analyzed_files']) * 100)
            
            summary = f"""Steganography Detection Analysis Completed!

Files Analyzed: {self.analysis_stats['analyzed_files']}
Suspicious Files: {self.analysis_stats['suspicious_files']}
Likely Hidden Data: {self.analysis_stats['hidden_data_detected']}
Detection Rate: {detection_rate:.1f}%

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
        parser = argparse.ArgumentParser(description="Advanced Forensic Steganography Detector")
        parser.add_argument('--input', help='Input directory with images')
        parser.add_argument('--output', help='Output directory for reports')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Forensic Steganography Detector v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting steganography detection analysis...")
        
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
                
            self.analysis_stats['total_files'] = len(image_files)
            self.analysis_stats['start_time'] = time.time()
            
            self.logger.info(f"Analyzing {len(image_files)} images for steganography...")
            
            # Create progress window
            self.create_progress_window(len(image_files))
            
            # Process images with thread pool
            with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
                future_to_image = {
                    executor.submit(self.analyze_steganography, image_file): image_file
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
    detector = SteganographyDetector()
    detector.run()

if __name__ == "__main__":
    main()