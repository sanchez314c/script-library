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
# Script Name: documents-pdf-review.py                                           
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Comprehensive PDF analysis and review tool with content extraction,
#              metadata inspection, quality assessment, and security validation  
#              for document integrity verification with universal macOS support.
#
# Usage: python documents-pdf-review.py [--input PATH] [--output DIR] [--deep-scan]
#
# Dependencies: PyPDF2, reportlab, tkinter, concurrent.futures, nltk           
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features intelligent content analysis, security scanning, metadata     
#        extraction, and comprehensive reporting with desktop logging.          
#                                                                                
####################################################################################

"""
Advanced PDF Review and Analysis Tool
====================================

Comprehensive PDF analyzer that extracts content, metadata, and security
information while performing quality assessments and integrity checks.
Optimized for batch processing with detailed reporting and macOS integration.
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

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "PyPDF2>=3.0.0",
        "reportlab>=3.6.0",
        "nltk>=3.7"
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
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

IS_MACOS = sys.platform == "darwin"

class PDFReviewer:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        self.review_stats = {
            'total_files': 0,
            'total_pages': 0,
            'total_text_chars': 0,
            'processed_files': 0,
            'failed_files': 0,
            'start_time': None
        }
        
        self.analysis_results = []
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "documents-pdf-review.log"
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
        
    def select_pdf_files(self):
        """Select PDF files via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        files = filedialog.askopenfilenames(
            title="Select PDF Files to Review",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return [Path(f) for f in files] if files else []
        
    def select_output_directory(self):
        """Select output directory via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        directory = filedialog.askdirectory(
            title="Select Output Directory for Reports"
        )
        
        root.destroy()
        return Path(directory) if directory else None
        
    def create_progress_window(self, total_files):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Analyzing PDFs")
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
        self.status_var = tk.StringVar(value="Initializing analysis...")
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
            
    def extract_pdf_metadata(self, pdf_path):
        """Extract comprehensive metadata from PDF"""
        metadata = {
            'file_info': {},
            'pdf_info': {},
            'security': {},
            'structure': {}
        }
        
        try:
            # File information
            stat = pdf_path.stat()
            metadata['file_info'] = {
                'filename': pdf_path.name,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / 1024 / 1024, 2),
                'created': time.ctime(stat.st_ctime),
                'modified': time.ctime(stat.st_mtime),
                'md5_hash': self.calculate_file_hash(pdf_path)
            }
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Basic PDF info
                metadata['pdf_info'] = {
                    'pages': len(reader.pages),
                    'version': getattr(reader, 'pdf_header', 'Unknown'),
                    'encrypted': reader.is_encrypted
                }
                
                # Document metadata
                if reader.metadata:
                    doc_meta = {}
                    for key, value in reader.metadata.items():
                        if value:
                            doc_meta[key.replace('/', '')] = str(value)
                    metadata['pdf_info']['document_metadata'] = doc_meta
                
                # Security information
                metadata['security'] = {
                    'encrypted': reader.is_encrypted,
                    'password_protected': reader.is_encrypted
                }
                
                # Structure analysis
                metadata['structure'] = {
                    'has_bookmarks': len(reader.outline) > 0 if hasattr(reader, 'outline') else False,
                    'bookmark_count': len(reader.outline) if hasattr(reader, 'outline') else 0
                }
                
        except Exception as e:
            self.logger.error(f"Metadata extraction error for {pdf_path}: {e}")
            metadata['error'] = str(e)
            
        return metadata
        
    def extract_pdf_text(self, pdf_path, max_pages=None):
        """Extract text content from PDF"""
        text_data = {
            'full_text': '',
            'page_texts': [],
            'statistics': {}
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                pages_to_process = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)
                
                for i in range(pages_to_process):
                    try:
                        page = reader.pages[i]
                        page_text = page.extract_text()
                        text_data['page_texts'].append(page_text)
                        text_data['full_text'] += page_text + '\n'
                    except Exception as e:
                        self.logger.warning(f"Could not extract text from page {i+1} of {pdf_path}: {e}")
                        text_data['page_texts'].append('')
                
                # Calculate statistics
                full_text = text_data['full_text']
                text_data['statistics'] = {
                    'total_characters': len(full_text),
                    'total_words': len(full_text.split()),
                    'pages_with_text': sum(1 for text in text_data['page_texts'] if text.strip()),
                    'average_chars_per_page': len(full_text) / len(text_data['page_texts']) if text_data['page_texts'] else 0
                }
                
                # Advanced text analysis if NLTK is available
                if NLTK_AVAILABLE and full_text.strip():
                    try:
                        sentences = sent_tokenize(full_text)
                        words = word_tokenize(full_text.lower())
                        stop_words = set(stopwords.words('english'))
                        filtered_words = [w for w in words if w.isalnum() and w not in stop_words]
                        
                        text_data['statistics'].update({
                            'sentences': len(sentences),
                            'unique_words': len(set(filtered_words)),
                            'reading_complexity': 'simple' if len(set(filtered_words)) < 1000 else 'complex'
                        })
                    except Exception as e:
                        self.logger.warning(f"NLTK analysis failed for {pdf_path}: {e}")
                
        except Exception as e:
            self.logger.error(f"Text extraction error for {pdf_path}: {e}")
            text_data['error'] = str(e)
            
        return text_data
        
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
            
    def analyze_pdf_quality(self, metadata, text_data):
        """Analyze PDF quality and provide recommendations"""
        quality_score = 100
        issues = []
        recommendations = []
        
        # File size analysis
        size_mb = metadata['file_info']['size_mb']
        if size_mb > 50:
            quality_score -= 10
            issues.append("Large file size")
            recommendations.append("Consider compressing the PDF")
        elif size_mb < 0.1:
            quality_score -= 5
            issues.append("Very small file size")
            
        # Text extraction quality
        if 'statistics' in text_data:
            stats = text_data['statistics']
            
            if stats['pages_with_text'] < metadata['pdf_info']['pages'] * 0.5:
                quality_score -= 20
                issues.append("Limited text extractable")
                recommendations.append("PDF may be image-based; consider OCR")
                
            if stats['average_chars_per_page'] < 100:
                quality_score -= 15
                issues.append("Low text density")
                
        # Security analysis
        if metadata['security']['encrypted']:
            quality_score -= 5
            issues.append("Password protected")
            
        # Structure analysis
        if not metadata['structure']['has_bookmarks'] and metadata['pdf_info']['pages'] > 10:
            quality_score -= 10
            issues.append("No bookmarks for multi-page document")
            recommendations.append("Add bookmarks for better navigation")
            
        return {
            'quality_score': max(0, quality_score),
            'grade': self.get_quality_grade(quality_score),
            'issues': issues,
            'recommendations': recommendations
        }
        
    def get_quality_grade(self, score):
        """Convert quality score to grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
            
    def analyze_pdf(self, pdf_path, deep_scan=False):
        """Perform comprehensive PDF analysis"""
        try:
            # Extract metadata
            metadata = self.extract_pdf_metadata(pdf_path)
            
            # Extract text (limit pages for performance unless deep scan)
            max_pages = None if deep_scan else 50
            text_data = self.extract_pdf_text(pdf_path, max_pages)
            
            # Quality analysis
            quality_analysis = self.analyze_pdf_quality(metadata, text_data)
            
            # Compile results
            analysis = {
                'file_path': str(pdf_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': metadata,
                'text_analysis': {
                    'statistics': text_data.get('statistics', {}),
                    'has_text': bool(text_data['full_text'].strip())
                },
                'quality_analysis': quality_analysis
            }
            
            # Update global stats
            if 'statistics' in text_data:
                self.review_stats['total_pages'] += metadata['pdf_info'].get('pages', 0)
                self.review_stats['total_text_chars'] += text_data['statistics'].get('total_characters', 0)
            
            self.review_stats['processed_files'] += 1
            self.logger.info(f"✓ Analyzed: {pdf_path.name} (Quality: {quality_analysis['grade']})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"✗ Analysis failed for {pdf_path}: {e}")
            self.review_stats['failed_files'] += 1
            return {
                'file_path': str(pdf_path),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
            
    def generate_analysis_report(self, output_dir):
        """Generate comprehensive analysis report"""
        try:
            # Create reports directory
            reports_dir = output_dir / "pdf_analysis_reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Generate JSON report
            json_report = reports_dir / f"pdf_analysis_{timestamp}.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                report_data = {
                    'analysis_summary': self.review_stats,
                    'individual_analyses': self.analysis_results
                }
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generate HTML report
            html_report = reports_dir / f"pdf_analysis_{timestamp}.html"
            self.generate_html_report(html_report)
            
            # Generate text summary
            txt_report = reports_dir / f"pdf_analysis_summary_{timestamp}.txt"
            self.generate_text_summary(txt_report)
            
            self.logger.info(f"✓ Reports generated in: {reports_dir}")
            return reports_dir
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
            
    def generate_html_report(self, html_path):
        """Generate HTML analysis report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PDF Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; text-align: center; }}
        .file-analysis {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .quality-A {{ border-left: 5px solid #28a745; }}
        .quality-B {{ border-left: 5px solid #17a2b8; }}
        .quality-C {{ border-left: 5px solid #ffc107; }}
        .quality-D {{ border-left: 5px solid #fd7e14; }}
        .quality-F {{ border-left: 5px solid #dc3545; }}
        .issues {{ color: #dc3545; }}
        .recommendations {{ color: #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PDF Analysis Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{self.review_stats['processed_files']}</h3>
            <p>Files Analyzed</p>
        </div>
        <div class="metric">
            <h3>{self.review_stats['total_pages']}</h3>
            <p>Total Pages</p>
        </div>
        <div class="metric">
            <h3>{self.review_stats['total_text_chars']:,}</h3>
            <p>Characters Extracted</p>
        </div>
        <div class="metric">
            <h3>{self.review_stats['failed_files']}</h3>
            <p>Failed Analyses</p>
        </div>
    </div>
"""
        
        # Add individual file analyses
        for analysis in self.analysis_results:
            if 'error' in analysis:
                continue
                
            quality = analysis.get('quality_analysis', {})
            grade = quality.get('grade', 'Unknown')
            
            html_content += f"""
    <div class="file-analysis quality-{grade}">
        <h3>{Path(analysis['file_path']).name}</h3>
        <p><strong>Quality Grade:</strong> {grade} ({quality.get('quality_score', 0)}/100)</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4>File Information</h4>
                <ul>
                    <li>Size: {analysis['metadata']['file_info'].get('size_mb', 0)} MB</li>
                    <li>Pages: {analysis['metadata']['pdf_info'].get('pages', 0)}</li>
                    <li>Encrypted: {analysis['metadata']['security'].get('encrypted', False)}</li>
                </ul>
            </div>
            <div>
                <h4>Text Analysis</h4>
                <ul>
                    <li>Characters: {analysis['text_analysis']['statistics'].get('total_characters', 0):,}</li>
                    <li>Words: {analysis['text_analysis']['statistics'].get('total_words', 0):,}</li>
                    <li>Has Text: {analysis['text_analysis'].get('has_text', False)}</li>
                </ul>
            </div>
        </div>
"""
            
            if quality.get('issues'):
                html_content += f"""
        <div class="issues">
            <h4>Issues Found:</h4>
            <ul>
                {''.join(f'<li>{issue}</li>' for issue in quality['issues'])}
            </ul>
        </div>
"""
            
            if quality.get('recommendations'):
                html_content += f"""
        <div class="recommendations">
            <h4>Recommendations:</h4>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in quality['recommendations'])}
            </ul>
        </div>
"""
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    def generate_text_summary(self, txt_path):
        """Generate text summary report"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("PDF ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Files Analyzed: {self.review_stats['processed_files']}\n")
            f.write(f"Failed Analyses: {self.review_stats['failed_files']}\n")
            f.write(f"Total Pages: {self.review_stats['total_pages']}\n")
            f.write(f"Total Characters: {self.review_stats['total_text_chars']:,}\n\n")
            
            # Quality distribution
            quality_dist = defaultdict(int)
            for analysis in self.analysis_results:
                if 'quality_analysis' in analysis:
                    grade = analysis['quality_analysis'].get('grade', 'Unknown')
                    quality_dist[grade] += 1
            
            f.write("QUALITY DISTRIBUTION:\n")
            for grade in ['A', 'B', 'C', 'D', 'F']:
                f.write(f"Grade {grade}: {quality_dist[grade]} files\n")
            f.write("\n")
            
            # Individual file summaries
            f.write("INDIVIDUAL FILE ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            
            for analysis in self.analysis_results:
                if 'error' in analysis:
                    f.write(f"❌ {Path(analysis['file_path']).name}: Analysis failed\n")
                    continue
                    
                filename = Path(analysis['file_path']).name
                quality = analysis.get('quality_analysis', {})
                grade = quality.get('grade', 'Unknown')
                score = quality.get('quality_score', 0)
                
                f.write(f"📄 {filename}: Grade {grade} ({score}/100)\n")
                
                if quality.get('issues'):
                    f.write(f"   Issues: {', '.join(quality['issues'])}\n")
                    
    def show_completion_summary(self, reports_dir):
        """Show analysis completion summary"""
        try:
            elapsed = time.time() - self.review_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""PDF Analysis Completed!

Files Analyzed: {self.review_stats['processed_files']}
Failed Analyses: {self.review_stats['failed_files']}
Total Pages: {self.review_stats['total_pages']}
Text Characters: {self.review_stats['total_text_chars']:,}

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
        parser = argparse.ArgumentParser(description="Advanced PDF Review and Analysis")
        parser.add_argument('--input', help='Input PDF file or directory')
        parser.add_argument('--output', help='Output directory for reports')
        parser.add_argument('--deep-scan', action='store_true', help='Perform deep content analysis')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("PDF Reviewer v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting PDF analysis and review...")
        
        try:
            # Get input PDFs
            if args.input:
                input_path = Path(args.input)
                if input_path.is_file() and input_path.suffix.lower() == '.pdf':
                    pdf_files = [input_path]
                elif input_path.is_dir():
                    pdf_files = list(input_path.glob("**/*.pdf"))
                else:
                    self.logger.error(f"Input path not found or invalid: {args.input}")
                    return
            else:
                pdf_files = self.select_pdf_files()
                
            if not pdf_files:
                self.logger.warning("No PDF files selected")
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
                
            self.review_stats['total_files'] = len(pdf_files)
            self.review_stats['start_time'] = time.time()
            
            self.logger.info(f"Analyzing {len(pdf_files)} PDF files...")
            
            # Create progress window
            self.create_progress_window(len(pdf_files))
            
            # Process files with thread pool
            deep_scan = args.deep_scan if args.deep_scan else False
            
            with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
                future_to_file = {
                    executor.submit(self.analyze_pdf, pdf_file, deep_scan): pdf_file
                    for pdf_file in pdf_files
                }
                
                completed = 0
                for future in as_completed(future_to_file):
                    pdf_file = future_to_file[future]
                    completed += 1
                    
                    try:
                        analysis = future.result()
                        self.analysis_results.append(analysis)
                        self.update_progress(completed, len(pdf_files), pdf_file.name)
                    except Exception as e:
                        self.logger.error(f"Error analyzing {pdf_file}: {e}")
                        self.review_stats['failed_files'] += 1
                        
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
    reviewer = PDFReviewer()
    reviewer.run()

if __name__ == "__main__":
    main()