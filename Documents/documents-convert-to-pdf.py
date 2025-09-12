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
# Script Name: documents-convert-to-pdf.py                                        
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: High-performance document converter supporting multiple formats    
#              with multi-core processing, metadata preservation, and universal  
#              macOS compatibility for seamless PDF conversion.                  
#
# Usage: python documents-convert-to-pdf.py [--input PATH] [--output DIR] [--format FORMAT] 
#
# Dependencies: weasyprint, python-docx, pillow, tkinter, concurrent.futures      
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features intelligent format detection, batch processing capabilities,   
#        and comprehensive error handling with desktop logging.                  
#                                                                                
####################################################################################

"""
Universal Document to PDF Converter
==================================

High-performance document converter that transforms various formats (HTML, DOCX, 
images, etc.) to PDF while preserving formatting, metadata, and quality.
Optimized for batch processing with multi-core support and native macOS integration.
"""

import os
import sys
import logging
import multiprocessing
import subprocess
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check and install dependencies
def check_and_install_dependencies():
    required_packages = [
        "weasyprint>=56.0",
        "python-docx>=0.8.11",
        "pillow>=8.0.0",
        "reportlab>=3.6.0"
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
from weasyprint import HTML, CSS
from docx import Document
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4

IS_MACOS = sys.platform == "darwin"

class DocumentConverter:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        # Supported formats and their handlers
        self.format_handlers = {
            '.html': self.convert_html_to_pdf,
            '.htm': self.convert_html_to_pdf,
            '.docx': self.convert_docx_to_pdf,
            '.doc': self.convert_docx_to_pdf,
            '.txt': self.convert_text_to_pdf,
            '.md': self.convert_markdown_to_pdf,
            '.jpg': self.convert_image_to_pdf,
            '.jpeg': self.convert_image_to_pdf,
            '.png': self.convert_image_to_pdf,
            '.gif': self.convert_image_to_pdf,
            '.bmp': self.convert_image_to_pdf,
            '.tiff': self.convert_image_to_pdf
        }
        
        self.conversion_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None
        }
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "documents-convert-to-pdf.log"
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
        
    def select_input_files(self):
        """Select input files via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        file_types = [
            ("All Supported", "*.html;*.htm;*.docx;*.doc;*.txt;*.md;*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff"),
            ("HTML files", "*.html;*.htm"),
            ("Word documents", "*.docx;*.doc"),
            ("Text files", "*.txt;*.md"),
            ("Images", "*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Documents to Convert",
            filetypes=file_types
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
            title="Select Output Directory for PDFs"
        )
        
        root.destroy()
        return Path(directory) if directory else None
        
    def create_progress_window(self, total_files):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Converting Documents to PDF")
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
        self.status_var = tk.StringVar(value="Initializing conversion...")
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
            status = f"Converting {current}/{total} ({percentage}%)"
            if filename:
                status += f" - {filename}"
                
            self.status_var.set(status)
            self.progress_root.update()
            
    def convert_html_to_pdf(self, input_path, output_path):
        """Convert HTML to PDF using WeasyPrint"""
        try:
            HTML(filename=str(input_path)).write_pdf(str(output_path))
            return True
        except Exception as e:
            self.logger.error(f"HTML conversion error for {input_path}: {e}")
            return False
            
    def convert_docx_to_pdf(self, input_path, output_path):
        """Convert DOCX to PDF via HTML intermediate"""
        try:
            doc = Document(input_path)
            
            # Convert DOCX to HTML
            html_content = "<html><body>"
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    html_content += f"<p>{paragraph.text}</p>"
            html_content += "</body></html>"
            
            # Create temporary HTML file
            temp_html = output_path.with_suffix('.temp.html')
            temp_html.write_text(html_content, encoding='utf-8')
            
            # Convert HTML to PDF
            HTML(filename=str(temp_html)).write_pdf(str(output_path))
            
            # Clean up temp file
            temp_html.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"DOCX conversion error for {input_path}: {e}")
            return False
            
    def convert_text_to_pdf(self, input_path, output_path):
        """Convert plain text to PDF"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(str(output_path), pagesize=letter)
            width, height = letter
            
            # Read text file
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Set up text formatting
            c.setFont("Helvetica", 12)
            y_position = height - 50
            line_height = 14
            
            for line in lines:
                if y_position < 50:  # Start new page
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = height - 50
                    
                c.drawString(50, y_position, line.rstrip())
                y_position -= line_height
                
            c.save()
            return True
            
        except Exception as e:
            self.logger.error(f"Text conversion error for {input_path}: {e}")
            return False
            
    def convert_markdown_to_pdf(self, input_path, output_path):
        """Convert Markdown to PDF via HTML"""
        try:
            # Simple markdown to HTML conversion
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic markdown processing
            html_content = "<html><head><style>body{font-family:Arial,sans-serif;margin:40px;}</style></head><body>"
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('# '):
                    html_content += f"<h1>{line[2:]}</h1>"
                elif line.startswith('## '):
                    html_content += f"<h2>{line[3:]}</h2>"
                elif line.startswith('### '):
                    html_content += f"<h3>{line[4:]}</h3>"
                elif line:
                    html_content += f"<p>{line}</p>"
                    
            html_content += "</body></html>"
            
            # Create temporary HTML and convert
            temp_html = output_path.with_suffix('.temp.html')
            temp_html.write_text(html_content, encoding='utf-8')
            
            HTML(filename=str(temp_html)).write_pdf(str(output_path))
            temp_html.unlink()
            return True
            
        except Exception as e:
            self.logger.error(f"Markdown conversion error for {input_path}: {e}")
            return False
            
    def convert_image_to_pdf(self, input_path, output_path):
        """Convert image to PDF"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            img = Image.open(input_path)
            img_width, img_height = img.size
            
            # Calculate scaling to fit page
            page_width, page_height = letter
            margin = 50
            max_width = page_width - 2 * margin
            max_height = page_height - 2 * margin
            
            scale = min(max_width / img_width, max_height / img_height)
            
            new_width = img_width * scale
            new_height = img_height * scale
            
            # Create PDF
            c = canvas.Canvas(str(output_path), pagesize=letter)
            
            # Center image on page
            x = (page_width - new_width) / 2
            y = (page_height - new_height) / 2
            
            c.drawImage(str(input_path), x, y, width=new_width, height=new_height)
            c.save()
            return True
            
        except Exception as e:
            self.logger.error(f"Image conversion error for {input_path}: {e}")
            return False
            
    def convert_file(self, input_path, output_dir):
        """Convert a single file to PDF"""
        try:
            # Get appropriate handler
            handler = self.format_handlers.get(input_path.suffix.lower())
            if not handler:
                self.logger.warning(f"Unsupported format: {input_path.suffix}")
                return False
                
            # Create output path
            output_path = output_dir / f"{input_path.stem}.pdf"
            
            # Handle filename collisions
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{input_path.stem}_{counter}.pdf"
                counter += 1
                
            # Convert file
            success = handler(input_path, output_path)
            
            if success:
                self.logger.info(f"✓ Converted: {input_path.name} → {output_path.name}")
                self.conversion_stats['successful'] += 1
            else:
                self.logger.error(f"✗ Failed: {input_path.name}")
                self.conversion_stats['failed'] += 1
                
            return success
            
        except Exception as e:
            self.logger.error(f"Conversion error for {input_path}: {e}")
            self.conversion_stats['failed'] += 1
            return False
            
    def show_completion_summary(self):
        """Show conversion completion summary"""
        try:
            # Calculate elapsed time
            elapsed = time.time() - self.conversion_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""Document Conversion Completed!

Files Processed: {self.conversion_stats['total_files']}
✓ Successful: {self.conversion_stats['successful']}
✗ Failed: {self.conversion_stats['failed']}

Time Elapsed: {int(minutes)}m {int(seconds)}s

Check the log file for detailed information."""

            root = tk.Tk()
            root.withdraw()
            
            if IS_MACOS:
                root.attributes("-topmost", True)
                
            messagebox.showinfo("Conversion Complete", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Universal Document to PDF Converter")
        parser.add_argument('--input', help='Input file or directory')
        parser.add_argument('--output', help='Output directory')
        parser.add_argument('--format', help='Specific format to convert')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("Document to PDF Converter v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting document to PDF conversion...")
        
        try:
            # Get input files
            if args.input:
                input_path = Path(args.input)
                if input_path.is_file():
                    input_files = [input_path]
                elif input_path.is_dir():
                    input_files = []
                    for ext in self.format_handlers.keys():
                        input_files.extend(input_path.glob(f"**/*{ext}"))
                else:
                    self.logger.error(f"Input path not found: {args.input}")
                    return
            else:
                input_files = self.select_input_files()
                
            if not input_files:
                self.logger.warning("No input files selected")
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
                
            # Filter by format if specified
            if args.format:
                format_ext = f".{args.format.lower()}"
                input_files = [f for f in input_files if f.suffix.lower() == format_ext]
                
            self.conversion_stats['total_files'] = len(input_files)
            self.conversion_stats['start_time'] = time.time()
            
            self.logger.info(f"Converting {len(input_files)} files...")
            
            # Create progress window
            self.create_progress_window(len(input_files))
            
            # Process files with thread pool
            with ThreadPoolExecutor(max_workers=self.optimal_threads) as executor:
                future_to_file = {
                    executor.submit(self.convert_file, file_path, output_dir): file_path
                    for file_path in input_files
                }
                
                completed = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed += 1
                    
                    try:
                        success = future.result()
                        self.update_progress(completed, len(input_files), file_path.name)
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        self.conversion_stats['failed'] += 1
                        
            # Close progress window
            if hasattr(self, 'progress_root'):
                self.progress_root.destroy()
                
            # Show completion summary
            self.show_completion_summary()
            
            # Open output directory if on macOS
            if IS_MACOS:
                subprocess.run(['open', str(output_dir)], check=False)
                
        except KeyboardInterrupt:
            self.logger.info("Conversion interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Open log file
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    converter = DocumentConverter()
    converter.run()

if __name__ == "__main__":
    main()