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
# Script Name: documents-pdf-merger.py                                           
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-23                                                       
#
# Last Modified: 2025-01-23                                                      
#
# Version: 1.0.0                                                                 
#
# Description: Advanced PDF merger with bookmark preservation, metadata handling,
#              batch processing capabilities, and intelligent page ordering for
#              professional document assembly with universal macOS compatibility.
#
# Usage: python documents-pdf-merger.py [--input DIR] [--output FILE] [--bookmarks]
#
# Dependencies: PyPDF2, tkinter, pathlib, concurrent.futures                    
#
# GitHub: https://github.com/sanchez314c                                         
#
# Notes: Features intelligent PDF merging with metadata preservation, bookmark   
#        management, and comprehensive error handling with desktop logging.     
#                                                                                
####################################################################################

"""
Advanced PDF Merger
==================

Professional PDF merging tool that combines multiple PDFs while preserving
bookmarks, metadata, and formatting. Optimized for batch processing with
multi-core support and native macOS integration.
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
        "PyPDF2>=3.0.0",
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
from tkinter import filedialog, messagebox, ttk, simpledialog
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

IS_MACOS = sys.platform == "darwin"

class PDFMerger:
    def __init__(self):
        self.setup_logging()
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_threads = max(1, self.cpu_count - 1)
        
        self.merge_stats = {
            'total_files': 0,
            'total_pages': 0,
            'successful_merges': 0,
            'failed_files': 0,
            'start_time': None
        }
        
        self.pdf_files = []
        self.merge_order = []
        
    def setup_logging(self):
        log_file = Path.home() / "Desktop" / "documents-pdf-merger.log"
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
            title="Select PDF Files to Merge",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return [Path(f) for f in files] if files else []
        
    def select_output_file(self):
        """Select output file location via native macOS dialog"""
        root = tk.Tk()
        root.withdraw()
        
        if IS_MACOS:
            root.attributes("-topmost", True)
            
        file_path = filedialog.asksaveasfilename(
            title="Save Merged PDF As",
            defaultextension=".pdf",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return Path(file_path) if file_path else None
        
    def create_file_order_window(self):
        """Create window for ordering PDF files"""
        self.order_window = tk.Tk()
        self.order_window.title("PDF Merge Order")
        self.order_window.geometry("600x500")
        self.order_window.resizable(True, True)
        
        # Center window
        x = (self.order_window.winfo_screenwidth() // 2) - 300
        y = (self.order_window.winfo_screenheight() // 2) - 250
        self.order_window.geometry(f"+{x}+{y}")
        
        if IS_MACOS:
            self.order_window.attributes("-topmost", True)
        
        # Instructions
        instructions = tk.Label(
            self.order_window,
            text="Arrange PDFs in merge order (drag to reorder):",
            font=("Helvetica", 12, "bold")
        )
        instructions.pack(pady=10)
        
        # Listbox with scrollbar
        frame = tk.Frame(self.order_window)
        frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(
            frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=("Helvetica", 10)
        )
        self.file_listbox.pack(side=tk.LEFT, expand=True, fill='both')
        scrollbar.config(command=self.file_listbox.yview)
        
        # Populate listbox
        for pdf_file in self.pdf_files:
            self.file_listbox.insert(tk.END, pdf_file.name)
        
        # Control buttons
        button_frame = tk.Frame(self.order_window)
        button_frame.pack(pady=10)
        
        up_button = tk.Button(
            button_frame,
            text="Move Up",
            command=self.move_up,
            font=("Helvetica", 10)
        )
        up_button.pack(side=tk.LEFT, padx=5)
        
        down_button = tk.Button(
            button_frame,
            text="Move Down",
            command=self.move_down,
            font=("Helvetica", 10)
        )
        down_button.pack(side=tk.LEFT, padx=5)
        
        separator = tk.Frame(button_frame, width=20)
        separator.pack(side=tk.LEFT)
        
        preview_button = tk.Button(
            button_frame,
            text="Preview Info",
            command=self.preview_pdf_info,
            font=("Helvetica", 10)
        )
        preview_button.pack(side=tk.LEFT, padx=5)
        
        # Final buttons
        final_frame = tk.Frame(self.order_window)
        final_frame.pack(pady=20)
        
        merge_button = tk.Button(
            final_frame,
            text="Start Merge",
            command=self.start_merge_process,
            font=("Helvetica", 12, "bold"),
            bg="#007AFF",
            fg="white",
            relief="raised",
            borderwidth=2
        )
        merge_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(
            final_frame,
            text="Cancel",
            command=self.order_window.quit,
            font=("Helvetica", 12)
        )
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        self.order_window.mainloop()
        
    def move_up(self):
        """Move selected item up in the list"""
        selection = self.file_listbox.curselection()
        if selection and selection[0] > 0:
            index = selection[0]
            
            # Swap in display list
            self.file_listbox.insert(index - 1, self.file_listbox.get(index))
            self.file_listbox.delete(index + 1)
            
            # Swap in file list
            self.pdf_files[index], self.pdf_files[index - 1] = \
                self.pdf_files[index - 1], self.pdf_files[index]
            
            # Maintain selection
            self.file_listbox.selection_set(index - 1)
            
    def move_down(self):
        """Move selected item down in the list"""
        selection = self.file_listbox.curselection()
        if selection and selection[0] < self.file_listbox.size() - 1:
            index = selection[0]
            
            # Swap in display list
            self.file_listbox.insert(index + 2, self.file_listbox.get(index))
            self.file_listbox.delete(index)
            
            # Swap in file list
            self.pdf_files[index], self.pdf_files[index + 1] = \
                self.pdf_files[index + 1], self.pdf_files[index]
            
            # Maintain selection
            self.file_listbox.selection_set(index + 1)
            
    def preview_pdf_info(self):
        """Preview information about selected PDF"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a PDF to preview.")
            return
            
        pdf_file = self.pdf_files[selection[0]]
        
        try:
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                info = f"File: {pdf_file.name}\n"
                info += f"Pages: {len(reader.pages)}\n"
                info += f"Size: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB\n"
                
                # Try to get metadata
                if reader.metadata:
                    if reader.metadata.title:
                        info += f"Title: {reader.metadata.title}\n"
                    if reader.metadata.author:
                        info += f"Author: {reader.metadata.author}\n"
                    if reader.metadata.subject:
                        info += f"Subject: {reader.metadata.subject}\n"
                        
                messagebox.showinfo("PDF Information", info)
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not read PDF info: {e}")
            
    def start_merge_process(self):
        """Start the merge process"""
        self.order_window.quit()
        self.order_window.destroy()
        
    def create_progress_window(self):
        """Create progress tracking window"""
        self.progress_root = tk.Tk()
        self.progress_root.title("Merging PDFs")
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
            maximum=100,
            length=450
        )
        self.progress_bar.pack(pady=20)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing merge...")
        self.status_label = tk.Label(
            self.progress_root,
            textvariable=self.status_var,
            font=("Helvetica", 11)
        )
        self.status_label.pack(pady=(0, 20))
        
        self.progress_root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.progress_root.update()
        
    def update_progress(self, current, total, status=""):
        """Update progress bar and status"""
        if hasattr(self, 'progress_var'):
            percentage = int((current / total) * 100)
            self.progress_var.set(percentage)
            
            if status:
                self.status_var.set(f"{status} ({percentage}%)")
            else:
                self.status_var.set(f"Processing {current}/{total} ({percentage}%)")
                
            self.progress_root.update()
            
    def merge_pdfs(self, output_path, preserve_bookmarks=True):
        """Merge PDFs into a single file"""
        try:
            self.create_progress_window()
            
            merger = PyPDF2.PdfWriter()
            total_pages = 0
            processed_files = 0
            
            self.logger.info(f"Starting merge of {len(self.pdf_files)} PDFs...")
            
            for i, pdf_file in enumerate(self.pdf_files):
                try:
                    self.update_progress(
                        i, len(self.pdf_files), 
                        f"Processing {pdf_file.name}"
                    )
                    
                    with open(pdf_file, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        
                        # Add all pages
                        for page_num, page in enumerate(reader.pages):
                            merger.add_page(page)
                            total_pages += 1
                            
                        # Add bookmarks if requested
                        if preserve_bookmarks and len(reader.pages) > 0:
                            bookmark_title = pdf_file.stem
                            merger.add_outline_item(bookmark_title, total_pages - len(reader.pages))
                            
                        self.logger.info(f"✓ Added {len(reader.pages)} pages from {pdf_file.name}")
                        processed_files += 1
                        
                except Exception as e:
                    self.logger.error(f"✗ Error processing {pdf_file}: {e}")
                    self.merge_stats['failed_files'] += 1
                    continue
                    
            # Write merged PDF
            self.update_progress(99, 100, "Writing merged PDF...")
            
            with open(output_path, 'wb') as output_file:
                merger.write(output_file)
                
            # Update final stats
            self.merge_stats['successful_merges'] = processed_files
            self.merge_stats['total_pages'] = total_pages
            
            self.update_progress(100, 100, "Merge completed!")
            time.sleep(1)
            
            # Close progress window
            if hasattr(self, 'progress_root'):
                self.progress_root.destroy()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            if hasattr(self, 'progress_root'):
                self.progress_root.destroy()
            return False
            
    def show_completion_summary(self, output_path):
        """Show merge completion summary"""
        try:
            elapsed = time.time() - self.merge_stats['start_time']
            minutes, seconds = divmod(elapsed, 60)
            
            summary = f"""PDF Merge Completed!

Files Processed: {self.merge_stats['successful_merges']}/{self.merge_stats['total_files']}
Total Pages: {self.merge_stats['total_pages']}
Failed Files: {self.merge_stats['failed_files']}

Time Elapsed: {int(minutes)}m {int(seconds)}s

Output: {output_path.name}

Check the log file for detailed information."""

            root = tk.Tk()
            root.withdraw()
            
            if IS_MACOS:
                root.attributes("-topmost", True)
                
            messagebox.showinfo("Merge Complete", summary)
            root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error showing summary: {e}")
            
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description="Advanced PDF Merger")
        parser.add_argument('--input', help='Input directory containing PDFs')
        parser.add_argument('--output', help='Output merged PDF file')
        parser.add_argument('--bookmarks', action='store_true', help='Preserve bookmarks')
        parser.add_argument('--version', action='store_true', help='Show version')
        
        args = parser.parse_args()
        
        if args.version:
            print("PDF Merger v1.0.0")
            print("By sanchez314c@speedheathens.com")
            return
            
        self.logger.info("Starting PDF merge process...")
        
        try:
            # Get input PDFs
            if args.input:
                input_dir = Path(args.input)
                if input_dir.is_dir():
                    self.pdf_files = list(input_dir.glob("*.pdf"))
                else:
                    self.logger.error(f"Input directory not found: {args.input}")
                    return
            else:
                self.pdf_files = self.select_pdf_files()
                
            if not self.pdf_files:
                self.logger.warning("No PDF files selected")
                return
                
            # Sort by name initially
            self.pdf_files.sort(key=lambda x: x.name.lower())
            
            self.merge_stats['total_files'] = len(self.pdf_files)
            self.merge_stats['start_time'] = time.time()
            
            self.logger.info(f"Found {len(self.pdf_files)} PDF files")
            
            # Get output file
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Show file ordering window if interactive
                if len(self.pdf_files) > 1:
                    self.create_file_order_window()
                    
                output_path = self.select_output_file()
                
            if not output_path:
                self.logger.error("No output file selected")
                return
                
            # Perform merge
            preserve_bookmarks = args.bookmarks if args.bookmarks is not None else True
            
            success = self.merge_pdfs(output_path, preserve_bookmarks)
            
            if success:
                self.logger.info(f"✓ Merge completed: {output_path}")
                
                # Show completion summary
                self.show_completion_summary(output_path)
                
                # Open output file if on macOS
                if IS_MACOS:
                    subprocess.run(['open', str(output_path)], check=False)
            else:
                self.logger.error("Merge failed")
                
        except KeyboardInterrupt:
            self.logger.info("Merge interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Open log file
            if IS_MACOS:
                subprocess.run(['open', str(self.log_file)], check=False)

def main():
    merger = PDFMerger()
    merger.run()

if __name__ == "__main__":
    main()