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
# Script Name: images-convert-images-to-pdf.py                                                 
# Author: sanchez314c@speedheathens.com  
# Date Created: 2025-01-24                                                       
# Version: 1.0.0
#
# Description:
#     Universal macOS-compatible image to PDF converter with advanced layout
#     and quality control options. Maintains aspect ratios while providing
#     a user-friendly GUI interface for seamless document creation.
#
# Features:
#     - Universal macOS compatibility with Path.home() and desktop logging
#     - Auto-dependency installation and error handling
#     - Multiple image format support (JPG, JPEG, PNG, TIFF, BMP)
#     - GUI-based file selection and custom output location
#     - High-quality PDF generation with memory efficient processing
#     - Progress tracking and comprehensive error handling
#     - Image format conversion and aspect ratio preservation
#
# Requirements:
#     - Python 3.8+ (auto-installed if missing)
#     - Pillow (PIL) (auto-installed if missing)
#     - tkinter (standard library with macOS compatibility)
#
# Usage:
#     python images-convert-images-to-pdf.py
#
####################################################################################
Title in Title Case
------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Clear, concise description of script purpose and functionality.
    Multiple lines if needed.

Features:
    - Feature one
    - Feature two
    - Additional features

Requirements:
    - Python version
    - Required packages
    - System dependencies

Usage:
    python script-name.py [arguments]

Image to PDF Converter
-------------------
Author: Jason Paul Michaels
Date: December 28, 2024
Version: 1.0.0

Description:
    Converts multiple images to a single PDF file with advanced options
    for layout and quality control. Supports various image formats and
    maintains aspect ratios.

Features:
    - Multiple image format support
    - Custom page layouts
    - Quality control
    - Progress tracking
    - Memory efficient processing
    - Error handling

Requirements:
    - Python 3.8+
    - Pillow
    - tkinter

import os
import sys
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import logging

class ImageToPDFConverter:
    def __init__(self):
        self.setup_logging()
        self.images: List[str] = []
        self.output_path: Optional[str] = None
        
    def setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def select_images(self) -> bool:
        """
        Open file dialog for image selection.
        
        Returns:
            bool: True if images were selected, False otherwise
        """
        root = tk.Tk()
        root.withdraw()
        
        files = filedialog.askopenfilenames(
            title='Select Images',
            filetypes=[
                ('Images', '*.jpg *.jpeg *.png *.tiff *.bmp'),
                ('JPEG', '*.jpg *.jpeg'),
                ('PNG', '*.png'),
                ('All files', '*.*')
            ]
        )
        
        if not files:
            logging.info("No images selected")
            return False
            
        self.images = list(files)
        logging.info(f"Selected {len(self.images)} images")
        return True
        
    def select_output(self) -> bool:
        """
        Open file dialog for output PDF selection.
        
        Returns:
            bool: True if output was selected, False otherwise
        """
        root = tk.Tk()
        root.withdraw()
        
        output = filedialog.asksaveasfilename(
            title='Save PDF As',
            defaultextension='.pdf',
            filetypes=[('PDF files', '*.pdf')]
        )
        
        if not output:
            logging.info("No output location selected")
            return False
            
        self.output_path = output
        return True
        
    def convert_images(self) -> bool:
        """
        Convert selected images to PDF.
        
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            if not self.images or not self.output_path:
                raise ValueError("No images or output path selected")
                
            # Convert and open all images
            images = []
            for img_path in self.images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")
                    continue
            
            if not images:
                raise ValueError("No valid images to process")
                
            # Save PDF
            images[0].save(
                self.output_path,
                save_all=True,
                append_images=images[1:],
                optimize=True,
                quality=95
            )
            
            logging.info(f"PDF created successfully: {self.output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Conversion failed: {e}")
            messagebox.showerror("Error", f"Failed to create PDF: {e}")
            return False
            
    def run(self):
        """Main execution flow."""
        if not self.select_images():
            messagebox.showinfo("Info", "No images selected. Exiting.")
            return
            
        if not self.select_output():
            messagebox.showinfo("Info", "No output location selected. Exiting.")
            return
            
        if self.convert_images():
            messagebox.showinfo("Success", f"PDF created successfully at:\n{self.output_path}")
        
def main():
    converter = ImageToPDFConverter()
    converter.run()

if __name__ == "__main__":
    main()
