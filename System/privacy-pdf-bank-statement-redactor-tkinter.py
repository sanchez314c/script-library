import fitz  # PyMuPDF
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def redact_pdf_file(input_pdf_path, output_pdf_path):
    """
    Redacts account numbers from a single PDF file using PyMuPDF,
    preserving the original layout.

    Args:
        input_pdf_path (str): The path to the input PDF file.
        output_pdf_path (str): The path to save the redacted PDF file.
    """
    # --- You can customize the patterns to find here ---
    # This list of regular expressions can be expanded to be more specific.
    regex_patterns = [
        r'\b\d{8,17}\b',  # Matches standalone sequences of 8 to 17 digits.
        r'Account Number:?\s*[\d-]+\d', # Matches "Account Number: 12345"
        r'Acct No:?\s*[\d-]+\d',        # Matches "Acct No: 12345"
    ]

    try:
        # Open the PDF using PyMuPDF
        doc = fitz.open(input_pdf_path)

        # Iterate through each page of the document
        for page in doc:
            # Search for each regex pattern on the page
            for pattern in regex_patterns:
                areas_to_redact = page.search_for(pattern)

                # Add a redaction annotation for each area found
                for area in areas_to_redact:
                    page.add_redact_annot(area, fill=(0, 0, 0)) # Fills with black

            # Apply the redactions to permanently remove the content
            page.apply_redactions()

        # Save the redacted PDF to the specified output path
        doc.save(output_pdf_path)
        doc.close()

    except Exception as e:
        # If a file is corrupted or not a PDF, it will raise an exception
        # We'll print the error and continue with the next files.
        print(f"Error processing file {os.path.basename(input_pdf_path)}: {e}")
        # Ensure the document is closed even if an error occurs
        if 'doc' in locals() and doc.is_open:
            doc.close()
        # Raise the exception again to be caught by the main processing loop
        raise

def process_pdfs_in_folder(input_folder, output_folder):
    """
    Processes all PDF files in a given folder and its subfolders.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    processed_files = 0
    error_files = 0

    # Walk through the input folder recursively
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                input_pdf_path = os.path.join(root, file)
                
                # Create a corresponding output path to maintain the folder structure
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Prepend "redacted_" to the original filename
                output_pdf_path = os.path.join(output_subfolder, f"redacted_{file}")

                try:
                    redact_pdf_file(input_pdf_path, output_pdf_path)
                    print(f"Successfully processed: {file}")
                    processed_files += 1
                except Exception:
                    # The specific error is already printed in the redact_pdf_file function
                    print(f"Failed to process: {file}")
                    error_files += 1
    
    return processed_files, error_files

def select_folders_and_process():
    """
    Main function to launch the Tkinter GUI for folder selection
    and start the redaction process.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    messagebox.showinfo("Instructions", "You will be asked to select two folders:\n\n1. The INPUT folder containing your original PDFs.\n2. The OUTPUT folder where the redacted PDFs will be saved.")

    # Ask the user to select the input folder
    input_folder = filedialog.askdirectory(title="Select Input Folder with PDFs")
    if not input_folder:
        messagebox.showerror("Error", "No input folder selected. Aborting.")
        return

    # Ask the user to select the output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder for Redacted PDFs")
    if not output_folder:
        messagebox.showerror("Error", "No output folder selected. Aborting.")
        return
        
    if input_folder == output_folder:
        messagebox.showerror("Error", "Input and Output folders cannot be the same. Aborting.")
        return

    # Process the PDFs and show a summary message
    try:
        processed_count, error_count = process_pdfs_in_folder(input_folder, output_folder)
        success_message = f"Processing complete!\n\nSuccessfully redacted: {processed_count} files.\nFailed to process: {error_count} files."
        if error_count > 0:
            success_message += "\n\nCheck the console window for specific error details."
        messagebox.showinfo("Success", success_message)
    except Exception as e:
        messagebox.showerror("An Unexpected Error Occurred", f"A critical error occurred: {str(e)}")

if __name__ == "__main__":
    select_folders_and_process()