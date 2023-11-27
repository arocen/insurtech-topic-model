# Extract txt from pdf analyse reports.

import pdfplumber
import os
from dotenv import load_dotenv

load_dotenv()

analyse_report_folder_pingan = os.environ.get("analyse_report_folder_pingan")

def extract_by_year(parent_folder):
    '''Extract txt from pdf files in each child folder of parent_folder, save txt files in each  child folder.'''
    child_directories = get_child_directories(parent_folder)
    for directory in child_directories:
        pdfs = sorted([f for f in os.listdir(directory) if f.endswith(".pdf")])

        # Get full paths of pdf
        pdfs = [os.path.join(directory, f) for f in pdfs]

        for pdf in pdfs:
            # Pdf to txt
            try:
                text = pdf2txt(pdf)
            except:
                print("Error parsing this pdf:", os.path.basename(pdf))
                continue

            # save
            pdf_filename = os.path.basename(pdf)
            txt_filename = pdf_filename.split(".")[0] + ".txt"
            save_path = os.path.join(directory, txt_filename)
            with open(save_path, 'w', encoding='UTF-8') as f:
                f.write(text)
    return

def get_child_directories(parent_directory):
    '''Get path to each child directory.'''
    child_directories = []
    
    # Get a list of all entries in the parent directory
    entries = os.listdir(parent_directory)
    
    for entry in entries:
        # Create the full path by joining the parent directory with the entry
        full_path = os.path.join(parent_directory, entry)
        
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            child_directories.append(full_path)
    
    return child_directories

def pdf2txt(file_address):
    res_pdf = pdfplumber.open(file_address)
    all_text = ""
    for page in res_pdf.pages:
        text = page.extract_text()
        all_text = all_text + "\n" + text
    return all_text


extract_by_year(analyse_report_folder_pingan)