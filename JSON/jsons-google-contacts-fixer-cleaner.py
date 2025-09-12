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
# Script Name: jsons-google-contacts-fixer-cleaner.py                                                 
# 
# Author: sanchez314c@speedheathens.com  
#                                              
# Date Created: 2025-01-24                                                       
#
# Last Modified: 2025-01-24                                                      
#
# Description: Enhanced Google Contacts Fixer & Cleaner - Advanced contact 
#              processing and standardization with intelligent deduplication.
#
# Version: 1.0.0
#
####################################################################################

import sys
import os
import subprocess
import json
import time
import platform
import traceback
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def install_dependencies():
    """Install required dependencies with enhanced error handling."""
    dependencies = [
        'pandas',
        'phonenumbers',
        'nameparser',
        'tkinter',
        'matplotlib',
        'seaborn'
    ]
    
    for package in dependencies:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--upgrade", "--quiet"
                ])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")
                return False
    return True

# Install dependencies before importing
if not install_dependencies():
    print("Failed to install required dependencies. Exiting.")
    sys.exit(1)

import pandas as pd
import phonenumbers
from nameparser import HumanName
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import seaborn as sns

class GoogleContactsProcessor:
    """Advanced Google Contacts processing and standardization suite."""
    
    def __init__(self):
        """Initialize the Google Contacts processor."""
        self.start_time = time.time()
        self.desktop_path = Path.home() / "Desktop"
        self.log_file = self.desktop_path / f"google_contacts_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.platform_info = self._get_platform_info()
        
        # Ensure desktop directory exists
        self.desktop_path.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Processing statistics
        self.stats = {
            'contacts_processed': 0,
            'names_fixed': 0,
            'phones_standardized': 0,
            'duplicates_found': 0,
            'errors_encountered': 0,
            'quality_improvements': 0
        }
        
        # Name processing patterns
        self.name_prefixes = {
            'mc', 'mac', 'o\'', 'de', 'del', 'della', 'von', 'van', 'la', 'le',
            'san', 'santa', 'st', 'saint', 'dos', 'das', 'bin', 'ibn'
        }
        
        # Business name indicators
        self.business_indicators = {
            'llc', 'inc', 'corp', 'ltd', 'company', 'co', 'agency', 'group',
            'services', 'solutions', 'consulting', 'enterprises', 'associates',
            'partners', 'firm', 'studio', 'shop', 'store', 'restaurant', 'cafe',
            'hotel', 'motel', 'clinic', 'hospital', 'center', 'centre'
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"Google Contacts Processor Log - {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
    
    def _log(self, message: str, level: str = "INFO"):
        """Log message to file and optionally print to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception:
            pass  # Fail silently for logging issues
        
        print(f"[{level}] {message}")
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            'system': platform.system(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def select_input_file(self) -> Optional[str]:
        """Select input contacts file using GUI dialog."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                title="Select Google Contacts File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("JSON files", "*.json"),
                    ("VCF files", "*.vcf"),
                    ("All files", "*.*")
                ],
                initialdir=str(Path.home())
            )
            
            root.destroy()
            
            if file_path:
                self._log(f"Selected input file: {file_path}")
                return file_path
            
            return None
            
        except Exception as e:
            self._log(f"Error selecting input file: {e}", "ERROR")
            return None
    
    def select_output_directory(self) -> Optional[str]:
        """Select output directory for processed files."""
        try:
            root = tk.Tk()
            root.withdraw()
            
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Processed Contacts",
                initialdir=str(self.desktop_path)
            )
            
            root.destroy()
            
            if output_dir:
                self._log(f"Selected output directory: {output_dir}")
                return output_dir
            
            return None
            
        except Exception as e:
            self._log(f"Error selecting output directory: {e}", "ERROR")
            return None
    
    def load_contacts_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load contacts data from various file formats."""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        self._log(f"Loaded CSV with {encoding} encoding: {len(df)} contacts")
                        return df
                    except UnicodeDecodeError:
                        continue
                
                # If all encodings fail
                raise ValueError("Could not decode CSV file with any standard encoding")
            
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert JSON to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Handle different JSON structures
                    if 'contacts' in data:
                        df = pd.DataFrame(data['contacts'])
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise ValueError("Unsupported JSON structure")
                
                self._log(f"Loaded JSON: {len(df)} contacts")
                return df
            
            elif file_ext == '.vcf':
                # Parse VCF file
                contacts = self._parse_vcf_file(file_path)
                df = pd.DataFrame(contacts)
                self._log(f"Loaded VCF: {len(df)} contacts")
                return df
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        
        except Exception as e:
            self._log(f"Error loading contacts data: {e}", "ERROR")
            return None
    
    def _parse_vcf_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse VCF (vCard) file format."""
        contacts = []
        current_contact = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line == 'BEGIN:VCARD':
                        current_contact = {}
                    elif line == 'END:VCARD':
                        if current_contact:
                            contacts.append(current_contact.copy())
                            current_contact = {}
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        
                        # Handle common vCard fields
                        if key.startswith('FN'):
                            current_contact['Name'] = value
                        elif key.startswith('TEL'):
                            phone_key = f"Phone {len([k for k in current_contact.keys() if k.startswith('Phone')]) + 1}"
                            current_contact[phone_key] = value
                        elif key.startswith('EMAIL'):
                            email_key = f"Email {len([k for k in current_contact.keys() if k.startswith('Email')]) + 1}"
                            current_contact[email_key] = value
                        elif key.startswith('ORG'):
                            current_contact['Organization'] = value
                        elif key.startswith('ADR'):
                            current_contact['Address'] = value
        
        except Exception as e:
            self._log(f"Error parsing VCF file: {e}", "ERROR")
        
        return contacts
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistent processing."""
        # Common column name mappings
        column_mappings = {
            # Name fields
            'full name': 'Name',
            'display name': 'Name',
            'contact name': 'Name',
            'first name': 'Given Name',
            'last name': 'Family Name',
            'middle name': 'Additional Name',
            
            # Phone fields
            'phone': 'Phone 1 - Value',
            'mobile': 'Phone 1 - Value',
            'cell': 'Phone 1 - Value',
            'telephone': 'Phone 1 - Value',
            'home phone': 'Phone 2 - Value',
            'work phone': 'Phone 3 - Value',
            
            # Email fields
            'email': 'E-mail 1 - Value',
            'email address': 'E-mail 1 - Value',
            'primary email': 'E-mail 1 - Value',
            
            # Organization fields
            'company': 'Organization 1 - Name',
            'organization': 'Organization 1 - Name',
            'employer': 'Organization 1 - Name',
            
            # Address fields
            'address': 'Address 1 - Formatted',
            'home address': 'Address 1 - Formatted',
            'work address': 'Address 2 - Formatted'
        }
        
        # Apply mappings
        df_renamed = df.copy()
        df_renamed.columns = [column_mappings.get(col.lower(), col) for col in df.columns]
        
        self._log(f"Standardized column names: {list(df_renamed.columns)}")
        return df_renamed
    
    def fix_name_capitalization(self, name: str) -> str:
        """Fix name capitalization using proper case rules."""
        if not name or pd.isna(name):
            return name
        
        # Convert to string and clean
        name = str(name).strip()
        if not name:
            return name
        
        # Check if it's likely a business name
        if self._is_business_name(name):
            return self._fix_business_name_capitalization(name)
        
        # Process as personal name
        try:
            # Use nameparser for intelligent name parsing
            parsed_name = HumanName(name)
            
            # Fix each component
            if parsed_name.first:
                parsed_name.first = self._fix_name_component(parsed_name.first)
            if parsed_name.middle:
                parsed_name.middle = self._fix_name_component(parsed_name.middle)
            if parsed_name.last:
                parsed_name.last = self._fix_name_component(parsed_name.last)
            if parsed_name.suffix:
                parsed_name.suffix = self._fix_name_component(parsed_name.suffix)
            if parsed_name.title:
                parsed_name.title = self._fix_name_component(parsed_name.title)
            
            fixed_name = str(parsed_name)
            
            if fixed_name != name:
                self.stats['names_fixed'] += 1
            
            return fixed_name
            
        except Exception:
            # Fallback to simple title case with prefix handling
            return self._simple_name_fix(name)
    
    def _is_business_name(self, name: str) -> bool:
        """Determine if name is likely a business name."""
        name_lower = name.lower()
        
        # Check for business indicators
        for indicator in self.business_indicators:
            if indicator in name_lower:
                return True
        
        # Check for patterns like "Smith & Associates"
        if ' & ' in name or ' and ' in name_lower:
            return True
        
        # Check for all caps (common in business names)
        if name.isupper() and len(name) > 5:
            return True
        
        return False
    
    def _fix_business_name_capitalization(self, name: str) -> str:
        """Fix capitalization for business names."""
        # For business names, use title case but preserve certain patterns
        words = name.split()
        fixed_words = []
        
        for word in words:
            # Preserve common business abbreviations
            if word.upper() in ['LLC', 'INC', 'CORP', 'LTD', 'CO', 'LP', 'PC', 'PA']:
                fixed_words.append(word.upper())
            elif word.lower() in ['and', 'of', 'the', 'for', 'in', 'on', 'at', 'by']:
                fixed_words.append(word.lower())
            else:
                fixed_words.append(word.capitalize())
        
        return ' '.join(fixed_words)
    
    def _fix_name_component(self, component: str) -> str:
        """Fix capitalization for individual name components."""
        if not component:
            return component
        
        component = component.strip()
        
        # Handle special prefixes
        component_lower = component.lower()
        
        # Scottish/Irish prefixes
        if component_lower.startswith('mc') and len(component) > 2:
            return 'Mc' + component[2:].capitalize()
        elif component_lower.startswith('mac') and len(component) > 3:
            return 'Mac' + component[3:].capitalize()
        elif component_lower.startswith('o\'') and len(component) > 2:
            return 'O\'' + component[2:].capitalize()
        
        # Other prefixes
        for prefix in self.name_prefixes:
            if component_lower.startswith(prefix.lower()) and len(component) > len(prefix):
                return prefix.capitalize() + component[len(prefix):].capitalize()
        
        # Hyphenated names
        if '-' in component:
            parts = component.split('-')
            return '-'.join(part.capitalize() for part in parts)
        
        # Default capitalization
        return component.capitalize()
    
    def _simple_name_fix(self, name: str) -> str:
        """Simple fallback name capitalization."""
        # Split on spaces and capitalize each word
        words = name.split()
        fixed_words = []
        
        for word in words:
            if word.lower() in ['jr', 'sr', 'ii', 'iii', 'iv', 'v']:
                fixed_words.append(word.upper())
            elif word.lower() in ['of', 'the', 'and', 'de', 'la', 'von', 'van']:
                fixed_words.append(word.lower())
            else:
                fixed_words.append(self._fix_name_component(word))
        
        return ' '.join(fixed_words)
    
    def standardize_phone_number(self, phone: str, default_country: str = 'US') -> str:
        """Standardize phone number format."""
        if not phone or pd.isna(phone):
            return phone
        
        phone = str(phone).strip()
        if not phone:
            return phone
        
        try:
            # Parse phone number
            parsed = phonenumbers.parse(phone, default_country)
            
            if phonenumbers.is_valid_number(parsed):
                # Format in international format
                formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                
                if formatted != phone:
                    self.stats['phones_standardized'] += 1
                
                return formatted
            else:
                # Invalid number, return cleaned version
                return self._clean_phone_number(phone)
        
        except phonenumbers.NumberParseException:
            # Could not parse, return cleaned version
            return self._clean_phone_number(phone)
    
    def _clean_phone_number(self, phone: str) -> str:
        """Clean phone number of unwanted characters."""
        # Remove common separators and formatting
        cleaned = re.sub(r'[^\d+\-\(\)\s]', '', phone)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def detect_duplicate_contacts(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """Detect potential duplicate contacts."""
        duplicates = []
        
        name_col = self._find_column(df, ['Name', 'Given Name', 'Full Name'])
        phone_col = self._find_column(df, ['Phone 1 - Value', 'Phone', 'Mobile'])
        email_col = self._find_column(df, ['E-mail 1 - Value', 'Email', 'E-mail'])
        
        if not any([name_col, phone_col, email_col]):
            return duplicates
        
        # Compare each contact with others
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                similarity = self._calculate_contact_similarity(
                    df.iloc[i], df.iloc[j], name_col, phone_col, email_col
                )
                
                if similarity > 0.8:  # High similarity threshold
                    duplicates.append((i, j, similarity))
        
        self.stats['duplicates_found'] = len(duplicates)
        return duplicates
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _calculate_contact_similarity(self, contact1: pd.Series, contact2: pd.Series,
                                    name_col: str, phone_col: str, email_col: str) -> float:
        """Calculate similarity between two contacts."""
        similarity_score = 0.0
        factors = 0
        
        # Name similarity
        if name_col and name_col in contact1 and name_col in contact2:
            name1 = str(contact1[name_col]).lower().strip() if pd.notna(contact1[name_col]) else ""
            name2 = str(contact2[name_col]).lower().strip() if pd.notna(contact2[name_col]) else ""
            
            if name1 and name2:
                name_sim = self._string_similarity(name1, name2)
                similarity_score += name_sim * 0.5  # Name weight: 50%
                factors += 0.5
        
        # Phone similarity
        if phone_col and phone_col in contact1 and phone_col in contact2:
            phone1 = str(contact1[phone_col]).strip() if pd.notna(contact1[phone_col]) else ""
            phone2 = str(contact2[phone_col]).strip() if pd.notna(contact2[phone_col]) else ""
            
            if phone1 and phone2:
                # Extract digits only for comparison
                digits1 = re.sub(r'\D', '', phone1)
                digits2 = re.sub(r'\D', '', phone2)
                
                if digits1 and digits2:
                    if digits1 == digits2:
                        similarity_score += 0.4  # Phone weight: 40%
                    elif digits1[-10:] == digits2[-10:]:  # Last 10 digits match
                        similarity_score += 0.3
                factors += 0.4
        
        # Email similarity
        if email_col and email_col in contact1 and email_col in contact2:
            email1 = str(contact1[email_col]).lower().strip() if pd.notna(contact1[email_col]) else ""
            email2 = str(contact2[email_col]).lower().strip() if pd.notna(contact2[email_col]) else ""
            
            if email1 and email2:
                if email1 == email2:
                    similarity_score += 0.1  # Email weight: 10%
                factors += 0.1
        
        return similarity_score / factors if factors > 0 else 0.0
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple character comparison."""
        if str1 == str2:
            return 1.0
        
        # Simple character-based similarity
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        # Count matching characters in order
        matches = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                matches += 1
        
        return matches / max_len
    
    def calculate_contact_quality_score(self, contact: pd.Series) -> float:
        """Calculate quality score for a contact (0-100)."""
        score = 0
        
        # Name completeness (30 points)
        name_fields = ['Name', 'Given Name', 'Family Name', 'Full Name']
        name_value = None
        for field in name_fields:
            if field in contact and pd.notna(contact[field]) and str(contact[field]).strip():
                name_value = str(contact[field]).strip()
                break
        
        if name_value:
            score += 20  # Has name
            if len(name_value.split()) >= 2:
                score += 10  # Has multiple name parts
        
        # Phone completeness (25 points)
        phone_fields = ['Phone 1 - Value', 'Phone', 'Mobile', 'Phone 2 - Value']
        for field in phone_fields:
            if field in contact and pd.notna(contact[field]) and str(contact[field]).strip():
                score += 12.5  # Up to 2 phone numbers
                break
        
        # Email completeness (25 points)
        email_fields = ['E-mail 1 - Value', 'Email', 'E-mail', 'E-mail 2 - Value']
        for field in email_fields:
            if field in contact and pd.notna(contact[field]) and str(contact[field]).strip():
                email = str(contact[field]).strip()
                if '@' in email and '.' in email:
                    score += 25
                break
        
        # Additional information (20 points)
        additional_fields = [
            'Organization 1 - Name', 'Company', 'Organization',
            'Address 1 - Formatted', 'Address',
            'Notes'
        ]
        
        for field in additional_fields:
            if field in contact and pd.notna(contact[field]) and str(contact[field]).strip():
                score += 5  # Up to 4 additional fields
                if score >= 100:
                    break
        
        return min(score, 100)
    
    def process_contacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all contacts with fixes and improvements."""
        self._log("Starting contact processing")
        
        # Create progress window
        progress_window = self._create_progress_window(len(df), "Processing Contacts")
        
        processed_df = df.copy()
        
        def update_progress(current, total, action):
            if progress_window:
                progress_window.update_progress(current, total, action)
        
        try:
            # Process each contact
            for idx, contact in processed_df.iterrows():
                update_progress(idx + 1, len(processed_df), f"Processing contact {idx + 1}")
                
                self.stats['contacts_processed'] += 1
                
                # Fix name fields
                name_fields = ['Name', 'Given Name', 'Family Name', 'Additional Name']
                for field in name_fields:
                    if field in contact and pd.notna(contact[field]):
                        original = contact[field]
                        fixed = self.fix_name_capitalization(str(original))
                        processed_df.at[idx, field] = fixed
                
                # Standardize phone fields
                phone_fields = [col for col in processed_df.columns if 'Phone' in col and 'Value' in col]
                for field in phone_fields:
                    if field in contact and pd.notna(contact[field]):
                        original = contact[field]
                        standardized = self.standardize_phone_number(str(original))
                        processed_df.at[idx, field] = standardized
                
                # Calculate quality score
                quality_score = self.calculate_contact_quality_score(contact)
                processed_df.at[idx, 'Quality Score'] = quality_score
                
                if quality_score > 70:
                    self.stats['quality_improvements'] += 1
        
        finally:
            if progress_window:
                progress_window.close()
        
        # Sort by quality score (highest first)
        processed_df = processed_df.sort_values('Quality Score', ascending=False)
        
        self._log("Contact processing completed")
        return processed_df
    
    def save_processed_contacts(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """Save processed contacts in multiple formats."""
        output_files = {}
        
        try:
            output_path = Path(output_dir)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as CSV
            csv_file = output_path / f"processed_contacts_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            output_files['csv'] = str(csv_file)
            
            # Save as JSON
            json_file = output_path / f"processed_contacts_{timestamp}.json"
            df.to_json(json_file, orient='records', indent=2)
            output_files['json'] = str(json_file)
            
            # Save high-quality contacts separately
            high_quality = df[df['Quality Score'] >= 80]
            if not high_quality.empty:
                hq_csv_file = output_path / f"high_quality_contacts_{timestamp}.csv"
                high_quality.to_csv(hq_csv_file, index=False, encoding='utf-8-sig')
                output_files['high_quality_csv'] = str(hq_csv_file)
            
            # Save duplicate analysis if duplicates found
            if self.stats['duplicates_found'] > 0:
                duplicates = self.detect_duplicate_contacts(df)
                if duplicates:
                    dup_file = output_path / f"duplicate_analysis_{timestamp}.json"
                    dup_data = {
                        'total_duplicates': len(duplicates),
                        'duplicate_pairs': [
                            {
                                'contact1_index': dup[0],
                                'contact2_index': dup[1],
                                'similarity_score': dup[2],
                                'contact1_name': str(df.iloc[dup[0]].get('Name', 'Unknown')),
                                'contact2_name': str(df.iloc[dup[1]].get('Name', 'Unknown'))
                            }
                            for dup in duplicates
                        ]
                    }
                    
                    with open(dup_file, 'w', encoding='utf-8') as f:
                        json.dump(dup_data, f, indent=2)
                    output_files['duplicates'] = str(dup_file)
            
            self._log(f"Saved processed contacts to: {output_path}")
            return output_files
            
        except Exception as e:
            self._log(f"Error saving processed contacts: {e}", "ERROR")
            return {}
    
    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        processing_time = time.time() - self.start_time
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'platform_info': self.platform_info,
                'log_file': str(self.log_file)
            },
            'statistics': self.stats,
            'processing_summary': {
                'total_contacts': self.stats['contacts_processed'],
                'names_fixed_percentage': (self.stats['names_fixed'] / max(1, self.stats['contacts_processed'])) * 100,
                'phones_standardized_percentage': (self.stats['phones_standardized'] / max(1, self.stats['contacts_processed'])) * 100,
                'quality_improvement_rate': (self.stats['quality_improvements'] / max(1, self.stats['contacts_processed'])) * 100
            }
        }
        
        return report
    
    def save_processing_report(self, report: Dict[str, Any], output_dir: str) -> str:
        """Save processing report to JSON file."""
        try:
            output_path = Path(output_dir)
            report_file = output_path / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self._log(f"Processing report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self._log(f"Error saving report: {e}", "ERROR")
            return ""
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str) -> str:
        """Create visualizations for contact processing results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Google Contacts Processing Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Quality Score Distribution
            ax1 = axes[0, 0]
            if 'Quality Score' in df.columns:
                df['Quality Score'].hist(bins=20, color='skyblue', alpha=0.7, ax=ax1)
                ax1.axvline(df['Quality Score'].mean(), color='red', linestyle='--', 
                           label=f'Average: {df["Quality Score"].mean():.1f}')
                ax1.set_xlabel('Quality Score')
                ax1.set_ylabel('Number of Contacts')
                ax1.set_title('Contact Quality Score Distribution')
                ax1.legend()
            
            # Plot 2: Processing Statistics
            ax2 = axes[0, 1]
            stats_labels = ['Names Fixed', 'Phones Standardized', 'Quality Improved', 'Duplicates Found']
            stats_values = [
                self.stats['names_fixed'],
                self.stats['phones_standardized'],
                self.stats['quality_improvements'],
                self.stats['duplicates_found']
            ]
            
            bars = ax2.bar(stats_labels, stats_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
            ax2.set_title('Processing Statistics')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_values)*0.01,
                        f'{value}', ha='center', va='bottom')
            
            # Plot 3: Quality Categories
            ax3 = axes[1, 0]
            if 'Quality Score' in df.columns:
                quality_categories = pd.cut(df['Quality Score'], 
                                          bins=[0, 40, 70, 90, 100], 
                                          labels=['Poor', 'Fair', 'Good', 'Excellent'])
                category_counts = quality_categories.value_counts()
                
                colors = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60']
                wedges, texts, autotexts = ax3.pie(category_counts.values, labels=category_counts.index, 
                                                  colors=colors, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Contact Quality Categories')
            
            # Plot 4: Field Completeness
            ax4 = axes[1, 1]
            important_fields = ['Name', 'Phone 1 - Value', 'E-mail 1 - Value', 'Organization 1 - Name']
            completeness_data = []
            field_labels = []
            
            for field in important_fields:
                if field in df.columns:
                    completeness = (df[field].notna() & (df[field] != '')).sum() / len(df) * 100
                    completeness_data.append(completeness)
                    field_labels.append(field.replace(' 1 - Value', '').replace(' 1 - Name', ''))
            
            if completeness_data:
                bars = ax4.bar(field_labels, completeness_data, color='lightcoral', alpha=0.7)
                ax4.set_title('Field Completeness')
                ax4.set_ylabel('Completeness (%)')
                ax4.set_ylim(0, 100)
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, completeness_data):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                            f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save visualization
            output_path = Path(output_dir)
            viz_file = output_path / f"contacts_processing_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"Visualization saved: {viz_file}")
            return str(viz_file)
            
        except Exception as e:
            self._log(f"Error creating visualizations: {e}", "ERROR")
            return ""
    
    def _create_progress_window(self, total_items: int, title: str = "Processing"):
        """Create progress tracking window."""
        try:
            root = tk.Tk()
            root.title(title)
            root.geometry("500x200")
            root.resizable(False, False)
            
            # Progress info
            info_label = tk.Label(root, text=f"Processing {total_items} contacts...", font=("Arial", 12))
            info_label.pack(pady=10)
            
            # Progress bar
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=400)
            progress_bar.pack(pady=10)
            
            # Status label
            status_label = tk.Label(root, text="Starting...", font=("Arial", 10))
            status_label.pack(pady=5)
            
            # Action label
            action_label = tk.Label(root, text="", font=("Arial", 9), fg="gray")
            action_label.pack(pady=5)
            
            class ProgressWindow:
                def __init__(self, root, progress_var, status_label, action_label):
                    self.root = root
                    self.progress_var = progress_var
                    self.status_label = status_label
                    self.action_label = action_label
                
                def update_progress(self, current, total, action):
                    progress = (current / total) * 100
                    self.progress_var.set(progress)
                    self.status_label.config(text=f"Processing {current}/{total} contacts")
                    self.action_label.config(text=action)
                    self.root.update()
                
                def close(self):
                    self.root.destroy()
            
            return ProgressWindow(root, progress_var, status_label, action_label)
            
        except Exception as e:
            self._log(f"Error creating progress window: {e}", "ERROR")
            return None
    
    def run_interactive_processing(self):
        """Run interactive contact processing with GUI dialogs."""
        try:
            # Welcome message
            root = tk.Tk()
            root.withdraw()
            
            messagebox.showinfo(
                "Google Contacts Processor",
                "Welcome to Google Contacts Processor v1.0.0\n\n"
                "This tool will help you clean and standardize your Google Contacts:\n"
                "• Fix name capitalization\n"
                "• Standardize phone numbers\n"
                "• Detect duplicates\n"
                "• Calculate quality scores\n\n"
                "Click OK to begin..."
            )
            
            # Select input file
            input_file = self.select_input_file()
            if not input_file:
                messagebox.showwarning("Cancelled", "No input file selected. Exiting.")
                return False
            
            # Select output directory
            output_dir = self.select_output_directory()
            if not output_dir:
                messagebox.showwarning("Cancelled", "No output directory selected. Exiting.")
                return False
            
            # Load contacts data
            self._log("Loading contacts data...")
            df = self.load_contacts_data(input_file)
            if df is None or df.empty:
                messagebox.showerror("Error", "Could not load contacts data or file is empty.")
                return False
            
            # Standardize column names
            df = self.standardize_column_names(df)
            
            # Confirm processing
            confirm = messagebox.askyesno(
                "Confirm Processing",
                f"Loaded {len(df)} contacts from {Path(input_file).name}\n\n"
                f"Detected columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}\n\n"
                f"Output directory: {output_dir}\n\n"
                f"Continue with processing?"
            )
            
            if not confirm:
                messagebox.showinfo("Cancelled", "Processing cancelled by user.")
                return False
            
            # Process contacts
            processed_df = self.process_contacts(df)
            
            # Save processed contacts
            output_files = self.save_processed_contacts(processed_df, output_dir)
            
            # Generate and save report
            report = self.generate_processing_report()
            report_file = self.save_processing_report(report, output_dir)
            
            # Create visualizations
            viz_file = self.create_visualizations(processed_df, output_dir)
            
            # Show completion message
            messagebox.showinfo(
                "Processing Complete",
                f"Google Contacts processing completed successfully!\n\n"
                f"Contacts processed: {self.stats['contacts_processed']}\n"
                f"Names fixed: {self.stats['names_fixed']}\n"
                f"Phones standardized: {self.stats['phones_standardized']}\n"
                f"Quality improvements: {self.stats['quality_improvements']}\n"
                f"Duplicates found: {self.stats['duplicates_found']}\n\n"
                f"Output files: {len(output_files)}\n"
                f"Report: {report_file}\n"
                f"Visualization: {viz_file}\n"
                f"Log file: {self.log_file}"
            )
            
            return True
            
        except Exception as e:
            self._log(f"Error in interactive processing: {e}", "ERROR")
            try:
                messagebox.showerror("Error", f"An error occurred: {e}")
            except:
                pass
            return False

def main():
    """Main function to run Google Contacts processor."""
    print("Starting Google Contacts Processor Suite...")
    
    try:
        # Initialize processor
        processor = GoogleContactsProcessor()
        
        # Run interactive processing
        success = processor.run_interactive_processing()
        
        if success:
            # Show macOS notification if available
            try:
                if platform.system() == "Darwin":
                    subprocess.run([
                        "osascript", "-e",
                        f'display notification "Google Contacts processing completed!" with title "GET SWIFTY - Contacts Processor"'
                    ], check=False)
            except:
                pass
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()