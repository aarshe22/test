#!/usr/bin/env python3
"""
Test script to verify data folder functionality
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_folder_scanning():
    """Test the data folder scanning functionality"""
    
    # Import the functions from app.py
    try:
        from app import scan_data_folder, DATA_DIR
    except ImportError as e:
        print(f"Error importing functions: {e}")
        return False
    
    print(f"Testing data folder scanning...")
    print(f"Data directory path: {DATA_DIR}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Make sure the ./data folder exists and is mounted correctly")
        return False
    
    print(f"✅ Data directory found: {DATA_DIR}")
    
    # Scan for documents
    documents = scan_data_folder()
    
    if not documents:
        print("❌ No documents found in data folder")
        print("Supported formats: .pdf, .doc, .docx, .csv, .txt")
        return False
    
    print(f"✅ Found {len(documents)} documents:")
    for rel_path, doc_info in documents.items():
        size_mb = doc_info['size'] / (1024 * 1024)
        print(f"  📄 {rel_path} ({size_mb:.2f} MB)")
    
    return True

def test_file_extraction():
    """Test text extraction from files"""
    
    try:
        from app import extract_text_from_file_path
    except ImportError as e:
        print(f"Error importing extraction function: {e}")
        return False
    
    # Test with a sample text file
    test_file = "./data/sample.txt"
    if os.path.exists(test_file):
        print(f"\nTesting text extraction from {test_file}...")
        try:
            text = extract_text_from_file_path(test_file)
            print(f"✅ Successfully extracted {len(text)} characters")
            print(f"Preview: {text[:100]}...")
            return True
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return False
    else:
        print(f"❌ Test file not found: {test_file}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Data Folder Functionality")
    print("=" * 50)
    
    # Test scanning
    scan_success = test_data_folder_scanning()
    
    # Test extraction
    extract_success = test_file_extraction()
    
    print("\n" + "=" * 50)
    if scan_success and extract_success:
        print("✅ All tests passed! Data folder functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1) 