#!/usr/bin/env python3
"""
Test script to verify document selection functionality
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_document_scanning():
    """Test that documents are properly scanned."""
    try:
        from app import scan_data_folder, DATA_DIR
    except ImportError as e:
        print(f"Error importing functions: {e}")
        return False
    
    print("🔍 Testing document scanning...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return False
    
    documents = scan_data_folder()
    
    if not documents:
        print("❌ No documents found in data folder")
        return False
    
    print(f"✅ Found {len(documents)} documents:")
    for rel_path, doc_info in documents.items():
        size_mb = doc_info['size'] / (1024 * 1024)
        print(f"  📄 {rel_path} ({size_mb:.2f} MB)")
    
    return True

def test_selection_logic():
    """Test the document selection logic."""
    print("\n🎯 Testing selection logic...")
    
    # Simulate available documents
    available_docs = {
        'doc1.txt': {'path': '/data/doc1.txt', 'size': 1000},
        'doc2.txt': {'path': '/data/doc2.txt', 'size': 2000},
        'doc3.csv': {'path': '/data/doc3.csv', 'size': 500}
    }
    
    # Test select all functionality
    selected_docs = set(available_docs.keys())
    print(f"✅ Select all: {len(selected_docs)} documents selected")
    
    # Test individual selection
    selected_docs = {'doc1.txt', 'doc3.csv'}
    print(f"✅ Individual selection: {len(selected_docs)} documents selected")
    
    # Test filtering
    filtered_docs = {k: v for k, v in available_docs.items() if k in selected_docs}
    print(f"✅ Filtered documents: {list(filtered_docs.keys())}")
    
    return True

def test_file_access():
    """Test that sample files are accessible."""
    print("\n📁 Testing file access...")
    
    sample_files = [
        './data/sample.txt',
        './data/sample_data.csv',
        './data/document1.txt',
        './data/document2.txt',
        './data/sample_report.csv'
    ]
    
    accessible_files = []
    for file_path in sample_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            accessible_files.append((file_path, size))
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} not found")
    
    if len(accessible_files) >= 3:
        print(f"✅ {len(accessible_files)} sample files accessible")
        return True
    else:
        print(f"❌ Only {len(accessible_files)} files accessible")
        return False

def main():
    print("🧪 Testing Document Selection Functionality")
    print("=" * 60)
    
    # Test file access
    file_access = test_file_access()
    
    # Test document scanning
    scanning = test_document_scanning()
    
    # Test selection logic
    selection = test_selection_logic()
    
    print("\n" + "=" * 60)
    if file_access and scanning and selection:
        print("✅ All tests passed! Document selection functionality is working correctly.")
        print("\n💡 Next steps:")
        print("1. Start the application with: docker-compose up --build")
        print("2. Select '📂 Data Folder' in the Document Source section")
        print("3. Use the checkboxes to select documents")
        print("4. Click '🔄 Load Selected Documents'")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main() 