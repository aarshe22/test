#!/bin/bash

# Setup script for data folder functionality

echo "📁 Setting up data folder for Mistral 7B Chat..."

# Create data directory if it doesn't exist
if [ ! -d "./data" ]; then
    echo "Creating ./data directory..."
    mkdir -p ./data
    echo "✅ Data directory created"
else
    echo "✅ Data directory already exists"
fi

# Check if sample files exist
if [ ! -f "./data/sample.txt" ]; then
    echo "Creating sample.txt..."
    cat > ./data/sample.txt << 'EOF'
This is a sample document for testing the data folder functionality.

The Mistral 7B model is a powerful language model that can process and understand various types of documents including:
- PDF files
- Word documents (DOC, DOCX)
- CSV spreadsheets
- Plain text files

This feature allows users to either upload documents manually or use documents that are pre-placed in the ./data folder, which gets mounted into the container at /data.

The system will automatically scan the data folder for supported document types and extract text from them for use in conversations with the AI assistant.
EOF
    echo "✅ Sample text file created"
else
    echo "✅ Sample text file already exists"
fi

if [ ! -f "./data/sample_data.csv" ]; then
    echo "Creating sample_data.csv..."
    cat > ./data/sample_data.csv << 'EOF'
Name,Age,Department,Salary
John Doe,30,Engineering,75000
Jane Smith,28,Marketing,65000
Bob Johnson,35,Finance,80000
Alice Brown,32,HR,60000
Charlie Wilson,29,Engineering,70000
EOF
    echo "✅ Sample CSV file created"
else
    echo "✅ Sample CSV file already exists"
fi

echo ""
echo "🎉 Data folder setup complete!"
echo ""
echo "📋 What you can do now:"
echo "1. Add your own documents to the ./data folder"
echo "2. Supported formats: PDF, DOC, DOCX, CSV, TXT"
echo "3. Start the application with: docker-compose up --build"
echo "4. Select '📂 Data Folder' in the Document Source section"
echo "5. Click '🔄 Load Documents from Data Folder'"
echo ""
echo "📁 Current data folder contents:"
ls -la ./data/ 