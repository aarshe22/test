# Mistral 7B Chat with Document Upload

A powerful AI chat application built with Streamlit that allows you to interact with the Mistral-7B-Instruct-v0.2 model while analyzing uploaded documents.

## ✨ New Features (Latest Update)

### 📂 Data Folder Integration
- **Automatic Document Loading**: Place documents in `./data` folder for automatic processing
- **Document Selection**: Choose specific documents with checkboxes and "Select All" option
- **Dual Document Sources**: Choose between manual upload or data folder documents
- **Recursive Scanning**: Automatically finds documents in subdirectories
- **Batch Processing**: Load selected documents with one click

### 🧠 Conversation Memory
- **Persistent Conversations**: Your chat history is maintained throughout the session
- **Configurable Context Window**: Adjust how many recent messages to include (5-20 messages)
- **Smart Context Management**: Automatically trims old messages to maintain performance
- **Conversation Summary**: Get a quick overview of your conversation

### 🌊 Real-time Response Streaming
- **Token-by-Token Display**: Watch responses generate in real-time
- **Smooth User Experience**: No more waiting for complete responses
- **Visual Feedback**: Immediate response generation feedback

### 🎛️ Enhanced Controls
- **Model Parameters**: Adjust temperature (0.1-1.5) and max tokens (100-1000)
- **Conversation Management**: Clear conversations, export to JSON
- **System Monitoring**: Real-time GPU, CPU, and RAM usage tracking

## 🚀 Features

### Core Functionality
- **AI Chat Interface**: Interactive conversations with Mistral 7B model
- **Document Upload**: Support for PDF, DOC, DOCX, CSV, and TXT files
- **Data Folder Integration**: Automatic processing of documents in `./data` folder
- **Document Selection**: Choose specific documents to process with checkboxes
- **Multi-document Analysis**: Upload and analyze multiple documents simultaneously
- **Context-Aware Responses**: AI considers both conversation history and document content

### Document Processing
- **Text Extraction**: Automatic text extraction from various file formats
- **Format Preservation**: Maintains document structure where possible
- **Batch Processing**: Handle multiple files efficiently
- **Dual Source Support**: Manual upload or automatic data folder scanning

### User Interface
- **Responsive Design**: Clean, modern interface with sidebar controls
- **Document Source Selection**: Choose between upload and data folder modes
- **Real-time Stats**: Monitor system performance during conversations
- **Export Capabilities**: Save conversations for later reference
- **Tabbed Views**: Organized conversation display with summary tabs

## 🛠️ Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- Hugging Face account with access to Mistral models

### Quick Start
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stack
   ```

2. **Add your Hugging Face token**
   ```bash
   echo "your_hf_token_here" > hf-token
   ```

3. **Set up the data folder** (optional but recommended)
   ```bash
   ./setup_data_folder.sh
   ```

4. **Download the model** (first time only)
   ```bash
   python download_model.py
   ```

5. **Start the application**
   ```bash
   docker-compose up --build
   ```

6. **Access the application**
   Open your browser and go to `http://localhost:8501`

## 📋 Usage

### Document Sources
The application supports two ways to provide documents for analysis:

#### 📤 Manual Upload
1. Select "📤 Upload Files" in the Document Source section
2. Use the file uploader to select PDF, DOC, DOCX, CSV, or TXT files
3. Files are automatically processed and text is extracted

#### 📂 Data Folder
1. Place your documents in the `./data` folder in the project root
2. Select "📂 Data Folder" in the Document Source section
3. Use the document selection checkboxes to choose which files to process
4. Click "🔄 Load Selected Documents" to process the selected files
5. The system will scan recursively and extract text from all selected files

### Starting a Conversation
1. Choose your document source (upload or data folder)
2. Load your documents using the appropriate method
3. Type your question in the chat input
4. Click "Generate Response" to get an AI response
5. Continue the conversation - the AI remembers previous context

### Managing Conversations
- **Adjust Context Window**: Use the slider in the sidebar to control memory
- **Clear Conversation**: Click "Clear Conversation" to start fresh
- **Export Conversation**: Download your conversation as JSON
- **View Summary**: Check the conversation summary in the sidebar

### Model Parameters
- **Temperature**: Controls creativity (higher = more creative, lower = more focused)
- **Max Tokens**: Sets maximum response length
- **Context Window**: Number of recent messages to include

## 🔧 Configuration

### Environment Variables
- `HF_TOKEN`: Your Hugging Face token (stored in `hf-token` file)
- `MODEL_DIR`: Local model directory (default: `./models/mistral-7b`)

### Docker Configuration
- **GPU Support**: Configured for NVIDIA GPUs with CUDA 11.8
- **Port**: Application runs on port 8501
- **Volumes**: 
  - App directory mounted for development
  - `./data` folder mounted to `/data` in container for document access

## 📊 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for model

### Software
- **OS**: Linux (tested on Ubuntu 22.04)
- **Docker**: Version 20.10+
- **NVIDIA Drivers**: Compatible with CUDA 11.8

## 🎯 Use Cases

### Document Analysis
- **Research Papers**: Ask questions about academic papers
- **Reports**: Analyze business reports and extract insights
- **Manuals**: Get help understanding technical documentation
- **Data Analysis**: Query CSV files and get insights

### Conversational AI
- **General Chat**: Have natural conversations with context
- **Follow-up Questions**: Ask follow-ups based on previous responses
- **Multi-turn Discussions**: Engage in complex multi-step conversations

## 🧪 Testing

### Data Folder Functionality
To test the data folder functionality:

1. **Run the setup script** (if not already done):
   ```bash
   ./setup_data_folder.sh
   ```

2. **Test the scanning functionality**:
   ```bash
   python test_data_folder.py
   ```

3. **Start the application and test**:
   - Start with `docker-compose up --build`
   - Select "📂 Data Folder" in the Document Source section
   - Click "🔄 Load Documents from Data Folder"
   - Verify that the sample documents are loaded

### Troubleshooting Data Folder Issues
- **No documents found**: Ensure files have supported extensions (.pdf, .doc, .docx, .csv, .txt)
- **Permission errors**: Check that the ./data folder has proper read permissions
- **Mount issues**: Verify the volume mount in docker-compose.yml is correct
- **Path issues**: The container expects documents at `/data`, mapped from `./data`

## 🔍 Troubleshooting

### Common Issues
1. **Model Download Fails**: Check your Hugging Face token and internet connection
2. **GPU Memory Issues**: Use the memory optimization script and adjust settings
3. **Slow Responses**: Check GPU utilization and system resources
4. **Docker Issues**: Use `reset-docker.sh` to clean up containers

### Performance Tips
- **Optimize Context Window**: Use smaller values for faster responses
- **Monitor Resources**: Watch system stats in the sidebar
- **Clear Conversations**: Regularly clear old conversations to free memory
- **Use Memory Management**: Adjust chunking and batch settings based on your GPU

## 📈 Future Enhancements

### Planned Features
- [ ] Advanced document chunking with overlap
- [ ] Semantic search for document sections
- [ ] Multi-modal support (images, audio)
- [ ] User authentication and session management
- [ ] Database storage for persistent conversations
- [ ] API endpoints for integration

### Contributing
Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the excellent Mistral-7B model
- **Hugging Face** for the transformers library
- **Streamlit** for the amazing web framework
- **NVIDIA** for CUDA support and GPU acceleration 

## 🧠 Memory Management

### CUDA Memory Issues
The application includes comprehensive memory management to prevent CUDA out of memory errors:

#### 🔧 Built-in Memory Management
- **Document Chunking**: Large documents are automatically split into smaller chunks
- **Batch Processing**: Limits the number of documents processed simultaneously
- **Character Limits**: Prevents processing documents that exceed memory capacity
- **Memory Monitoring**: Real-time GPU memory usage display
- **Automatic Cleanup**: Periodic GPU memory clearing during processing

#### ⚙️ Memory Settings
Adjust these settings in the sidebar under "💾 Memory Management":
- **Max Chunk Size**: Characters per document chunk (1000-8000)
- **Max Documents per Batch**: Documents processed at once (1-10)
- **Max Total Characters**: Total characters across all documents (500K-2M)
- **Clear GPU Memory**: Manual memory cleanup button

#### 🚀 Memory Optimization Script
Run the memory optimization script to diagnose and fix memory issues:
```bash
python optimize_memory.py
```

This script will:
- Check system resources (CPU, RAM, GPU)
- Analyze PyTorch memory usage
- Provide GPU-specific recommendations
- Clear memory if requested

#### 💡 Memory Optimization Tips

**For GPUs with < 8GB VRAM:**
- Set Max Chunk Size to 2000-3000 characters
- Set Max Documents per Batch to 2-3
- Set Max Total Characters to 500,000
- Use smaller documents when possible

**For GPUs with 8-12GB VRAM:**
- Set Max Chunk Size to 3000-4000 characters
- Set Max Documents per Batch to 3-5
- Set Max Total Characters to 800,000

**For GPUs with > 12GB VRAM:**
- Set Max Chunk Size to 4000-6000 characters
- Set Max Documents per Batch to 5-8
- Set Max Total Characters to 1,000,000+

#### 🔄 General Memory Management
- Process documents in smaller batches
- Use the "Clear GPU Memory" button regularly
- Restart the application if memory usage gets too high
- Consider using the data folder feature instead of uploading large files
- Monitor memory usage in the sidebar

## 🧪 Testing

### Data Folder Functionality
To test the data folder functionality:

1. **Run the setup script** (if not already done):
   ```bash
   ./setup_data_folder.sh
   ```

2. **Test the scanning functionality**:
   ```bash
   python test_data_folder.py
   ```

3. **Start the application and test**:
   - Start with `docker-compose up --build`
   - Select "📂 Data Folder" in the Document Source section
   - Click "🔄 Load Documents from Data Folder"
   - Verify that the sample documents are loaded

### Troubleshooting Data Folder Issues
- **No documents found**: Ensure files have supported extensions (.pdf, .doc, .docx, .csv, .txt)
- **Permission errors**: Check that the ./data folder has proper read permissions
- **Mount issues**: Verify the volume mount in docker-compose.yml is correct
- **Path issues**: The container expects documents at `/data`, mapped from `./data`

## 🔍 Troubleshooting

### Common Issues
1. **Model Download Fails**: Check your Hugging Face token and internet connection
2. **GPU Memory Issues**: Use the memory optimization script and adjust settings
3. **Slow Responses**: Check GPU utilization and system resources
4. **Docker Issues**: Use `reset-docker.sh` to clean up containers

### Performance Tips
- **Optimize Context Window**: Use smaller values for faster responses
- **Monitor Resources**: Watch system stats in the sidebar
- **Clear Conversations**: Regularly clear old conversations to free memory
- **Use Memory Management**: Adjust chunking and batch settings based on your GPU

## 📈 Future Enhancements

### Planned Features
- [ ] Advanced document chunking with overlap
- [ ] Semantic search for document sections
- [ ] Multi-modal support (images, audio)
- [ ] User authentication and session management
- [ ] Database storage for persistent conversations
- [ ] API endpoints for integration

### Contributing
Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the excellent Mistral-7B model
- **Hugging Face** for the transformers library
- **Streamlit** for the amazing web framework
- **NVIDIA** for CUDA support and GPU acceleration 