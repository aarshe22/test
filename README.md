# Mistral 7B Chat with Document Upload

A powerful AI chat application built with Streamlit that allows you to interact with the Mistral-7B-Instruct-v0.2 model while analyzing uploaded documents.

## ✨ New Features (Latest Update)

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
- **Multi-document Analysis**: Upload and analyze multiple documents simultaneously
- **Context-Aware Responses**: AI considers both conversation history and document content

### Document Processing
- **Text Extraction**: Automatic text extraction from various file formats
- **Format Preservation**: Maintains document structure where possible
- **Batch Processing**: Handle multiple files efficiently

### User Interface
- **Responsive Design**: Clean, modern interface with sidebar controls
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

3. **Download the model** (first time only)
   ```bash
   python download_model.py
   ```

4. **Start the application**
   ```bash
   docker-compose up --build
   ```

5. **Access the application**
   Open your browser and go to `http://localhost:8501`

## 📋 Usage

### Starting a Conversation
1. Upload documents (optional) using the file uploader
2. Type your question in the chat input
3. Click "Generate Response" to get an AI response
4. Continue the conversation - the AI remembers previous context

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
- **Volumes**: App directory mounted for development

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

## 🔍 Troubleshooting

### Common Issues
1. **Model Download Fails**: Check your Hugging Face token and internet connection
2. **GPU Memory Issues**: Reduce max tokens or context window
3. **Slow Responses**: Check GPU utilization and system resources
4. **Docker Issues**: Use `reset-docker.sh` to clean up containers

### Performance Tips
- **Optimize Context Window**: Use smaller values for faster responses
- **Monitor Resources**: Watch system stats in the sidebar
- **Clear Conversations**: Regularly clear old conversations to free memory

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