import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unstructured.partition.auto import partition
import tempfile
import pandas as pd
import PyPDF2
import docx
import GPUtil
import psutil
import time
import json
from typing import List, Dict, Any

MODEL_DIR = "./models/mistral-7b"

st.set_page_config(page_title="Mistral 7B Chat with Document Upload", layout="wide")

# Initialize session state for conversation memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "document_context" not in st.session_state:
    st.session_state.document_context = ""
if "context_window" not in st.session_state:
    st.session_state.context_window = 10  # Number of recent messages to keep

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
    )
    return tokenizer, model

def extract_text_from_file(uploaded_file):
    suffix = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == "pdf":
        elements = partition(tmp_path)
        text = "\n".join(str(el) for el in elements)
    elif suffix in ["doc", "docx"]:
        doc = docx.Document(tmp_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif suffix == "csv":
        df = pd.read_csv(tmp_path)
        text = df.to_string()
    else:
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()

    os.unlink(tmp_path)
    return text

def build_conversation_prompt(user_input: str, conversation_history: List[Dict], document_context: str = "") -> str:
    """Build a prompt that includes conversation history and document context."""
    
    # Start with system instruction
    prompt = "You are a helpful AI assistant. Answer questions based on the provided documents and conversation context.\n\n"
    
    # Add document context if available
    if document_context:
        prompt += f"### DOCUMENTS ###\n{document_context}\n\n"
    
    # Add conversation history (limited by context window)
    if conversation_history:
        prompt += "### CONVERSATION HISTORY ###\n"
        for msg in conversation_history[-st.session_state.context_window:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"
    
    # Add current user input
    prompt += f"### CURRENT QUESTION ###\n{user_input}\n\n### ANSWER ###\n"
    
    return prompt

def stream_generate(prompt: str, tokenizer, model, max_new_tokens=500, temperature=0.7):
    """Generate response with real-time token streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create a placeholder for streaming
    response_placeholder = st.empty()
    full_response = ""
    
    with torch.no_grad():
        # Generate tokens one by one for streaming effect
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        
        # Stream the response token by token
        for i in range(len(inputs.input_ids[0]), len(generated_ids.sequences[0])):
            # Decode up to current token
            current_ids = generated_ids.sequences[0][:i+1]
            current_text = tokenizer.decode(current_ids, skip_special_tokens=True)
            
            # Extract only the new response part
            if "### ANSWER ###" in current_text:
                response = current_text.split("### ANSWER ###")[-1].strip()
            else:
                response = current_text.strip()
            
            # Update the placeholder with current response
            response_placeholder.markdown(f"**🤖 Assistant:** {response}")
            
            # Small delay for streaming effect
            time.sleep(0.01)
            
            full_response = response
    
    return full_response

def add_message_to_history(role: str, content: str):
    """Add a message to the conversation history."""
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })
    
    # Trim history if it exceeds context window
    if len(st.session_state.conversation_history) > st.session_state.context_window * 2:
        st.session_state.conversation_history = st.session_state.conversation_history[-st.session_state.context_window:]

def clear_conversation():
    """Clear the conversation history."""
    st.session_state.conversation_history = []
    st.rerun()

def export_conversation():
    """Export conversation history to JSON."""
    if st.session_state.conversation_history:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        
        export_data = {
            "timestamp": timestamp,
            "document_context": st.session_state.document_context,
            "conversation": st.session_state.conversation_history
        }
        
        return json.dumps(export_data, indent=2), filename
    return None, None

def get_system_stats():
    gpu_load = gpu_mem_used = gpu_mem_total = 0
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = round(gpu.load * 100, 1)
        gpu_mem_used = round(gpu.memoryUsed, 1)
        gpu_mem_total = round(gpu.memoryTotal, 1)

    cpu_load = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used = round(ram.used / (1024 ** 3), 2)
    ram_total = round(ram.total / (1024 ** 3), 2)

    return gpu_load, gpu_mem_used, gpu_mem_total, cpu_load, ram_used, ram_total

def get_conversation_summary(conversation_history: List[Dict]) -> str:
    """Generate a brief summary of the conversation."""
    if not conversation_history:
        return "No conversation yet."
    
    user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
    assistant_messages = [msg["content"] for msg in conversation_history if msg["role"] == "assistant"]
    
    summary = f"Conversation has {len(conversation_history)} messages:\n"
    summary += f"• {len(user_messages)} user questions\n"
    summary += f"• {len(assistant_messages)} assistant responses\n"
    
    if user_messages:
        summary += f"• Last question: {user_messages[-1][:100]}{'...' if len(user_messages[-1]) > 100 else ''}"
    
    return summary

def main():
    st.title("💡 Mistral 7B Chat with Document Upload")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Context window control
        context_window = st.slider(
            "Conversation Memory (messages)", 
            min_value=5, 
            max_value=20, 
            value=st.session_state.context_window,
            help="Number of recent messages to include in context"
        )
        st.session_state.context_window = context_window
        
        # Model parameters
        st.header("🎛️ Model Parameters")
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=1.5, 
            value=0.7, 
            step=0.1,
            help="Higher values make output more creative, lower values more focused"
        )
        max_tokens = st.slider(
            "Max Tokens", 
            min_value=100, 
            max_value=1000, 
            value=500, 
            step=50,
            help="Maximum number of tokens to generate"
        )
        
        # Conversation controls
        st.header("💬 Conversation")
        if st.button("🗑️ Clear Conversation"):
            clear_conversation()
        
        # Conversation summary
        if st.session_state.conversation_history:
            with st.expander("📋 Conversation Summary"):
                st.text(get_conversation_summary(st.session_state.conversation_history))
        
        # Export conversation
        export_data, filename = export_conversation()
        if export_data:
            st.download_button(
                label="📥 Export Conversation",
                data=export_data,
                file_name=filename,
                mime="application/json"
            )
        
        # System stats
        st.header("📊 System Stats")
        gpu_load, gpu_mem_used, gpu_mem_total, cpu_load, ram_used, ram_total = get_system_stats()
        st.metric("GPU Utilization", f"{gpu_load}%")
        st.metric("GPU VRAM", f"{gpu_mem_used}/{gpu_mem_total} GB")
        st.metric("CPU Utilization", f"{cpu_load}%")
        st.metric("RAM Usage", f"{ram_used}/{ram_total} GB")

    tokenizer, model = load_model_and_tokenizer()

    # Document upload
    uploaded_files = st.file_uploader("Upload PDF, DOC, DOCX, CSV or TXT files", type=["pdf", "doc", "docx", "csv", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"Extracting text from {len(uploaded_files)} files...")
        progress_bar = st.progress(0)
        combined_text = ""
        
        for idx, file in enumerate(uploaded_files):
            text = extract_text_from_file(file)
            combined_text += f"\n--- Start of {file.name} ---\n{text}\n--- End of {file.name} ---\n"
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        progress_bar.empty()
        st.success("Text extraction complete.")
        
        # Update document context in session state
        st.session_state.document_context = combined_text
        
        # Display document text in expander
        with st.expander("📄 View Document Text"):
            st.text_area("Combined Document Text", combined_text, height=300, key="document_text_display")

    # Chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history in a more compact format
    if st.session_state.conversation_history:
        st.subheader(f"📝 Conversation History ({len(st.session_state.conversation_history)} messages)")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["💬 Messages", "📊 Summary"])
        
        with tab1:
            for i, msg in enumerate(st.session_state.conversation_history):
                if msg["role"] == "user":
                    st.markdown(f"**👤 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {msg['content']}")
                st.divider()
        
        with tab2:
            st.text(get_conversation_summary(st.session_state.conversation_history))

    # User input
    user_input = st.text_input("💬 Enter your question or prompt:", key="user_input")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        generate_button = st.button("🚀 Generate Response")
    with col2:
        if st.session_state.conversation_history:
            st.caption(f"📊 Using last {len(st.session_state.conversation_history)} messages as context")
    with col3:
        if st.session_state.document_context:
            st.caption("📄 Document context available")

    if generate_button and user_input:
        # Add user message to history
        add_message_to_history("user", user_input)
        
        # Build prompt with conversation history
        full_prompt = build_conversation_prompt(
            user_input, 
            st.session_state.conversation_history, 
            st.session_state.document_context
        )
        
        with st.spinner("Generating response..."):
            # Pass model parameters to the generation function
            response = stream_generate(full_prompt, tokenizer, model, max_new_tokens=max_tokens, temperature=temperature)
            
            # Add assistant response to history
            add_message_to_history("assistant", response)
            
            # Rerun to update the conversation display
            st.rerun()

if __name__ == "__main__":
    main()

