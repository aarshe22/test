# Memory Management Guide

## 🚨 CUDA Out of Memory Error Solutions

This guide addresses the `torch.OutOfMemoryError: CUDA out of memory` error that can occur when processing large documents with the Mistral 7B model.

## 🔧 Implemented Solutions

### 1. Document Chunking
- **Automatic Chunking**: Large documents are automatically split into smaller chunks
- **Configurable Chunk Size**: Adjustable from 1000-8000 characters per chunk
- **Smart Splitting**: Chunks are created at sentence/paragraph boundaries when possible

### 2. Batch Processing
- **Document Limits**: Maximum 5 documents processed simultaneously (configurable)
- **Size-based Sorting**: Smaller documents processed first
- **Progressive Loading**: Documents loaded in batches to prevent memory spikes

### 3. Memory Limits
- **Character Limits**: Maximum 1,000,000 total characters across all documents
- **Token Estimation**: Rough token count estimation for memory planning
- **Truncation**: Automatic text truncation when limits are exceeded

### 4. Memory Monitoring
- **Real-time Display**: GPU memory usage shown in sidebar
- **Memory Metrics**: Allocated and reserved memory tracking
- **Progress Indicators**: Memory usage during document processing

### 5. Automatic Cleanup
- **Periodic Clearing**: GPU memory cleared every 2 documents
- **Garbage Collection**: Python garbage collection after processing
- **Manual Cleanup**: "Clear GPU Memory" button in sidebar

### 6. Environment Configuration
- **PyTorch Settings**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Docker Limits**: 16GB memory limit in container
- **Memory Allocation**: Optimized PyTorch memory allocation

## 🛠️ Tools and Scripts

### Memory Optimization Script
```bash
python optimize_memory.py
```

**Features:**
- System resource analysis (CPU, RAM, GPU)
- PyTorch memory diagnostics
- GPU-specific recommendations
- Memory cleanup functionality

### Memory Settings in UI
Located in sidebar under "💾 Memory Management":
- **Max Chunk Size**: 1000-8000 characters
- **Max Documents per Batch**: 1-10 documents
- **Max Total Characters**: 500K-2M characters
- **Clear GPU Memory**: Manual cleanup button

## 💡 Optimization Recommendations

### For GPUs with < 8GB VRAM:
- Max Chunk Size: 2000-3000 characters
- Max Documents per Batch: 2-3
- Max Total Characters: 500,000
- Use smaller documents when possible

### For GPUs with 8-12GB VRAM:
- Max Chunk Size: 3000-4000 characters
- Max Documents per Batch: 3-5
- Max Total Characters: 800,000

### For GPUs with > 12GB VRAM:
- Max Chunk Size: 4000-6000 characters
- Max Documents per Batch: 5-8
- Max Total Characters: 1,000,000+

## 🔄 Best Practices

1. **Start Small**: Begin with smaller documents and increase gradually
2. **Monitor Memory**: Watch GPU memory usage in the sidebar
3. **Clear Regularly**: Use the "Clear GPU Memory" button frequently
4. **Batch Processing**: Process documents in smaller batches
5. **Restart if Needed**: Restart the application if memory gets too high
6. **Use Data Folder**: Consider using the data folder feature for better control

## 🚀 Quick Fixes

### Immediate Actions:
1. Click "Clear GPU Memory" in the sidebar
2. Reduce "Max Documents per Batch" to 2-3
3. Reduce "Max Chunk Size" to 2000-3000
4. Process fewer documents at once

### If Still Having Issues:
1. Run `python optimize_memory.py`
2. Restart the application
3. Check for other GPU-intensive applications
4. Consider upgrading GPU or reducing document sizes

## 📊 Memory Usage Monitoring

The application provides real-time memory monitoring:
- **GPU Memory Used**: Currently allocated memory
- **GPU Memory Reserved**: Memory reserved by PyTorch
- **System Stats**: CPU, RAM, and GPU utilization
- **Processing Progress**: Memory usage during document processing

## 🔍 Troubleshooting

### Common Issues:
- **High Memory Usage**: Reduce chunk size and batch size
- **Slow Processing**: Clear memory and restart
- **Document Errors**: Check file formats and sizes
- **System Crashes**: Reduce memory limits and restart

### Error Messages:
- `CUDA out of memory`: Reduce document size or batch size
- `Memory fragmentation`: Use the memory optimization script
- `Slow performance`: Clear memory and reduce settings

## 📈 Performance Impact

### Memory Management Overhead:
- **Chunking**: Minimal overhead, improves stability
- **Batch Processing**: Slight delay, prevents crashes
- **Memory Monitoring**: Negligible impact
- **Cleanup**: Brief pauses, significant memory savings

### Benefits:
- **Stability**: Prevents crashes and errors
- **Scalability**: Handles larger document sets
- **Reliability**: Consistent performance across different GPUs
- **User Control**: Adjustable settings for different hardware

## 🔮 Future Improvements

### Planned Enhancements:
- Advanced chunking with overlap
- Semantic document chunking
- Dynamic memory allocation
- Memory usage prediction
- Automatic optimization based on hardware 