#!/usr/bin/env python3
"""
Memory optimization script for Mistral 7B Chat
Helps diagnose and fix CUDA memory issues
"""

import os
import sys
import torch
import gc
import psutil
import GPUtil

def check_system_resources():
    """Check system resources and provide recommendations."""
    print("🔍 System Resource Check")
    print("=" * 50)
    
    # CPU and RAM
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {ram_used_gb:.2f}GB / {ram_total_gb:.2f}GB ({ram.percent}%)")
    
    if ram.percent > 80:
        print("⚠️  High RAM usage detected. Consider closing other applications.")
    
    # GPU
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_load = gpu.load * 100
            gpu_mem_used = gpu.memoryUsed
            gpu_mem_total = gpu.memoryTotal
            gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
            
            print(f"GPU Load: {gpu_load:.1f}%")
            print(f"GPU Memory: {gpu_mem_used:.1f}GB / {gpu_mem_total:.1f}GB ({gpu_mem_percent:.1f}%)")
            
            if gpu_mem_percent > 80:
                print("⚠️  High GPU memory usage detected.")
                print("   Consider reducing document size or number of documents.")
            
            if gpu_mem_total < 8:
                print("⚠️  GPU has less than 8GB VRAM. Large documents may cause issues.")
        else:
            print("❌ No GPU detected")
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")

def check_pytorch_memory():
    """Check PyTorch memory usage."""
    print("\n🧠 PyTorch Memory Check")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Clear cache first
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Get memory info
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"Current Allocated: {allocated:.2f}GB")
    print(f"Current Reserved: {reserved:.2f}GB")
    print(f"Peak Allocated: {max_allocated:.2f}GB")
    
    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()
    
    if allocated > 0:
        print("⚠️  PyTorch memory is not empty. Consider restarting the application.")

def optimize_memory_settings():
    """Provide memory optimization recommendations."""
    print("\n⚙️  Memory Optimization Recommendations")
    print("=" * 50)
    
    # Check GPU memory
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_mem_total = gpus[0].memoryTotal
            
            if gpu_mem_total < 8:
                print("🔧 For GPUs with < 8GB VRAM:")
                print("   - Set Max Chunk Size to 2000-3000 characters")
                print("   - Set Max Documents per Batch to 2-3")
                print("   - Set Max Total Characters to 500,000")
                print("   - Use smaller documents when possible")
            
            elif gpu_mem_total < 12:
                print("🔧 For GPUs with 8-12GB VRAM:")
                print("   - Set Max Chunk Size to 3000-4000 characters")
                print("   - Set Max Documents per Batch to 3-5")
                print("   - Set Max Total Characters to 800,000")
            
            else:
                print("🔧 For GPUs with > 12GB VRAM:")
                print("   - Set Max Chunk Size to 4000-6000 characters")
                print("   - Set Max Documents per Batch to 5-8")
                print("   - Set Max Total Characters to 1,000,000+")
        
    except Exception as e:
        print(f"⚠️  Could not determine GPU memory: {e}")
    
    print("\n🔧 General Recommendations:")
    print("   - Process documents in smaller batches")
    print("   - Use the 'Clear GPU Memory' button regularly")
    print("   - Restart the application if memory usage gets too high")
    print("   - Consider using the data folder feature instead of uploading large files")

def clear_memory():
    """Clear all available memory."""
    print("\n🧹 Memory Cleanup")
    print("=" * 50)
    
    # Clear PyTorch memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ PyTorch GPU memory cleared")
    
    # Clear Python garbage
    collected = gc.collect()
    print(f"✅ Python garbage collection: {collected} objects collected")
    
    # Check memory after cleanup
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"📊 Memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def main():
    print("🚀 Mistral 7B Memory Optimizer")
    print("=" * 60)
    
    # Check system resources
    check_system_resources()
    
    # Check PyTorch memory
    check_pytorch_memory()
    
    # Provide recommendations
    optimize_memory_settings()
    
    # Ask if user wants to clear memory
    print("\n" + "=" * 60)
    response = input("Do you want to clear memory now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        clear_memory()
    
    print("\n✅ Memory optimization complete!")
    print("\n💡 Next steps:")
    print("1. Restart the application if needed")
    print("2. Adjust memory settings in the sidebar")
    print("3. Process documents in smaller batches")
    print("4. Use the 'Clear GPU Memory' button regularly")

if __name__ == "__main__":
    main() 