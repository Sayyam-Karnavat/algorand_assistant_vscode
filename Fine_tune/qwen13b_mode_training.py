import subprocess
import sys
import logging
import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import gc


def install_packages():
    """Install required packages with proper error handling."""
    packages = [
        "--no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo",
        'sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer',
        "transformers==4.51.3",
        "--no-deps unsloth"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
            print(f"✓ Successfully installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            raise
# Configuration
class ModelConfig:
    def __init__(self):
        self.model_name = "unsloth/Meta-Llama-3.1-8B"
        self.max_seq_length = 2048
        self.dtype = None  # Auto-detection
        self.load_in_4bit = True
        self.device_map = "auto"
        self.trust_remote_code = True
        
    def get_optimal_dtype(self):
        """Determine optimal dtype based on GPU capability."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "a100" in gpu_name or "h100" in gpu_name or "rtx" in gpu_name:
                return torch.bfloat16
            elif "v100" in gpu_name or "t4" in gpu_name:
                return torch.float16
        return None