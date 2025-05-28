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

def load_model_and_tokenizer(config):
    """Load model and tokenizer with error handling."""
    try:
        print(f"Loading model: {config.model_name}")
        print(f"Max sequence length: {config.max_seq_length}")
        print(f"4-bit quantization: {config.load_in_4bit}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=config.get_optimal_dtype(),
            load_in_4bit=config.load_in_4bit,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            # token="hf_...", # Add your HuggingFace token if needed
        )
        
        print("✓ Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

# Initialize configuration and load model
config = ModelConfig()
model, tokenizer = load_model_and_tokenizer(config)




def setup_lora_adapter(model, config):
    """Configure LoRA adapter with optimized parameters."""
    lora_config = {
        "r": 16,  # Rank - balance between adaptation capacity and efficiency
        "target_modules": [
            # Attention layers
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP layers  
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_alpha": 16,  # Scaling factor (typically same as rank)
        "lora_dropout": 0.1,  # Slight dropout for regularization
        "bias": "none",  # No bias adaptation for efficiency
        "use_gradient_checkpointing": "unsloth",  # Memory optimization
        "random_state": 3407,  # Reproducibility
        "use_rslora": False,  # Rank Stabilized LoRA (experimental)
        "loftq_config": None,  # LoftQ initialization (experimental)
    }
    
    try:
        print("Configuring LoRA adapter...")
        for key, value in lora_config.items():
            print(f"  {key}: {value}")
            
        adapted_model = FastLanguageModel.get_peft_model(model, **lora_config)
        print("✓ LoRA adapter configured successfully")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in adapted_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return adapted_model
        
    except Exception as e:
        print(f"✗ Error configuring LoRA: {e}")
        raise

model = setup_lora_adapter(model, config)