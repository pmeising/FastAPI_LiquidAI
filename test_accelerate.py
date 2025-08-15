#!/usr/bin/env python3
"""
Test script to verify accelerate dependency and model loading fallback
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_accelerate_availability():
    """Test if accelerate is available and working"""
    try:
        import accelerate
        print(f"✓ Accelerate is available (version: {accelerate.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Accelerate not available: {e}")
        return False

def test_device_map_loading():
    """Test loading with device_map"""
    try:
        print("Testing model loading with device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",  # Small model for testing
            device_map="auto",
            torch_dtype=torch.float32
        )
        print("✓ Model loaded successfully with device_map")
        return True, model
    except Exception as e:
        print(f"✗ Failed to load with device_map: {e}")
        return False, None

def test_fallback_loading():
    """Test loading without device_map (fallback method)"""
    try:
        print("Testing fallback model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",  # Small model for testing
            torch_dtype=torch.float32
        )
        model = model.to(device)
        print(f"✓ Model loaded successfully and moved to {device}")
        return True, model
    except Exception as e:
        print(f"✗ Failed fallback loading: {e}")
        return False, None

def test_liquidai_model():
    """Test loading the actual LiquidAI model with our improved logic"""
    try:
        print("Testing LiquidAI/LFM2-350M model loading...")
        model_id = "LiquidAI/LFM2-350M"
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✓ Tokenizer loaded")
        
        # Determine device and loading strategy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Try to use accelerate for better device management, fallback if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            print(f"✓ LiquidAI model loaded with accelerate device_map on {device}")
        except (ImportError, ValueError) as e:
            print(f"⚠ Could not use device_map: {e}")
            print("Trying fallback method...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            model = model.to(device)
            print(f"✓ LiquidAI model loaded with fallback method on {device}")
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✓ Pad token configured")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"✗ Failed to load LiquidAI model: {e}")
        return False, None, None

def main():
    print("=== Testing Accelerate and Model Loading ===\n")
    
    print("1. Checking PyTorch and CUDA availability...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    print("2. Testing accelerate availability...")
    accelerate_available = test_accelerate_availability()
    print()
    
    print("3. Testing device_map loading...")
    device_map_success, _ = test_device_map_loading()
    print()
    
    print("4. Testing fallback loading...")
    fallback_success, _ = test_fallback_loading()
    print()
    
    if accelerate_available and device_map_success:
        print("5. Testing LiquidAI model loading...")
        liquidai_success, model, tokenizer = test_liquidai_model()
        print()
        
        if liquidai_success:
            print("✓ All tests passed! The API should work correctly.")
        else:
            print("✗ LiquidAI model loading failed, but fallback should work.")
    else:
        print("5. Skipping LiquidAI test due to dependency issues.")
        print("   Please ensure accelerate is properly installed:")
        print("   pip install accelerate")
        print()
    
    print("=== Test Summary ===")
    print(f"Accelerate available: {'✓' if accelerate_available else '✗'}")
    print(f"Device map loading: {'✓' if device_map_success else '✗'}")
    print(f"Fallback loading: {'✓' if fallback_success else '✗'}")

if __name__ == "__main__":
    main()
