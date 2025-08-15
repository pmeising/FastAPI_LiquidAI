#!/usr/bin/env python3
"""
Basic test script to verify the model loading works locally
"""
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model_loading():
    """Test if we can load any of our fallback models"""
    
    model_options = [
        "microsoft/DialoGPT-medium",
        "gpt2", 
        "distilgpt2"
    ]
    
    for model_name in model_options:
        try:
            print(f"Testing {model_name}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"  ✓ Tokenizer loaded")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            print(f"  ✓ Model loaded")
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"  ✓ Pad token set")
            
            # Test a simple generation
            test_prompt = "Hello, how are you"
            inputs = tokenizer.encode(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  ✓ Generation test: '{generated_text}'")
            print(f"SUCCESS: {model_name} works correctly!\n")
            return True, model_name
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}\n")
            continue
    
    print("ERROR: All models failed to load")
    return False, None

if __name__ == "__main__":
    success, model_name = test_model_loading()
    if not success:
        sys.exit(1)
    else:
        print(f"Recommended model: {model_name}")
