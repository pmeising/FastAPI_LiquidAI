import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:8002"

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")

def test_metrics():
    """Test the metrics endpoint"""
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Metrics endpoint status: {response.status_code}")
    if response.status_code == 200:
        # Show first few lines of metrics
        metrics_lines = response.text.split('\n')[:10]
        print("Sample metrics:")
        for line in metrics_lines:
            if line and not line.startswith('#'):
                print(f"  {line}")

def generate_text(prompt, max_length=512, temperature=0.3, use_chat_template=True):
    """Generate text using the API with LiquidAI recommended parameters"""
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "min_p": 0.15,
        "repetition_penalty": 1.05,
        "use_chat_template": use_chat_template
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"Request time: {end_time - start_time:.2f}s")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    # Test health
    print("=== Testing Health Endpoint ===")
    test_health()
    
    print("\n=== Testing Metrics Endpoint ===")
    test_metrics()
    
    # Example prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to success in machine learning is"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n=== Testing prompt {i+1}: '{prompt}' ===")
        generate_text(prompt, max_length=256, temperature=0.3, use_chat_template=True)
        time.sleep(1)  # Small delay between requests
    
    print("\n=== Final Health Check (with metrics) ===")
    test_health()
