# FastAPI LiquidAI LFM2-350M Inference Service

A fast and lightw### Python
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "What is machine learning?",
    "max_length": 256,
    "temperature": 0.3,
    "min_p": 0.15,
    "repetition_penalty": 1.05,
    "use_chat_template": True
})

result = response.json()
print(result["generated_text"])
```

### cURL
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, how are you?", "max_length": 256, "temperature": 0.3}'
```e for running inference with the LiquidAI/LFM2-350M language model in Docker.

## Features

- **Fast**: Optimized for quick inference with minimal overhead
- **Dockerized**: Easy deployment with Docker and docker-compose
- **RESTful API**: Simple HTTP endpoints for text generation
- **Health Monitoring**: Built-in health checks
- **GPU Support**: Automatic GPU detection and usage when available
- **Prometheus Metrics**: Comprehensive monitoring with 15+ metrics for performance tracking
- **Grafana Ready**: Metrics compatible with Grafana dashboards for visualization

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### Using Docker

```bash
# Build the image
docker build -t liquidai-api .

# Run the container
docker run -p 8002:8000 liquidai-api
```

## API Endpoints

### Health Check
```
GET /health
```
Returns detailed health information including model status, device info, GPU memory usage, and request statistics.

### Prometheus Metrics
```
GET /metrics
```
Prometheus-compatible metrics endpoint for monitoring and alerting. Tracks generation performance, error rates, and resource usage.

### Text Generation
```
POST /generate
Content-Type: application/json

{
    "prompt": "What is artificial intelligence?",
    "max_length": 512,
    "temperature": 0.3,
    "min_p": 0.15,
    "repetition_penalty": 1.05,
    "use_chat_template": true
}
```

## Usage Examples

### Python
```python
import requests

response = requests.post("http://localhost:8002/generate", json={
    "prompt": "The future of artificial intelligence is",
    "max_length": 100,
    "temperature": 0.7
})

result = response.json()
print(result["generated_text"])
```

### cURL
```bash
curl -X POST "http://localhost:8002/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello world", "max_length": 50}'
```

## Testing

Run the example usage script:
```bash
python example_usage.py
```

## Model Information

- **Model**: LiquidAI/LFM2-350M
- **Type**: Causal Language Model
- **Parameters**: 350M
- **Framework**: Transformers/PyTorch

## Requirements

- Docker
- At least 2GB RAM (4GB recommended)
- Optional: NVIDIA GPU for faster inference

## Metrics and Monitoring

The service exposes comprehensive Prometheus metrics at `/metrics`:

### Request Metrics
- `liquidai_generation_requests_total` - Total generation requests
- `liquidai_generation_errors_total` - Total generation errors
- `liquidai_generation_duration_seconds` - End-to-end generation time

### Performance Metrics
- `liquidai_inference_duration_seconds` - Model inference time
- `liquidai_tokenization_duration_seconds` - Input tokenization time
- `liquidai_decoding_duration_seconds` - Output decoding time
- `liquidai_text_processing_duration_seconds` - Text processing time

### Content Metrics
- `liquidai_prompt_length_characters` - Input prompt length
- `liquidai_generated_text_length_characters` - Generated text length
- `liquidai_prompt_tokens_total` - Input token count
- `liquidai_generated_tokens_total` - Generated token count

### Model Metrics
- `liquidai_model_loaded` - Model loading status (0/1)
- `liquidai_model_temperature` - Temperature values used
- `liquidai_model_max_length` - Max length values used

### Integration with Grafana
These metrics are designed to work with your existing Prometheus/Grafana setup in the `MLOps_Monitoring` directory.

## API Documentation

Once running, visit `http://localhost:8002/docs` for interactive API documentation.

## Troubleshooting

### Common Issues

#### "accelerate" Package Error
If you see an error like:
```
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`
```

**Solution**: The service now includes automatic fallback logic. This error should be caught and handled gracefully. If it persists:

1. Ensure you're using Docker (recommended):
   ```bash
   docker-compose up --build
   ```

2. If running locally, install accelerate:
   ```bash
   pip install accelerate>=0.25.0
   ```

3. Test the setup:
   ```bash
   python test_accelerate.py
   ```

#### GPU Memory Issues
If you encounter CUDA out-of-memory errors:

1. The service automatically detects and uses available GPU memory
2. Check available GPU memory in health endpoint: `GET /health`
3. Reduce batch size or use CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`

#### Model Loading Timeout
For slow model downloads on first run:

1. Ensure stable internet connection
2. The LiquidAI/LFM2-350M model (~700MB) is downloaded on first startup
3. Subsequent runs will use cached model

#### Service Won't Start
1. Check logs: `docker-compose logs liquidai-api`
2. Verify port 8002 is available
3. Ensure Docker has sufficient memory allocated (minimum 2GB)
