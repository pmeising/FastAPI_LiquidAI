from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import time
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
GENERATION_REQUESTS = Counter('liquidai_generation_requests_total', 'Total text generation requests')
GENERATION_DURATION = Histogram('liquidai_generation_duration_seconds', 'Time spent on text generation')
INFERENCE_DURATION = Histogram('liquidai_inference_duration_seconds', 'Time spent on model inference')
TEXT_PROCESSING_DURATION = Histogram('liquidai_text_processing_duration_seconds', 'Time spent processing text')
TOKENIZATION_DURATION = Histogram('liquidai_tokenization_duration_seconds', 'Time spent on tokenization')
DECODING_DURATION = Histogram('liquidai_decoding_duration_seconds', 'Time spent decoding output')
GENERATION_ERRORS = Counter('liquidai_generation_errors_total', 'Total text generation errors')
MODEL_LOADED = Gauge('liquidai_model_loaded', 'Whether the LiquidAI model is loaded (1) or not (0)')
PROMPT_LENGTH_CHARS = Histogram('liquidai_prompt_length_characters', 'Length of input prompts in characters')
GENERATED_TEXT_LENGTH_CHARS = Histogram('liquidai_generated_text_length_characters', 'Length of generated text in characters')
PROMPT_TOKENS = Histogram('liquidai_prompt_tokens_total', 'Number of tokens in input prompts')
GENERATED_TOKENS = Histogram('liquidai_generated_tokens_total', 'Number of tokens generated')
MODEL_TEMPERATURE = Histogram('liquidai_model_temperature', 'Temperature values used for generation')
MODEL_MAX_LENGTH = Histogram('liquidai_model_max_length', 'Max length values used for generation')

app = FastAPI(title="LiquidAI LFM2-350M Inference API", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None
loaded_model_name = None

class InferenceRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 32756
    temperature: Optional[float] = 0.3
    min_p: Optional[float] = 0.15
    repetition_penalty: Optional[float] = 1.05
    use_chat_template: Optional[bool] = True

class InferenceResponse(BaseModel):
    generated_text: str
    prompt: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer, loaded_model_name
    try:
        logger.info("Loading LiquidAI/LFM2-350M model...")
        MODEL_LOADED.set(0)  # Set to 0 during loading
        
        # Load model and tokenizer following LiquidAI recommendations
        model_id = "LiquidAI/LFM2-350M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Determine device and loading strategy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Try to use accelerate for better device management, fallback if not available
        try:
            # This will work if accelerate is properly installed
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            logger.info(f"Model loaded with accelerate device_map on {device}")
        except (ImportError, ValueError) as e:
            # Fallback: load model without device_map
            logger.warning(f"Could not use device_map (accelerate issue): {e}")
            logger.info("Falling back to manual device placement...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            # Manually move to device
            model = model.to(device)
            logger.info(f"Model loaded and moved to {device}")
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        loaded_model_name = "LiquidAI/LFM2-350M"  # Track which model was loaded
        MODEL_LOADED.set(1)  # Set to 1 when successfully loaded
        logger.info("LiquidAI/LFM2-350M model loaded successfully!")
        
    except Exception as e:
        MODEL_LOADED.set(0)  # Set to 0 on failure
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/health")
async def health_check():
    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": device_info,
        "gpu_memory_gb": gpu_memory,
        "model_name": loaded_model_name or "LiquidAI/LFM2-350M",
        "total_requests": int(GENERATION_REQUESTS._value._value),
        "total_errors": int(GENERATION_ERRORS._value._value)
    }

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint for monitoring and alerting.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Increment request counter and record parameters
    GENERATION_REQUESTS.inc()
    PROMPT_LENGTH_CHARS.observe(len(request.prompt))
    MODEL_TEMPERATURE.observe(request.temperature)
    MODEL_MAX_LENGTH.observe(request.max_length)
    
    start_time = time.time()
    
    try:
        # Prepare input with chat template if requested
        tokenization_start = time.time()
        
        if request.use_chat_template:
            # Use LiquidAI's chat template
            messages = [{"role": "user", "content": request.prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True
            ).to(model.device)
        else:
            # Direct tokenization without chat template
            input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
            # Move to the same device as the model
            input_ids = input_ids.to(model.device)
        
        prompt_token_count = input_ids.shape[1]
        PROMPT_TOKENS.observe(prompt_token_count)
        
        tokenization_time = time.time() - tokenization_start
        TOKENIZATION_DURATION.observe(tokenization_time)
        
        # Generate text using LiquidAI recommended parameters
        inference_start = time.time()
        with torch.no_grad():
            max_new_tokens = min(request.max_length, 512)  # Cap new tokens
            
            outputs = model.generate(
                input_ids,
                do_sample=True,
                temperature=request.temperature,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - inference_start
        INFERENCE_DURATION.observe(inference_time)
        
        # Decode output
        decoding_start = time.time()
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_token_count = outputs[0].shape[0] - prompt_token_count
        GENERATED_TOKENS.observe(generated_token_count)
        
        # Extract only the assistant's response if using chat template
        if request.use_chat_template:
            # Look for the assistant's response after the last <|im_start|>assistant
            if "<|im_start|>assistant" in generated_text:
                # Split and get the last assistant response
                parts = generated_text.split("<|im_start|>assistant")
                if len(parts) > 1:
                    assistant_response = parts[-1]
                    # Remove any trailing <|im_end|> tokens
                    assistant_response = assistant_response.replace("<|im_end|>", "").strip()
                    generated_text = assistant_response
        else:
            # Remove the original prompt from the generated text (for non-chat template)
            if generated_text.startswith(request.prompt):
                generated_text = generated_text[len(request.prompt):].strip()
        
        decoding_time = time.time() - decoding_start
        DECODING_DURATION.observe(decoding_time)
        
        # Record text length metrics
        GENERATED_TEXT_LENGTH_CHARS.observe(len(generated_text))
        
        # Record total duration
        total_time = time.time() - start_time
        GENERATION_DURATION.observe(total_time)
        
        return InferenceResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
        
    except Exception as e:
        GENERATION_ERRORS.inc()
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "LiquidAI LFM2-350M Inference API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
