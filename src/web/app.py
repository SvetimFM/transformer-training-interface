from fastapi import FastAPI, WebSocket, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import asyncio
import json
import os
import sys
import queue
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import app_config, ModelConfig, TrainingConfig, GenerationConfig, DatasetConfig
from models.bigram import BigramLM
from training.trainer import Trainer
from utils.dataset_preparation import get_dataset
from utils.dataset_manager import DatasetManager
from utils.training_utils import batchifier
import torch.nn.functional as F
from visualization.hooks import register_model_components
from visualization.component_registry import component_registry
from visualization.activation_tracker import activation_tracker
from visualization.attention_capture import attention_capture

# Removed PCN managers for public release

app = FastAPI(title="Transformer Training UI")

# Add CORS middleware
# Configure CORS - in production, replace with specific origins
ALLOWED_ORIGINS = [
    "http://localhost:8050",
    "http://localhost:8051",
    "http://127.0.0.1:8050",
    "http://127.0.0.1:8051",
    # Add your production domain here when deploying
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
trainer = None
vocab = None
vocab_size = None
encode = None
decode = None
train_data = None
val_data = None
websocket_clients = []
metrics_queue = queue.Queue()
dataset_manager = None  # DatasetManager instance

# Removed PCN managers

# Request/Response models
class ConfigUpdate(BaseModel):
    model: dict = None
    training: dict = None
    generation: dict = None
    dataset: dict = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0

class TrainingControl(BaseModel):
    action: str  # "start" or "stop"

class AttentionCaptureRequest(BaseModel):
    text: str = "The quick brown fox jumps over the lazy dog"

class DatasetConfigRequest(BaseModel):
    dataset_type: str = "shakespeare"
    dataset_path: str = None
    dataset_url: str = None
    tokenizer_type: str = "character"
    tokenizer_model: str = "gpt2"
    vocab_size: int = None

# Initialize model and data
def initialize_model():
    global model, vocab, vocab_size, encode, decode, train_data, val_data, trainer, dataset_manager
    
    try:
        # Initialize or use existing dataset manager
        if dataset_manager is None:
            dataset_manager = DatasetManager(app_config.dataset)
        
        # Load dataset and create tokenizer
        dataset_text, tokenizer, dataset_info = dataset_manager.prepare_data()
        
        # Update global tokenizer functions
        encode = tokenizer.encode
        decode = tokenizer.decode
        vocab_size = tokenizer.vocab_size()
        
        # For character tokenizer, maintain vocab for compatibility
        if hasattr(tokenizer, 'char_to_id'):
            vocab = list(tokenizer.char_to_id.keys())
        else:
            vocab = None
        
        # Update config with actual vocab size
        app_config.model.vocab_size = vocab_size
        
        # Prepare data using tokenizer
        data = torch.tensor(encode(dataset_text), dtype=torch.long)
        train_size = int(app_config.training.train_split * len(data))
        train_data = data[:train_size].to(app_config.training.device)
        val_data = data[train_size:].to(app_config.training.device)
        
        # Create model
        model = BigramLM(
            vocab_size=app_config.model.vocab_size,
            batch_size=app_config.training.batch_size,
            block_size=app_config.model.block_size,
            config=app_config
        ).to(app_config.training.device)
        
        # Register model components for visualization
        register_model_components(model, app_config)
        
        # Create trainer
        trainer = Trainer(model, train_data, val_data, app_config, app_config.training.device)
        
        # Add callbacks to broadcast metrics
        trainer.add_callback("on_step", broadcast_metrics)
        trainer.add_callback("on_eval", broadcast_metrics)
        trainer.add_callback("on_training_end", broadcast_training_complete)
        
        print(f"Model initialized with {vocab_size} vocabulary size using {app_config.dataset.tokenizer_type} tokenizer")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

# WebSocket connection manager
def broadcast_metrics(metrics):
    """Thread-safe function to queue metrics for broadcasting"""
    try:
        metrics_data = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
        
        # Check if we should also send activation states
        # In visualization mode, always broadcast activations
        if app_config.training.visualization_mode or activation_tracker.should_broadcast():
            activation_states = activation_tracker.get_current_state()
            metrics_queue.put({
                "type": "activation_update", 
                "data": activation_states
            })
        
        metrics_queue.put({"type": "metrics", "data": metrics_data})
        print(f"Queued metrics: step={metrics_data.get('step', 'unknown')}, loss={metrics_data.get('train_loss', 'unknown'):.4f}")
    except Exception as e:
        print(f"Error queuing metrics: {e}")

def broadcast_training_complete(final_metrics):
    """Broadcast training completion with final metrics"""
    try:
        metrics_data = final_metrics.to_dict() if hasattr(final_metrics, 'to_dict') else final_metrics
        completion_data = {
            "type": "training_complete",
            "data": {
                "final_metrics": metrics_data,
                "message": f"Training completed! Final loss: {metrics_data['train_loss']:.4f}, Val loss: {metrics_data['val_loss']:.4f}"
            }
        }
        metrics_queue.put(completion_data)
        print(f"Training completed: {completion_data['data']['message']}")
    except Exception as e:
        print(f"Error broadcasting completion: {e}")

async def metrics_broadcaster():
    """Background task that broadcasts metrics from the queue"""
    while True:
        try:
            # Check for metrics in queue (non-blocking)
            if not metrics_queue.empty():
                message_data = metrics_queue.get_nowait()
                
                if websocket_clients:
                    # Handle both old format (raw metrics) and new format (with type)
                    if isinstance(message_data, dict) and "type" in message_data:
                        message = json.dumps(message_data)
                    else:
                        # Old format compatibility
                        message = json.dumps({
                            "type": "metrics",
                            "data": message_data
                        })
                    
                    disconnected_clients = []
                    for client in websocket_clients:
                        try:
                            await client.send_text(message)
                            print(f"Sent {message_data.get('type', 'metrics')} to client")
                        except Exception as e:
                            print(f"Failed to send to client: {e}")
                            disconnected_clients.append(client)
                    
                    for client in disconnected_clients:
                        websocket_clients.remove(client)
                        print(f"Removed disconnected client")
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"Error in metrics broadcaster: {e}")
            await asyncio.sleep(1)

# Routes
@app.on_event("startup")
async def startup_event():
    # Mount static files
    app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
    
    # Initialize model on startup
    initialize_model()
    
    # Start the metrics broadcaster background task
    asyncio.create_task(metrics_broadcaster())
    print("Started metrics broadcaster background task")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("src/web/templates/index.html", "r") as f:
        return f.read()

@app.get("/api/config")
async def get_config():
    return app_config.to_dict()

@app.post("/api/config")
async def update_config(config_update: ConfigUpdate):
    global model, trainer, dataset_manager
    
    try:
        # Update configuration
        model_updated = False
        training_updated = False
        dataset_updated = False
        
        if config_update.dataset:
            app_config.dataset = DatasetConfig(**config_update.dataset)
            dataset_updated = True
            # Reinitialize dataset manager with new config
            dataset_manager = DatasetManager(app_config.dataset)
        
        if config_update.model:
            # Validate model configuration
            new_model_config = config_update.model
            n_embed = new_model_config.get('n_embed', app_config.model.n_embed)
            n_heads = new_model_config.get('n_heads', app_config.model.n_heads)
            
            if n_embed % n_heads != 0:
                head_size = n_embed / n_heads
                valid_sizes = [n_heads * i for i in range(8, 17)]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration: n_embed ({n_embed}) must be divisible by n_heads ({n_heads}). "
                           f"Current head_size would be {head_size:.2f}, but it must be an integer. "
                           f"Try n_embed values like: {', '.join(map(str, valid_sizes))}"
                )
            
            head_size = n_embed // n_heads
            if head_size < 4:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid configuration: head_size ({head_size}) is too small. "
                           f"With n_heads={n_heads}, you need n_embed >= {n_heads * 4}."
                )
            
            app_config.model = ModelConfig(**config_update.model)
            model_updated = True
        if config_update.training:
            app_config.training = TrainingConfig(**config_update.training)
            training_updated = True
        if config_update.generation:
            app_config.generation = GenerationConfig(**config_update.generation)
        
        # Reinitialize model if model config or dataset changed
        if (model_updated or dataset_updated) and model is not None:
            initialize_model()
        
        # Update scheduler if only training config changed (lighter than full reinit)
        elif training_updated and trainer is not None and not model_updated and not dataset_updated:
            # Update the scheduler without reinitializing everything
            trainer.update_scheduler()
        
        return {"status": "success", "config": app_config.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/train")
async def control_training(control: TrainingControl):
    global trainer
    
    if trainer is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    if control.action == "start":
        activation_tracker.start_tracking()
        success = trainer.start_training()
        if success:
            return {"status": "training_started"}
        else:
            return {"status": "already_training"}
    
    elif control.action == "stop":
        success = trainer.stop_training()
        activation_tracker.stop_tracking()
        if success:
            # Broadcast status update to all clients
            status_update = {
                "type": "status",
                "data": trainer.get_status()
            }
            for client in websocket_clients:
                try:
                    await client.send_json(status_update)
                except:
                    pass
            return {"status": "training_stopped"}
        else:
            return {"status": "not_training"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@app.get("/api/train/status")
async def get_training_status():
    if trainer is None:
        return {"status": "not_initialized"}
    
    return trainer.get_status()

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    global model, encode, decode
    
    if model is None or encode is None or decode is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    try:
        # Encode prompt
        prompt_encoded = encode(request.prompt) if request.prompt else [0]
        idx = torch.tensor([prompt_encoded], dtype=torch.long).to(app_config.training.device)
        
        # Generate
        with torch.no_grad():
            for _ in range(request.max_tokens):
                idx_cond = idx[:, -app_config.model.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / request.temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        # Decode
        generated = decode(idx[0].tolist())
        return {"generated": generated, "prompt": request.prompt}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/config")
async def update_dataset_config(dataset_config: DatasetConfigRequest):
    """Update dataset and tokenizer configuration"""
    global dataset_manager
    
    try:
        # Update dataset configuration
        app_config.dataset.dataset_type = dataset_config.dataset_type
        app_config.dataset.dataset_path = dataset_config.dataset_path
        app_config.dataset.dataset_url = dataset_config.dataset_url
        app_config.dataset.tokenizer_type = dataset_config.tokenizer_type
        app_config.dataset.tokenizer_model = dataset_config.tokenizer_model
        app_config.dataset.vocab_size = dataset_config.vocab_size
        
        # Reinitialize dataset manager
        dataset_manager = DatasetManager(app_config.dataset)
        
        # Get dataset info for response
        dataset_info = dataset_manager.get_dataset_info()
        
        return {
            "status": "success",
            "config": app_config.dataset.model_dump(),
            "info": dataset_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about the current dataset"""
    global dataset_manager
    
    if dataset_manager is None:
        dataset_manager = DatasetManager(app_config.dataset)
    
    try:
        info = dataset_manager.get_dataset_info()
        return {
            "status": "success",
            "info": info,
            "config": app_config.dataset.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Handle custom dataset file upload"""
    import tempfile
    import shutil
    
    try:
        # Validate file type
        if not file.filename.endswith(('.txt', '.text')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt or .text files are allowed"
            )
        
        # Validate content type
        if file.content_type not in ['text/plain', 'text/x-plain', 'application/text', None]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {file.content_type}. Only text files are allowed."
            )
        
        # Validate file size (50MB limit)
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > app_config.dataset.max_file_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {size_mb:.1f}MB > {app_config.dataset.max_file_size_mb}MB limit"
            )
        
        # Basic content validation - ensure it's valid text
        try:
            text_content = contents.decode('utf-8')
            if len(text_content.strip()) < 100:
                raise HTTPException(
                    status_code=400,
                    detail="File content too short. Please provide at least 100 characters of text."
                )
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File contains invalid characters. Please ensure it's a valid UTF-8 text file."
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Update configuration
        app_config.dataset.dataset_type = "custom"
        app_config.dataset.dataset_path = tmp_path
        
        # Reinitialize dataset manager
        dataset_manager = DatasetManager(app_config.dataset)
        info = dataset_manager.get_dataset_info()
        
        return {
            "status": "success",
            "filename": file.filename,
            "size_mb": size_mb,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/metrics/history")
async def get_metrics_history():
    if trainer is None:
        return {"metrics": []}
    
    return {"metrics": [m.to_dict() for m in trainer.metrics_history]}

@app.get("/api/architecture")
async def get_architecture():
    """Get the model architecture for visualization"""
    return component_registry.get_architecture_graph()

@app.post("/api/attention/capture")
async def capture_attention(request: AttentionCaptureRequest):
    """Capture attention patterns from next forward pass"""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    try:
        # Start attention capture
        attention_capture.start_capture()
        
        # Do a single forward pass with some sample data
        with torch.no_grad():
            sample_text = request.text
            encoded = encode(sample_text[:min(len(sample_text), app_config.model.block_size)])
            x = torch.tensor([encoded], dtype=torch.long).to(app_config.training.device)
            
            # Forward pass (this will capture attention weights)
            model(x)
        
        # Stop capture and get patterns
        attention_capture.stop_capture()
        patterns = attention_capture.get_attention_patterns()
        head_info = attention_capture.get_head_info()
        
        # Convert patterns to serializable format
        result = {
            "text": sample_text,
            "tokens": [decode([token]) for token in encoded],
            "patterns": {},
            "head_info": head_info
        }
        
        for head_id, pattern in patterns.items():
            result["patterns"][head_id] = pattern.tolist()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lr_schedule")
async def get_lr_schedule():
    """Get the learning rate schedule for visualization"""
    if trainer is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    if hasattr(trainer.scheduler, 'get_full_schedule'):
        steps, lrs = trainer.scheduler.get_full_schedule()
        return {
            "steps": steps,
            "learning_rates": lrs,
            "schedule_info": trainer.scheduler.get_schedule_info()
        }
    else:
        return {"error": "Scheduler does not support schedule preview"}

# Removed experimental endpoints for public release

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)
    print(f"WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        # Send initial status
        if trainer:
            status = trainer.get_status()
            await websocket.send_text(json.dumps({
                "type": "status",
                "data": status
            }))
            print(f"Sent initial status to client: {status}")
        
        # Keep connection alive
        while True:
            # Wait for any message from client (ping/pong)
            data = await websocket.receive_text()
            # Could handle client messages here if needed
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)
            print(f"WebSocket client disconnected. Total clients: {len(websocket_clients)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)