"""
Streamlit UI for LLaMA LoRA fine-tuning.
"""

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
import sys
import os
from datetime import datetime
import gc
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_lora import create_llama_lora, estimate_llama_lora_params
from training.llama_lora_trainer import create_llama_lora_trainer, LLaMALoRAConfig
from utils.llama_loader import estimate_lora_memory_usage, cleanup_memory
import threading

# Page config
st.set_page_config(
    page_title="LLaMA LoRA Studio",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #1e5128;
        border: 1px solid #2e7d32;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #5d4e37;
        border: 1px solid #f57c00;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #1e3a5f;
        border: 1px solid #1976d2;
    }
    .model-card {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'llama_model' not in st.session_state:
    st.session_state.llama_model = None
if 'llama_tokenizer' not in st.session_state:
    st.session_state.llama_tokenizer = None
if 'llama_trainer' not in st.session_state:
    st.session_state.llama_trainer = None
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'training_active' not in st.session_state:
    st.session_state.training_active = False


def main():
    st.title("🦙 LLaMA LoRA Fine-tuning Studio")
    st.markdown("Fine-tune Large Language Models with Low-Rank Adaptation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        model_options = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama 1.1B (Recommended for testing)",
            "meta-llama/Llama-2-7b-hf": "Llama 2 7B",
            "meta-llama/Llama-2-13b-hf": "Llama 2 13B",
            "mistralai/Mistral-7B-v0.1": "Mistral 7B",
            "custom": "Custom Model Path"
        }
        
        selected_model = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        if selected_model == "custom":
            custom_model_path = st.text_input("Model Path/Name")
            model_name = custom_model_path if custom_model_path else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        else:
            model_name = selected_model
        
        # Quantization settings
        st.subheader("Quantization")
        use_8bit = st.checkbox("Load in 8-bit", value=True, help="Reduces memory usage by ~50%")
        use_4bit = st.checkbox("Load in 4-bit", value=False, help="Reduces memory usage by ~75%")
        
        if use_4bit:
            use_8bit = False
        
        # LoRA configuration
        st.subheader("LoRA Configuration")
        lora_r = st.slider("LoRA Rank (r)", 4, 64, 16, 4,
                          help="Higher rank = more parameters but better quality")
        lora_alpha = st.slider("LoRA Alpha", 8, 128, 32, 8,
                             help="Scaling factor, typically 2x rank")
        lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.05, 0.05)
        
        target_modules = st.multiselect(
            "Target Modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ["q_proj", "v_proj"],
            help="Which layers to apply LoRA to"
        )
        
        # Training configuration
        st.subheader("Training Configuration")
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-3,
            value=2e-4,
            format="%.2e",
            help="Higher for LoRA than full fine-tuning"
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=1,
            help="Reduce if OOM"
        )
        
        gradient_accumulation = st.number_input(
            "Gradient Accumulation Steps",
            min_value=1,
            max_value=64,
            value=16,
            help="Effective batch = batch_size * accumulation"
        )
        
        num_epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            max_value=10,
            value=3
        )
        
        max_seq_length = st.slider(
            "Max Sequence Length",
            256, 4096, 2048, 256,
            help="Longer = more memory"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "📊 Dataset",
        "🎯 Training", 
        "🔬 Testing",
        "📈 Monitoring"
    ])
    
    # Home Tab
    with tab1:
        st.header("Welcome to LLaMA LoRA Studio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What is LoRA?
            
            **Low-Rank Adaptation** is a technique that:
            - Trains only 0.1-1% of model parameters
            - Reduces memory usage by 90%+
            - Maintains model quality
            - Allows multiple adapters on one base model
            
            ### Supported Models
            - Llama 2 (7B, 13B, 70B)
            - Llama 3
            - Mistral
            - TinyLlama (for testing)
            - Any HuggingFace causal LM
            """)
        
        with col2:
            # Memory estimation
            st.subheader("Memory Requirements")
            
            model_size = "7b" if "7b" in model_name.lower() else "1b"
            if "13b" in model_name.lower():
                model_size = "13b"
            elif "tiny" in model_name.lower():
                model_size = "1b"
            
            # Estimate memory
            base_model_gb = {"1b": 2, "7b": 14, "13b": 26, "70b": 140}.get(model_size, 14)
            
            memory_est = estimate_lora_memory_usage(
                model_size_gb=base_model_gb,
                rank=lora_r,
                target_modules_fraction=len(target_modules) * 0.02,
                batch_size=batch_size,
                sequence_length=max_seq_length,
                gradient_accumulation_steps=gradient_accumulation,
                quantized=use_8bit or use_4bit
            )
            
            # Display metrics
            col1_m, col2_m = st.columns(2)
            with col1_m:
                st.metric("Model Memory", f"{memory_est['model_memory_gb']:.1f} GB")
                st.metric("LoRA Parameters", f"{memory_est['lora_params_gb']*1024:.1f} MB")
            with col2_m:
                st.metric("Training Memory", f"{memory_est['total_memory_gb']:.1f} GB")
                st.metric("Recommended GPU", f"{memory_est['recommended_gpu_memory_gb']:.0f} GB")
            
            # Parameter estimation
            if model_size in ["7b", "13b"]:
                param_est = estimate_llama_lora_params(
                    model_size=model_size,
                    rank=lora_r,
                    target_modules=target_modules
                )
                
                st.info(f"""
                **Parameter Efficiency**
                - Base model: {param_est['base_parameters']/1e9:.1f}B parameters
                - LoRA parameters: {param_est['lora_parameters']/1e6:.1f}M parameters
                - Efficiency: {param_est['percentage']:.3f}% trainable
                """)
    
    # Dataset Tab
    with tab2:
        st.header("Dataset Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Training Data")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'json', 'jsonl', 'csv'],
                help="Upload your training dataset"
            )
            
            if uploaded_file is not None:
                # Read file
                content = uploaded_file.read().decode('utf-8')
                
                # Parse based on file type
                if uploaded_file.name.endswith('.json'):
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            texts = data
                        elif isinstance(data, dict) and 'texts' in data:
                            texts = data['texts']
                        else:
                            texts = [str(data)]
                    except:
                        texts = content.split('\n\n')
                elif uploaded_file.name.endswith('.jsonl'):
                    texts = []
                    for line in content.strip().split('\n'):
                        try:
                            item = json.loads(line)
                            texts.append(item.get('text', str(item)))
                        except:
                            continue
                else:
                    texts = content.split('\n\n')
                
                # Filter empty texts
                texts = [t.strip() for t in texts if t.strip()]
                
                st.success(f"Loaded {len(texts)} text samples")
                
                # Preview
                st.text_area(
                    "Dataset Preview (first 3 samples)",
                    '\n\n---\n\n'.join(texts[:3]),
                    height=300
                )
                
                # Save to session state
                st.session_state.dataset_texts = texts
                st.session_state.dataset_name = uploaded_file.name.split('.')[0]
        
        with col2:
            if 'dataset_texts' in st.session_state:
                st.subheader("Dataset Statistics")
                
                texts = st.session_state.dataset_texts
                total_chars = sum(len(t) for t in texts)
                avg_length = total_chars / len(texts) if texts else 0
                
                st.metric("Total Samples", f"{len(texts):,}")
                st.metric("Total Characters", f"{total_chars:,}")
                st.metric("Average Length", f"{avg_length:.0f} chars")
                
                # Length distribution
                lengths = [len(t) for t in texts]
                df = pd.DataFrame({'Length': lengths})
                
                fig = px.histogram(
                    df, x='Length',
                    title="Text Length Distribution",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Training Tab
    with tab3:
        st.header("LoRA Fine-tuning")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Load Model", type="primary", use_container_width=True):
                with st.spinner(f"Loading {model_name}..."):
                    try:
                        # Clear memory first
                        cleanup_memory()
                        
                        # Create model
                        lora_model, tokenizer = create_llama_lora(
                            model_name=model_name,
                            load_in_8bit=use_8bit,
                            load_in_4bit=use_4bit,
                            lora_r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            target_modules=target_modules
                        )
                        
                        st.session_state.llama_model = lora_model
                        st.session_state.llama_tokenizer = tokenizer
                        
                        st.success("Model loaded successfully!")
                        
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
                        if "CUDA out of memory" in str(e):
                            st.warning("Try reducing batch size or using 8-bit/4-bit quantization")
        
        with col2:
            if st.button("▶️ Start Training", type="primary", use_container_width=True):
                if st.session_state.llama_model is None:
                    st.error("Please load model first")
                elif 'dataset_texts' not in st.session_state:
                    st.error("Please upload dataset first")
                elif st.session_state.training_active:
                    st.warning("Training already in progress")
                else:
                    # Create trainer
                    output_dir = f"./llama_lora_{st.session_state.dataset_name}"
                    
                    with st.spinner("Preparing trainer..."):
                        try:
                            trainer = create_llama_lora_trainer(
                                model_name=model_name,
                                dataset_path=None,  # We'll use texts directly
                                output_dir=output_dir,
                                load_in_8bit=use_8bit,
                                lora_r=lora_r,
                                lora_alpha=lora_alpha,
                                learning_rate=learning_rate,
                                num_epochs=num_epochs,
                                per_device_batch_size=batch_size,
                                gradient_accumulation_steps=gradient_accumulation,
                                max_seq_length=max_seq_length
                            )
                            
                            # Override with our dataset
                            from training.llama_lora_trainer import prepare_dataset_for_llama
                            import numpy as np
                            
                            # Split dataset
                            texts = st.session_state.dataset_texts
                            np.random.shuffle(texts)
                            split_idx = int(len(texts) * 0.9)
                            
                            train_dataset = prepare_dataset_for_llama(
                                texts[:split_idx],
                                st.session_state.llama_tokenizer,
                                max_seq_length
                            )
                            eval_dataset = prepare_dataset_for_llama(
                                texts[split_idx:],
                                st.session_state.llama_tokenizer,
                                max_seq_length
                            )
                            
                            trainer.train_dataset = train_dataset
                            trainer.eval_dataset = eval_dataset
                            trainer.trainer.train_dataset = train_dataset
                            trainer.trainer.eval_dataset = eval_dataset
                            
                            st.session_state.llama_trainer = trainer
                            
                            # Start training in thread
                            def train_model():
                                st.session_state.training_active = True
                                try:
                                    trainer.train()
                                except Exception as e:
                                    print(f"Training error: {e}")
                                finally:
                                    st.session_state.training_active = False
                            
                            thread = threading.Thread(target=train_model)
                            thread.start()
                            st.session_state.training_thread = thread
                            
                            st.success("Training started!")
                            
                        except Exception as e:
                            st.error(f"Failed to start training: {str(e)}")
        
        with col3:
            if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                if st.session_state.training_active:
                    # TODO: Implement proper stopping mechanism
                    st.info("Stopping training... (may take a moment)")
                    st.session_state.training_active = False
        
        # Training status
        if st.session_state.training_active:
            st.markdown("### Training Progress")
            
            # Create placeholders
            progress_container = st.container()
            metrics_container = st.container()
            
            # Show live updates
            with progress_container:
                st.info("🔄 Training in progress...")
                
                # TODO: Add proper progress tracking from HF Trainer
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress (replace with actual progress)
                for i in range(100):
                    if not st.session_state.training_active:
                        break
                    progress_bar.progress(i / 100)
                    status_text.text(f"Step {i}/100")
                    time.sleep(0.1)
            
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Loss", "0.000")
                with col2:
                    st.metric("Learning Rate", f"{learning_rate:.2e}")
                with col3:
                    st.metric("Samples/sec", "0")
                with col4:
                    st.metric("Time Elapsed", "00:00")
    
    # Testing Tab
    with tab4:
        st.header("Model Testing")
        
        if st.session_state.llama_model is None:
            st.warning("Please load a model first")
        else:
            # Generation settings
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Generation Settings")
                gen_max_tokens = st.slider("Max Tokens", 10, 500, 100)
                gen_temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
                gen_top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
                gen_top_k = st.slider("Top-k", 1, 100, 50)
            
            with col1:
                st.subheader("Text Generation")
                
                prompt = st.text_area(
                    "Enter your prompt:",
                    "Once upon a time in a land far away,",
                    height=100
                )
                
                col1_g, col2_g = st.columns(2)
                
                with col1_g:
                    if st.button("Generate with Base Model", type="secondary"):
                        with st.spinner("Generating..."):
                            try:
                                # Disable LoRA
                                st.session_state.llama_model.disable_adapters()
                                
                                # Generate
                                inputs = st.session_state.llama_tokenizer(
                                    prompt, return_tensors="pt"
                                ).to(st.session_state.llama_model.model.device)
                                
                                outputs = st.session_state.llama_model.model.generate(
                                    **inputs,
                                    max_new_tokens=gen_max_tokens,
                                    temperature=gen_temperature,
                                    top_p=gen_top_p,
                                    top_k=gen_top_k,
                                    do_sample=True,
                                    pad_token_id=st.session_state.llama_tokenizer.pad_token_id
                                )
                                
                                generated = st.session_state.llama_tokenizer.decode(
                                    outputs[0], skip_special_tokens=True
                                )
                                
                                st.text_area("Base Model Output:", generated, height=200)
                                
                                # Re-enable LoRA
                                st.session_state.llama_model.enable_adapters()
                                
                            except Exception as e:
                                st.error(f"Generation failed: {str(e)}")
                
                with col2_g:
                    if st.button("Generate with LoRA", type="primary"):
                        with st.spinner("Generating..."):
                            try:
                                # Generate with LoRA
                                inputs = st.session_state.llama_tokenizer(
                                    prompt, return_tensors="pt"
                                ).to(st.session_state.llama_model.model.device)
                                
                                outputs = st.session_state.llama_model.model.generate(
                                    **inputs,
                                    max_new_tokens=gen_max_tokens,
                                    temperature=gen_temperature,
                                    top_p=gen_top_p,
                                    top_k=gen_top_k,
                                    do_sample=True,
                                    pad_token_id=st.session_state.llama_tokenizer.pad_token_id
                                )
                                
                                generated = st.session_state.llama_tokenizer.decode(
                                    outputs[0], skip_special_tokens=True
                                )
                                
                                st.text_area("LoRA Model Output:", generated, height=200)
                                
                            except Exception as e:
                                st.error(f"Generation failed: {str(e)}")
    
    # Monitoring Tab
    with tab5:
        st.header("Training Monitoring")
        
        # Check for saved checkpoints
        checkpoint_dirs = list(Path(".").glob("llama_lora_*"))
        
        if checkpoint_dirs:
            st.subheader("Saved Checkpoints")
            
            for checkpoint_dir in checkpoint_dirs:
                with st.expander(f"📁 {checkpoint_dir.name}"):
                    # List checkpoint files
                    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
                    
                    if checkpoints:
                        st.write(f"Found {len(checkpoints)} checkpoints")
                        
                        # Load button
                        if st.button(f"Load Latest from {checkpoint_dir.name}", key=checkpoint_dir.name):
                            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                            st.info(f"Loading from {latest}...")
                            # TODO: Implement checkpoint loading
                    
                    # Training history
                    trainer_state_path = checkpoint_dir / "trainer_state.json"
                    if trainer_state_path.exists():
                        with open(trainer_state_path) as f:
                            trainer_state = json.load(f)
                        
                        # Plot loss history
                        if 'log_history' in trainer_state:
                            history = trainer_state['log_history']
                            
                            # Extract metrics
                            steps = []
                            losses = []
                            eval_losses = []
                            
                            for entry in history:
                                if 'loss' in entry:
                                    steps.append(entry['step'])
                                    losses.append(entry['loss'])
                                if 'eval_loss' in entry:
                                    eval_losses.append(entry['eval_loss'])
                            
                            # Create plot
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=steps,
                                y=losses,
                                mode='lines',
                                name='Training Loss'
                            ))
                            
                            if eval_losses:
                                eval_steps = [e['step'] for e in history if 'eval_loss' in e]
                                fig.add_trace(go.Scatter(
                                    x=eval_steps,
                                    y=eval_losses,
                                    mode='lines+markers',
                                    name='Validation Loss'
                                ))
                            
                            fig.update_layout(
                                title="Training History",
                                xaxis_title="Steps",
                                yaxis_title="Loss",
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No checkpoints found yet. Train a model to see results here.")


if __name__ == "__main__":
    main()