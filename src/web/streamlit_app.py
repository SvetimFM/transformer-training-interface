"""
Streamlit UI for LoRA fine-tuning and model management.
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import app_config
from models.bigram import BigramLM
from models.lora_model import LoRABigramLM
from training.lora_trainer import LoRATrainer
from utils.dataset_preparation import get_dataset, prepare_vocab
from utils.lora_dataset import LoRADatasetManager
import threading

# Page config
st.set_page_config(
    page_title="Transformer LoRA Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #1e3a5f;
        border: 1px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'base_model' not in st.session_state:
    st.session_state.base_model = None
if 'lora_model' not in st.session_state:
    st.session_state.lora_model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None


def load_base_model():
    """Load the base pretrained model."""
    with st.spinner("Loading base model..."):
        try:
            # Get base vocabulary
            dataset = get_dataset()
            vocab = sorted(list(set(dataset)))
            vocab_size = len(vocab)
            
            # Update config
            app_config.model.vocab_size = vocab_size
            
            # Load latest checkpoint
            checkpoint_dir = Path(app_config.training.checkpoint_dir)
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
            
            if not checkpoints:
                st.error("No base model checkpoints found. Please train a base model first.")
                return None
                
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            # Load model
            checkpoint = torch.load(latest_checkpoint)
            model = BigramLM(
                vocab_size=vocab_size,
                batch_size=app_config.training.batch_size,
                block_size=app_config.model.block_size,
                config=app_config
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(app_config.training.device)
            model.eval()
            
            st.session_state.base_model = model
            st.session_state.vocab = vocab
            st.success(f"Base model loaded from {latest_checkpoint.name}")
            
            return model
            
        except Exception as e:
            st.error(f"Failed to load base model: {e}")
            return None


def create_lora_model(base_model):
    """Create LoRA-adapted model from base model."""
    lora_config = {
        'rank': app_config.lora.rank,
        'alpha': app_config.lora.alpha,
        'dropout': app_config.lora.dropout
    }
    
    lora_model = LoRABigramLM(
        base_model=base_model,
        lora_config=lora_config,
        target_modules=app_config.lora.target_modules
    )
    
    st.session_state.lora_model = lora_model
    return lora_model


def main():
    st.title("🤖 Transformer LoRA Studio")
    st.markdown("Fine-tune transformer models with Low-Rank Adaptation (LoRA)")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LoRA parameters
        st.subheader("LoRA Settings")
        app_config.lora.rank = st.slider("Rank", 1, 64, app_config.lora.rank)
        app_config.lora.alpha = st.slider("Alpha", 1, 128, app_config.lora.alpha)
        app_config.lora.dropout = st.slider("Dropout", 0.0, 0.5, app_config.lora.dropout, 0.05)
        
        # Training parameters
        st.subheader("Training Settings")
        app_config.training.learning_rate = st.number_input(
            "Learning Rate", 
            min_value=1e-6, 
            max_value=1e-2, 
            value=app_config.training.learning_rate,
            format="%.6f"
        )
        app_config.lora.lora_lr_multiplier = st.slider(
            "LoRA LR Multiplier", 
            0.1, 10.0, 
            app_config.lora.lora_lr_multiplier, 
            0.1
        )
        app_config.training.batch_size = st.number_input(
            "Batch Size", 
            min_value=1, 
            max_value=256, 
            value=app_config.training.batch_size
        )
        app_config.training.epochs = st.number_input(
            "Epochs", 
            min_value=1, 
            max_value=100, 
            value=5  # Fewer epochs for LoRA
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset", 
        "🎯 LoRA Training", 
        "🔬 Model Testing", 
        "📈 Results"
    ])
    
    # Dataset Tab
    with tab1:
        st.header("Dataset Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Custom Dataset")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'json', 'csv'],
                help="Upload your custom dataset for fine-tuning"
            )
            
            if uploaded_file is not None:
                # Load and preview dataset
                file_type = uploaded_file.name.split('.')[-1]
                content = uploaded_file.read().decode('utf-8')
                
                st.text_area(
                    "Dataset Preview (first 1000 chars)",
                    content[:1000] + "..." if len(content) > 1000 else content,
                    height=200
                )
                
                if st.button("Process Dataset"):
                    with st.spinner("Processing dataset..."):
                        # Initialize dataset manager
                        if st.session_state.base_model is None:
                            load_base_model()
                            
                        if st.session_state.base_model is not None:
                            manager = LoRADatasetManager(
                                st.session_state.vocab,
                                device=app_config.training.device
                            )
                            
                            # Prepare dataset
                            train_data, val_data, info = manager.prepare_dataset(
                                content,
                                train_split=0.9
                            )
                            
                            st.session_state.train_data = train_data
                            st.session_state.val_data = val_data
                            st.session_state.dataset_info = info
                            st.session_state.dataset_name = uploaded_file.name.split('.')[0]
                            
                            st.success("Dataset processed successfully!")
        
        with col2:
            if st.session_state.dataset_info:
                st.subheader("Dataset Statistics")
                info = st.session_state.dataset_info
                
                st.metric("Total Characters", f"{info['num_characters']:,}")
                st.metric("Training Size", f"{info['train_size']:,}")
                st.metric("Validation Size", f"{info['val_size']:,}")
                st.metric("Vocabulary Coverage", f"{info['vocab_coverage']:.1%}")
                
                # Character frequency chart
                if 'most_common_chars' in info:
                    chars_df = pd.DataFrame(
                        info['most_common_chars'],
                        columns=['Character', 'Count']
                    )
                    chars_df['Character'] = chars_df['Character'].apply(
                        lambda x: repr(x) if x in ['\n', '\t', ' '] else x
                    )
                    
                    fig = px.bar(
                        chars_df, 
                        x='Character', 
                        y='Count',
                        title="Most Common Characters"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # LoRA Training Tab
    with tab2:
        st.header("LoRA Fine-tuning")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Initialize Models", type="primary"):
                if st.session_state.base_model is None:
                    load_base_model()
                
                if st.session_state.base_model is not None:
                    lora_model = create_lora_model(st.session_state.base_model)
                    
                    total_params = lora_model.get_num_total_params()
                    trainable_params = lora_model.get_num_trainable_params()
                    
                    st.success(f"""
                    LoRA model created!
                    - Total parameters: {total_params:,}
                    - Trainable parameters: {trainable_params:,}
                    - Parameter efficiency: {trainable_params/total_params:.2%}
                    """)
        
        with col2:
            if st.button("Start Training", type="primary"):
                if st.session_state.lora_model is None:
                    st.error("Please initialize models first")
                elif st.session_state.dataset_info is None:
                    st.error("Please upload and process a dataset first")
                else:
                    # Create trainer
                    trainer = LoRATrainer(
                        lora_model=st.session_state.lora_model,
                        base_model=st.session_state.base_model,
                        train_data=st.session_state.train_data,
                        val_data=st.session_state.val_data,
                        config=app_config,
                        dataset_name=st.session_state.dataset_name
                    )
                    
                    st.session_state.trainer = trainer
                    
                    # Start training in background thread
                    def train():
                        trainer.start_training()
                    
                    thread = threading.Thread(target=train)
                    thread.start()
                    
                    st.success("Training started!")
        
        with col3:
            if st.button("Stop Training"):
                if st.session_state.trainer and st.session_state.trainer.is_training:
                    st.session_state.trainer.stop_training()
                    st.info("Training stopped")
        
        # Training progress
        if st.session_state.trainer:
            st.subheader("Training Progress")
            
            # Create placeholder for live updates
            progress_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Update loop
            while st.session_state.trainer and st.session_state.trainer.is_training:
                with progress_placeholder.container():
                    status = st.session_state.trainer.get_status()
                    progress = status['current_step'] / status['total_steps']
                    st.progress(progress)
                    st.text(f"Step {status['current_step']} / {status['total_steps']}")
                
                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    metrics = status['current_metrics']
                    with col1:
                        st.metric("Train Loss", f"{metrics['train_loss']:.4f}")
                    with col2:
                        st.metric("Val Loss", f"{metrics.get('val_loss', 0):.4f}")
                    with col3:
                        if 'improvement' in metrics and metrics['improvement']:
                            st.metric(
                                "Improvement", 
                                f"{metrics['improvement']:.1f}%",
                                delta=f"{metrics['improvement']:.1f}%"
                            )
                    with col4:
                        st.metric("Tokens/sec", f"{metrics.get('tokens_per_second', 0):.0f}")
                
                # Update chart
                if st.session_state.trainer.metrics_history:
                    df = pd.DataFrame([m.to_dict() for m in st.session_state.trainer.metrics_history])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['step'],
                        y=df['train_loss'],
                        mode='lines',
                        name='Train Loss'
                    ))
                    if 'val_loss' in df:
                        fig.add_trace(go.Scatter(
                            x=df['step'],
                            y=df['val_loss'],
                            mode='lines',
                            name='Val Loss'
                        ))
                    if 'base_model_loss' in df:
                        fig.add_trace(go.Scatter(
                            x=df['step'],
                            y=df['base_model_loss'],
                            mode='lines',
                            name='Base Model Loss',
                            line=dict(dash='dash')
                        ))
                    
                    fig.update_layout(
                        title="Training Progress",
                        xaxis_title="Step",
                        yaxis_title="Loss",
                        height=400
                    )
                    
                    with chart_placeholder:
                        st.plotly_chart(fig, use_container_width=True)
                
                time.sleep(1)  # Update every second
    
    # Model Testing Tab
    with tab3:
        st.header("Model Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Base Model")
            base_prompt = st.text_area(
                "Enter prompt for base model:",
                "The quick brown fox",
                height=100
            )
            
            if st.button("Generate (Base)", key="gen_base"):
                if st.session_state.base_model is not None:
                    with st.spinner("Generating..."):
                        # Encode prompt
                        manager = LoRADatasetManager(st.session_state.vocab)
                        prompt_enc = torch.tensor(
                            manager.encode(base_prompt),
                            dtype=torch.long
                        ).unsqueeze(0).to(app_config.training.device)
                        
                        # Generate
                        output = st.session_state.base_model.generate(
                            prompt_enc,
                            max_new_tokens=100
                        )
                        
                        # Decode
                        generated = manager.decode(output[0].tolist())
                        st.text_area("Base Model Output:", generated, height=200)
        
        with col2:
            st.subheader("LoRA Model")
            lora_prompt = st.text_area(
                "Enter prompt for LoRA model:",
                "The quick brown fox",
                height=100
            )
            
            if st.button("Generate (LoRA)", key="gen_lora"):
                if st.session_state.lora_model is not None:
                    with st.spinner("Generating..."):
                        # Encode prompt
                        manager = LoRADatasetManager(st.session_state.vocab)
                        prompt_enc = torch.tensor(
                            manager.encode(lora_prompt),
                            dtype=torch.long
                        ).unsqueeze(0).to(app_config.training.device)
                        
                        # Generate
                        output = st.session_state.lora_model.generate(
                            prompt_enc,
                            max_new_tokens=100
                        )
                        
                        # Decode
                        generated = manager.decode(output[0].tolist())
                        st.text_area("LoRA Model Output:", generated, height=200)
    
    # Results Tab
    with tab4:
        st.header("Training Results")
        
        # Load saved LoRA models
        lora_dir = Path(app_config.training.checkpoint_dir)
        lora_dirs = [d for d in lora_dir.iterdir() if d.is_dir() and d.name.startswith("lora_")]
        
        if lora_dirs:
            selected_lora = st.selectbox(
                "Select LoRA checkpoint:",
                [d.name for d in lora_dirs]
            )
            
            if selected_lora:
                lora_path = lora_dir / selected_lora
                
                # Load training info
                info_path = lora_path / "training_info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Dataset", info['dataset_name'])
                    with col2:
                        st.metric("Best Val Loss", f"{info.get('best_val_loss', 'N/A'):.4f}")
                    with col3:
                        st.metric("Training Steps", info.get('step', 'N/A'))
                    
                    # LoRA config
                    st.subheader("LoRA Configuration")
                    st.json(info.get('lora_config', {}))
                    
                    # Load button
                    if st.button("Load This LoRA"):
                        if st.session_state.base_model is None:
                            load_base_model()
                        
                        if st.session_state.base_model:
                            # Find best checkpoint
                            checkpoints = list(lora_path.glob("lora_*.pt"))
                            if checkpoints:
                                best_checkpoint = lora_path / "lora_best.pt"
                                if not best_checkpoint.exists():
                                    best_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                                
                                # Create LoRA model and load weights
                                lora_model = create_lora_model(st.session_state.base_model)
                                lora_model.load_lora_checkpoint(str(best_checkpoint))
                                
                                st.success(f"LoRA loaded from {best_checkpoint.name}")


if __name__ == "__main__":
    main()