"""
PCN-Guided Attention Mechanism

Implements attention with PCN-generated biases to guide the model
toward more effective attention patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class PCNGuidedAttention(nn.Module):
    """
    Multi-head attention with PCN-generated guidance biases.
    
    The PCN biases nudge queries, keys, and values toward regions
    that lead to better token predictions based on energy minimization.
    """
    
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        use_guidance: bool = True,
        guidance_strength: float = 0.1
    ):
        """
        Args:
            n_embed: Embedding dimension
            n_heads: Number of attention heads
            block_size: Maximum sequence length
            dropout: Dropout probability
            use_guidance: Whether to use PCN guidance
            guidance_strength: How strongly to apply guidance biases
        """
        super().__init__()
        
        assert n_embed % n_heads == 0
        
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.block_size = block_size
        self.use_guidance = use_guidance
        self.guidance_strength = guidance_strength
        
        # Standard attention projections
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        
        # Output projection
        self.proj = nn.Linear(n_embed, n_embed)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            )
        )
        
        # Guidance adaptation layers (learn how to apply biases)
        if use_guidance:
            self.q_adapt = nn.Linear(n_embed, n_embed)
            self.k_adapt = nn.Linear(n_embed, n_embed)
            self.v_adapt = nn.Linear(n_embed, n_embed)
            
            # Guidance gates (learn when to apply biases)
            self.q_gate = nn.Linear(n_embed * 2, n_embed)
            self.k_gate = nn.Linear(n_embed * 2, n_embed)
            self.v_gate = nn.Linear(n_embed * 2, n_embed)
    
    def apply_guidance(
        self,
        tensor: torch.Tensor,
        bias: torch.Tensor,
        adapt_layer: nn.Module,
        gate_layer: nn.Module,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply PCN guidance bias with learned adaptation and gating.
        
        Args:
            tensor: The tensor to be guided (Q, K, or V)
            bias: PCN-generated bias
            adapt_layer: Adaptation layer for the bias
            gate_layer: Gating layer to control bias application
            original: Original input for gate computation
            
        Returns:
            Guided tensor
        """
        # Adapt the bias to match the tensor space
        adapted_bias = adapt_layer(bias)
        
        # Compute gate based on tensor and bias similarity
        gate_input = torch.cat([tensor, adapted_bias], dim=-1)
        gate = torch.sigmoid(gate_layer(gate_input))
        
        # Apply gated bias
        guided = tensor + gate * adapted_bias * self.guidance_strength
        
        return guided
    
    def forward(
        self,
        x: torch.Tensor,
        pcn_biases: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional PCN guidance.
        
        Args:
            x: Input tensor (batch, seq_len, n_embed)
            pcn_biases: Optional (q_bias, k_bias, v_bias) from PCN
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: Attention output
            attention_weights: Optional attention weights for visualization
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Apply PCN guidance if provided
        if self.use_guidance and pcn_biases is not None:
            q_bias, k_bias, v_bias = pcn_biases
            
            # Ensure biases match sequence length
            if q_bias.shape[1] != T:
                # Interpolate or truncate as needed
                if q_bias.shape[1] > T:
                    q_bias = q_bias[:, :T]
                    k_bias = k_bias[:, :T]
                    v_bias = v_bias[:, :T]
                else:
                    # Pad with zeros
                    pad_len = T - q_bias.shape[1]
                    q_bias = F.pad(q_bias, (0, 0, 0, pad_len))
                    k_bias = F.pad(k_bias, (0, 0, 0, pad_len))
                    v_bias = F.pad(v_bias, (0, 0, 0, pad_len))
            
            # Apply guidance with adaptation and gating
            Q = self.apply_guidance(Q, q_bias, self.q_adapt, self.q_gate, x)
            K = self.apply_guidance(K, k_bias, self.k_adapt, self.k_gate, x)
            V = self.apply_guidance(V, v_bias, self.v_adapt, self.v_gate, x)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.proj_dropout(self.proj(attn_output))
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output, None


class PCNGuidedTransformerBlock(nn.Module):
    """
    Transformer block with PCN-guided attention.
    """
    
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
        use_pcn_guidance: bool = True,
        guidance_strength: float = 0.1
    ):
        super().__init__()
        
        # Guided attention
        self.attention = PCNGuidedAttention(
            n_embed=n_embed,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            use_guidance=use_pcn_guidance,
            guidance_strength=guidance_strength
        )
        
        # Standard transformer components
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        pcn_biases: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor
            pcn_biases: Optional PCN guidance biases
            
        Returns:
            Output tensor
        """
        # Self-attention with residual
        attn_out, _ = self.attention(self.ln1(x), pcn_biases)
        x = x + attn_out
        
        # Feedforward with residual
        x = x + self.ff(self.ln2(x))
        
        return x


class MultiScaleGuidedAttention(nn.Module):
    """
    Attention mechanism with multi-scale PCN guidance.
    
    Different attention heads receive guidance at different scales,
    allowing the model to capture both local and global patterns.
    """
    
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        block_size: int,
        scales: List[int] = [1, 4, 8],
        dropout: float = 0.1
    ):
        """
        Args:
            n_embed: Embedding dimension
            n_heads: Number of attention heads
            block_size: Maximum sequence length
            scales: Different scales for guidance (heads per scale)
            dropout: Dropout probability
        """
        super().__init__()
        
        assert sum(scales) == n_heads, "Scales must sum to n_heads"
        
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.scales = scales
        self.block_size = block_size
        
        # Create separate attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            PCNGuidedAttention(
                n_embed=n_embed,
                n_heads=n_heads_scale,
                block_size=block_size,
                dropout=dropout,
                use_guidance=True,
                guidance_strength=0.1 * (i + 1)  # Stronger guidance for larger scales
            )
            for i, n_heads_scale in enumerate(scales)
        ])
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(n_embed, n_embed)
            for _ in scales
        ])
        
        # Output combination
        self.combine = nn.Linear(n_embed * len(scales), n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        pcn_biases: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with multi-scale attention.
        
        Args:
            x: Input tensor
            pcn_biases: Optional PCN guidance biases
            
        Returns:
            Combined attention output
        """
        scale_outputs = []
        
        for i, (attn, proj) in enumerate(zip(self.scale_attentions, self.scale_projections)):
            # Apply attention at this scale
            if pcn_biases is not None:
                # Scale the biases for this attention module
                scale_factor = 2 ** i  # Exponentially increasing influence
                scaled_biases = tuple(
                    bias * scale_factor for bias in pcn_biases
                )
                scale_out, _ = attn(x, scaled_biases)
            else:
                scale_out, _ = attn(x)
            
            # Project to common space
            scale_out = proj(scale_out)
            scale_outputs.append(scale_out)
        
        # Combine scale outputs
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.combine(combined)
        output = self.dropout(output)
        
        return output