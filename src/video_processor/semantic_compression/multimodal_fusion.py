"""
Multimodal Fusion Module for Semantic Compression

This module implements advanced fusion techniques to combine importance scores
from multiple modalities (visual, audio, transcript) into a unified importance score.
The fusion is context-aware, temporally aligned, and adaptively weighted based on content type.

Key components:
1. Cross-modal attention mechanisms
2. Temporal alignment algorithms
3. Dynamic weighting strategies
4. Content-type adaptive processing
5. GPU/CPU optimization

Author: HyperbolicLearner Team
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
import math

# Configure logging
logger = logging.getLogger(__name__)

# Define content types for adaptive processing
class ContentType(Enum):
    """Enum representing different types of video content"""
    LECTURE = "lecture"                  # Educational lecture content
    TUTORIAL = "tutorial"                # Step-by-step tutorials
    DEMONSTRATION = "demonstration"      # Product/technique demonstrations
    CONVERSATION = "conversation"        # Dialogue between multiple people
    PRESENTATION = "presentation"        # Slides or visual presentation
    TECHNICAL = "technical"              # Technical explanations with diagrams
    ENTERTAINMENT = "entertainment"      # Entertainment content
    MIXED = "mixed"                      # Mixed content types


@dataclass
class ModalityFeatures:
    """Container for features extracted from a specific modality"""
    features: torch.Tensor                # Feature tensor [batch_size, seq_len, feature_dim]
    attention_mask: torch.Tensor         # Attention mask [batch_size, seq_len]
    timestamps: torch.Tensor             # Timestamps for each feature [batch_size, seq_len]
    confidence: torch.Tensor             # Confidence scores [batch_size, seq_len]


@dataclass
class FusionConfig:
    """Configuration for the fusion process"""
    # General configuration
    hidden_dim: int = 256                # Hidden dimension for fusion models
    num_heads: int = 8                   # Number of attention heads
    dropout: float = 0.1                 # Dropout rate
    num_layers: int = 4                  # Number of transformer layers
    
    # Modality-specific weights
    modality_weights: Dict[str, float] = None  # Initial weights for each modality
    
    # Temporal alignment configuration
    max_temporal_distance: float = 2.0   # Maximum temporal distance in seconds
    temporal_sigma: float = 0.5          # Sigma for temporal Gaussian weighting
    
    # Content-type specific configurations
    content_type_configs: Dict[ContentType, Dict] = None
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.modality_weights is None:
            self.modality_weights = {
                "visual": 1.0, 
                "audio": 1.0, 
                "transcript": 1.0
            }
            
        if self.content_type_configs is None:
            self.content_type_configs = {
                ContentType.LECTURE: {
                    "visual_weight": 0.7,
                    "audio_weight": 0.8,
                    "transcript_weight": 1.0,
                },
                ContentType.TUTORIAL: {
                    "visual_weight": 1.0,
                    "audio_weight": 0.7,
                    "transcript_weight": 0.8,
                },
                ContentType.DEMONSTRATION: {
                    "visual_weight": 1.0,
                    "audio_weight": 0.6,
                    "transcript_weight": 0.7,
                },
                ContentType.CONVERSATION: {
                    "visual_weight": 0.5,
                    "audio_weight": 1.0,
                    "transcript_weight": 0.9,
                },
                ContentType.PRESENTATION: {
                    "visual_weight": 0.9,
                    "audio_weight": 0.6,
                    "transcript_weight": 0.8,
                },
                ContentType.TECHNICAL: {
                    "visual_weight": 0.9,
                    "audio_weight": 0.7,
                    "transcript_weight": 1.0,
                },
                ContentType.ENTERTAINMENT: {
                    "visual_weight": 0.8,
                    "audio_weight": 0.9,
                    "transcript_weight": 0.6,
                },
                ContentType.MIXED: {
                    "visual_weight": 0.8,
                    "audio_weight": 0.8,
                    "transcript_weight": 0.8,
                },
            }


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalAlignment(nn.Module):
    """
    Aligns features from different modalities based on their timestamps.
    Uses a temporal attention mechanism to align features across time.
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
    def compute_temporal_weights(self, 
                               source_timestamps: torch.Tensor, 
                               target_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal weights between source and target timestamps
        
        Args:
            source_timestamps: Source timestamps [batch_size, source_seq_len]
            target_timestamps: Target timestamps [batch_size, target_seq_len]
            
        Returns:
            Temporal weights [batch_size, target_seq_len, source_seq_len]
        """
        # Reshape timestamps for broadcasting
        # [batch_size, target_seq_len, 1]
        target = target_timestamps.unsqueeze(-1)
        # [batch_size, 1, source_seq_len]
        source = source_timestamps.unsqueeze(1)
        
        # Calculate temporal distance
        # [batch_size, target_seq_len, source_seq_len]
        temporal_distance = torch.abs(target - source)
        
        # Apply Gaussian weighting
        # [batch_size, target_seq_len, source_seq_len]
        weights = torch.exp(-(temporal_distance**2) / (2 * self.config.temporal_sigma**2))
        
        # Apply cutoff for max temporal distance
        mask = (temporal_distance > self.config.max_temporal_distance)
        weights = weights.masked_fill(mask, 0.0)
        
        # Normalize weights
        weights = F.normalize(weights, p=1, dim=2)
        
        return weights
    
    def align_features(self, 
                      source_features: ModalityFeatures, 
                      target_features: ModalityFeatures) -> torch.Tensor:
        """
        Align source features to target feature timestamps
        
        Args:
            source_features: Source modality features
            target_features: Target modality features (reference timestamps)
            
        Returns:
            Aligned source features [batch_size, target_seq_len, feature_dim]
        """
        # Compute temporal weights
        # [batch_size, target_seq_len, source_seq_len]
        temporal_weights = self.compute_temporal_weights(
            source_features.timestamps, 
            target_features.timestamps
        )
        
        # Apply mask based on confidence and attention mask
        source_mask = source_features.attention_mask.unsqueeze(1)  # [batch_size, 1, source_seq_len]
        temporal_weights = temporal_weights * source_mask
        
        # Re-normalize after masking
        normalizer = torch.sum(temporal_weights, dim=2, keepdim=True).clamp(min=1e-10)
        temporal_weights = temporal_weights / normalizer
        
        # Apply weighted combination
        # [batch_size, target_seq_len, feature_dim]
        aligned_features = torch.bmm(temporal_weights, source_features.features)
        
        return aligned_features


class CrossModalAttention(nn.Module):
    """
    Implements cross-modal attention to allow different modalities to attend to each other.
    Uses a multi-head attention mechanism for more expressive feature interactions.
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
    def forward(self, 
               query_features: torch.Tensor, 
               key_value_features: torch.Tensor,
               query_mask: torch.Tensor = None,
               kv_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply cross-modal attention
        
        Args:
            query_features: Query features [batch_size, query_seq_len, hidden_dim]
            key_value_features: Key-value features [batch_size, kv_seq_len, hidden_dim]
            query_mask: Attention mask for query [batch_size, query_seq_len]
            kv_mask: Attention mask for key-value [batch_size, kv_seq_len]
            
        Returns:
            Attended features [batch_size, query_seq_len, hidden_dim]
        """
        # Convert boolean masks to float attention masks and combine them
        if query_mask is not None:
            query_attn_mask = ~query_mask
        else:
            query_attn_mask = None
            
        if kv_mask is not None:
            kv_attn_mask = ~kv_mask
        else:
            kv_attn_mask = None
            
        # Create cross attention mask if needed
        attn_mask = None
        if query_attn_mask is not None and kv_attn_mask is not None:
            # For cross-attention, the mask should be of shape [batch_size, query_seq_len, kv_seq_len]
            attn_mask = torch.bmm(
                query_attn_mask.float().unsqueeze(-1), 
                kv_attn_mask.float().unsqueeze(1)
            )
            # Convert to boolean mask for MultiheadAttention
            attn_mask = attn_mask.to(torch.bool)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(
            query=query_features,
            key=key_value_features,
            value=key_value_features,
            attn_mask=attn_mask
        )
        
        # Add & norm (first residual connection)
        query_features = self.norm1(query_features + attn_output)
        
        # Feed-forward network
        ff_output = self.ff(query_features)
        
        # Add & norm (second residual connection)
        output = self.norm2(query_features + ff_output)
        
        return output


class ModalityProjection(nn.Module):
    """
    Projects features from each modality to a common embedding space.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to common embedding space
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Projected features [batch_size, seq_len, output_dim]
        """
        return self.norm(self.projection(x))


class ModalityFusionTransformer(nn.Module):
    """
    Transformer-based model for fusing features from multiple modalities.
    Includes cross-modal attention, self-attention, and feed-forward layers.
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Importance score prediction
        self.importance_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    def forward(self, 
               visual_features: torch.Tensor,
               audio_features: torch.Tensor,
               transcript_features: torch.Tensor,
               visual_mask: torch.Tensor = None,
               audio_mask: torch.Tensor = None,
               transcript_mask: torch.Tensor = None) -> torch

