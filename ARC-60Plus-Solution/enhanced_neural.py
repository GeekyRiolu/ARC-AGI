"""
🧠 Enhanced Neural Networks for ARC-AGI 60+ Score Solution

Advanced neural architectures with attention mechanisms,
data augmentation, and specialized pattern recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import random

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for pattern recognition"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.output(attention_output)

class EnhancedARCTransformer(nn.Module):
    """Enhanced transformer with attention and pattern-specific layers"""
    
    def __init__(self, grid_size=30, num_colors=10, d_model=256, num_heads=8, num_layers=6):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.d_model = d_model
        
        # Enhanced embeddings
        self.color_embedding = nn.Embedding(num_colors, d_model)
        self.position_embedding = nn.Embedding(grid_size * grid_size, d_model)
        self.row_embedding = nn.Embedding(grid_size, d_model // 4)
        self.col_embedding = nn.Embedding(grid_size, d_model // 4)
        
        # Pattern-specific attention layers
        self.geometric_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.color_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.spatial_attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Pattern-specific heads
        self.geometric_head = nn.Linear(d_model, num_colors)
        self.color_head = nn.Linear(d_model, num_colors)
        self.completion_head = nn.Linear(d_model, num_colors)
        
        # Final fusion layer
        self.fusion = nn.Linear(num_colors * 3, num_colors)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, height, width = x.shape
        seq_len = height * width
        
        # Flatten and create embeddings
        x_flat = x.view(batch_size, -1)
        
        # Color embeddings
        color_emb = self.color_embedding(x_flat)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Row/column embeddings
        rows = torch.arange(height, device=x.device).repeat_interleave(width).unsqueeze(0).expand(batch_size, -1)
        cols = torch.arange(width, device=x.device).repeat(height).unsqueeze(0).expand(batch_size, -1)
        row_emb = self.row_embedding(rows)
        col_emb = self.col_embedding(cols)
        
        # Combine embeddings
        embeddings = color_emb + pos_emb + torch.cat([row_emb, col_emb, 
                                                     torch.zeros(batch_size, seq_len, d_model//2, device=x.device)], dim=-1)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pattern-specific attention
        geometric_features = self.geometric_attention(embeddings)
        color_features = self.color_attention(embeddings)
        spatial_features = self.spatial_attention(embeddings)
        
        # Transformer layers
        output = embeddings
        for layer in self.transformer_layers:
            output = layer(output)
        
        # Pattern-specific predictions
        geometric_pred = self.geometric_head(geometric_features)
        color_pred = self.color_head(color_features)
        completion_pred = self.completion_head(spatial_features)
        
        # Fuse predictions
        fused = torch.cat([geometric_pred, color_pred, completion_pred], dim=-1)
        final_logits = self.fusion(fused)
        
        # Reshape to grid
        final_logits = final_logits.view(batch_size, height, width, self.num_colors)
        
        return final_logits

class EnhancedARCCNN(nn.Module):
    """Enhanced CNN with residual connections and pattern-specific filters"""
    
    def __init__(self, num_colors=10):
        super().__init__()
        self.num_colors = num_colors
        
        # Multi-scale feature extraction
        self.scale1_conv = nn.Sequential(
            nn.Conv2d(num_colors, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.scale2_conv = nn.Sequential(
            nn.Conv2d(num_colors, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.scale3_conv = nn.Sequential(
            nn.Conv2d(num_colors, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Pattern-specific filters
        self.geometric_filters = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.color_filters = nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_colors, 1)
        )
        
    def forward(self, x):
        # Convert to one-hot
        x_onehot = F.one_hot(x.long(), self.num_colors).float()
        x_onehot = x_onehot.permute(0, 3, 1, 2)
        
        # Multi-scale feature extraction
        scale1_features = self.scale1_conv(x_onehot)
        scale2_features = self.scale2_conv(x_onehot)
        scale3_features = self.scale3_conv(x_onehot)
        
        # Combine scales
        combined_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)
        
        # Pattern-specific processing
        geometric_features = self.geometric_filters(combined_features)
        color_features = self.color_filters(combined_features)
        
        # Combine pattern features
        pattern_features = torch.cat([geometric_features, color_features], dim=1)
        
        # Apply attention
        attention_weights = self.attention(pattern_features)
        attended_features = pattern_features * attention_weights
        
        # Generate output
        output = self.output_conv(attended_features)
        
        return output.permute(0, 2, 3, 1)

class ARCDataAugmenter:
    """Advanced data augmentation for ARC tasks"""
    
    def __init__(self):
        self.augmentation_strategies = [
            self.rotate_augment,
            self.reflect_augment,
            self.color_permute_augment,
            self.noise_augment,
            self.crop_augment
        ]
    
    def augment_task(self, task, num_augmentations=5):
        """Generate augmented versions of a task"""
        augmented_tasks = []
        
        for _ in range(num_augmentations):
            strategy = random.choice(self.augmentation_strategies)
            try:
                augmented_task = strategy(task)
                if augmented_task:
                    augmented_tasks.append(augmented_task)
            except:
                continue
        
        return augmented_tasks
    
    def rotate_augment(self, task):
        """Rotate task by 90, 180, or 270 degrees"""
        rotation = random.choice([1, 2, 3])
        
        augmented_task = {'train': [], 'test': []}
        
        for example in task['train']:
            inp = np.rot90(np.array(example['input']), rotation)
            out = np.rot90(np.array(example['output']), rotation)
            augmented_task['train'].append({
                'input': inp.tolist(),
                'output': out.tolist()
            })
        
        for example in task['test']:
            inp = np.rot90(np.array(example['input']), rotation)
            augmented_task['test'].append({'input': inp.tolist()})
        
        return augmented_task
    
    def reflect_augment(self, task):
        """Reflect task horizontally or vertically"""
        axis = random.choice([0, 1])  # 0 for vertical, 1 for horizontal
        
        augmented_task = {'train': [], 'test': []}
        
        for example in task['train']:
            inp = np.flip(np.array(example['input']), axis)
            out = np.flip(np.array(example['output']), axis)
            augmented_task['train'].append({
                'input': inp.tolist(),
                'output': out.tolist()
            })
        
        for example in task['test']:
            inp = np.flip(np.array(example['input']), axis)
            augmented_task['test'].append({'input': inp.tolist()})
        
        return augmented_task
    
    def color_permute_augment(self, task):
        """Permute colors in the task"""
        # Get all colors used in the task
        all_colors = set()
        for example in task['train']:
            all_colors.update(np.unique(example['input']))
            all_colors.update(np.unique(example['output']))
        
        colors = list(all_colors)
        if len(colors) < 2:
            return None
        
        # Create random permutation
        permuted_colors = colors.copy()
        random.shuffle(permuted_colors)
        color_map = dict(zip(colors, permuted_colors))
        
        augmented_task = {'train': [], 'test': []}
        
        for example in task['train']:
            inp = self._apply_color_map(example['input'], color_map)
            out = self._apply_color_map(example['output'], color_map)
            augmented_task['train'].append({
                'input': inp,
                'output': out
            })
        
        for example in task['test']:
            inp = self._apply_color_map(example['input'], color_map)
            augmented_task['test'].append({'input': inp})
        
        return augmented_task
    
    def _apply_color_map(self, grid, color_map):
        """Apply color mapping to grid"""
        grid_array = np.array(grid)
        result = grid_array.copy()
        
        for old_color, new_color in color_map.items():
            result[grid_array == old_color] = new_color
        
        return result.tolist()
    
    def noise_augment(self, task):
        """Add small amount of noise to task"""
        # Only add noise to background (color 0) with low probability
        noise_prob = 0.05
        
        augmented_task = {'train': [], 'test': []}
        
        for example in task['train']:
            inp = self._add_noise(example['input'], noise_prob)
            # Don't add noise to output to maintain correctness
            augmented_task['train'].append({
                'input': inp,
                'output': example['output']
            })
        
        for example in task['test']:
            inp = self._add_noise(example['input'], noise_prob)
            augmented_task['test'].append({'input': inp})
        
        return augmented_task
    
    def _add_noise(self, grid, noise_prob):
        """Add noise to grid"""
        grid_array = np.array(grid)
        result = grid_array.copy()
        
        # Only add noise to background cells
        background_mask = (grid_array == 0)
        noise_mask = np.random.random(grid_array.shape) < noise_prob
        
        # Apply noise only to background cells
        noise_cells = background_mask & noise_mask
        if np.any(noise_cells):
            # Add random colors (1-9) to noise cells
            result[noise_cells] = np.random.randint(1, 10, np.sum(noise_cells))
        
        return result.tolist()
    
    def crop_augment(self, task):
        """Crop and pad task to create size variations"""
        # This is more complex and would require careful implementation
        # to maintain task validity
        return None

print("🧠 Enhanced neural networks with attention and augmentation loaded")
