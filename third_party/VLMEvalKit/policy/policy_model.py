#!/usr/bin/env python3
"""
Policy Model Class
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from pathlib import Path

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_groups: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        # Default was groupnorm with 32 groups
        # Ensure groups doesn't exceed channels
        groups = min(norm_groups, out_channels)
        while groups > 1 and out_channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class VisionRegressor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512, # Inferred from input but usually 512/1024 depending on model
        out_dim: int = 1, # len(ratios) == 1 by default
        ratios: list[float] = [1.0],
        perceiver_dim: int = 512,
        conv_depth: int = 2,
        conv_kernel: int = 3,
        norm_groups: int = 32,
        mlp_hidden_dims: list[int] = [32, 16],
        num_classes: int = 2,
    ):
        super().__init__()
        self.register_buffer("ratio_values", torch.tensor(ratios, dtype=torch.float32))
        self.out_dim = out_dim
        
        # Tokenizer: Sequence of Conv1d blocks
        conv_blocks = []
        for idx in range(conv_depth):
            in_ch = hidden_size if idx == 0 else perceiver_dim
            conv_blocks.append(
                ConvBlock1d(
                    in_channels=in_ch,
                    out_channels=perceiver_dim,
                    kernel_size=conv_kernel,
                    norm_groups=norm_groups,
                )
            )
        self.tokenizer = nn.Sequential(*conv_blocks)
        
        # Global Average Pooling
        self.token_pool = nn.AdaptiveAvgPool1d(out_dim)
        
        # Pre-head Norm (Layernorm)
        token_feature_dim = perceiver_dim
        self.pre_head_norm = nn.LayerNorm(token_feature_dim)

        # Ratio Conditioning: Concat + Embed
        # Input dim increases by 1 for concat
        in_dim = token_feature_dim + 1
        # Embedding for ratio
        self.ratio_embed = nn.Embedding(out_dim, token_feature_dim)

        # MLP Head
        trunk_layers = []
        prev_dim = in_dim
        for dim in mlp_hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, dim))
            trunk_layers.append(nn.SiLU())
            prev_dim = dim
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        
        # Final Classification Head
        self.head = nn.Linear(prev_dim, num_classes)

    def forward_from_embeds(self, image_embeds: torch.Tensor) -> torch.Tensor:
        # image_embeds: (B, seq_len, hidden_size)
        if image_embeds.ndim == 2:
            image_embeds = image_embeds.unsqueeze(0)
            
        # Tokenize (Conv1d expects B, C, L)
        x = image_embeds.transpose(1, 2)
        x = self.tokenizer(x)
        
        # Pooling
        x = self.token_pool(x)
        # Back to (B, L, C)
        tokens = x.transpose(1, 2)
        
        # Pre-head norm
        tokens = self.pre_head_norm(tokens)
        
        # Ratio Conditioning (Concat + Embed)
        # Add embedding
        ratio_embed = self.ratio_embed.weight.unsqueeze(0).expand(tokens.size(0), -1, -1)
        tokens = tokens + ratio_embed
        
        # Concat ratio value
        ratio_vals = self.ratio_values.view(1, -1, 1).expand(tokens.size(0), -1, 1)
        tokens = torch.cat([tokens, ratio_vals], dim=-1)
        
        # MLP
        feats = self.trunk(tokens)
        
        # Logits
        out = self.head(feats)
        return out


class PolicyModel(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
    ):
        """
        PolicyModel initialized with fixed architecture parameters matching the trained checkpoint.
        """
        super().__init__()
        
        self.device = torch.device(device if isinstance(device, str) else device)
        if hasattr(self.device, "type") and self.device.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.", file=sys.stderr)
            self.device = torch.device("cpu")

        # Fixed Architecture Parameters based on "QwenAdaptiveK" defaults
        self.model = VisionRegressor(
            hidden_size=512,     
            out_dim=1,
            ratios=[1.0],
            perceiver_dim=512,
            conv_depth=2,
            conv_kernel=3,
            norm_groups=32,
            mlp_hidden_dims=[32, 16],
            num_classes=2,
        ).to(self.device)

        self.model_dtype = torch.float32 # Default
        
        # Load weights
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

    def _load_checkpoint(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        state = torch.load(path, map_location=self.device)
        
        # Check for tokenizer input shape mismatch fix
        # If the checkpoint first conv weight has different in_channels than our init (512), re-init the layer.
        if "model_state" in state:
            s = state["model_state"]
        elif "tokenizer" in state: # Heuristic for flattened state
             s = state
        else:
             s = state

        # Inspect tokenizer.0.conv.weight if available to adjust input dim
        keys = [k for k in s.keys() if "tokenizer.0.conv.weight" in k]
        if keys:
            key = keys[0]
            weight = s[key]
            # weight shape: (out, in, kernel)
            ckpt_in_channels = weight.shape[1]
            if ckpt_in_channels != self.model.tokenizer[0].conv.in_channels:
                 # Reconstruct the first layer with correct input dim
                 print(f"Adapting input channels from {self.model.tokenizer[0].conv.in_channels} to {ckpt_in_channels} based on checkpoint.")
                 self.model.tokenizer[0] = ConvBlock1d(
                    in_channels=ckpt_in_channels,
                    out_channels=512, # perceiver_dim
                    kernel_size=3,
                    norm_groups=32
                 ).to(self.device)

        # Load
        if "model_state" in state:
            self.model.load_state_dict(state["model_state"], strict=False)
        else:
            # Try loading components
            if "tokenizer" in state:
                self.model.tokenizer.load_state_dict(state["tokenizer"], strict=False)
            if "pre_head_norm" in state:
                self.model.pre_head_norm.load_state_dict(state["pre_head_norm"], strict=False)
            elif "norm" in state:
                 self.model.pre_head_norm.load_state_dict(state["norm"], strict=False)
            if "ratio_embed" in state:
                self.model.ratio_embed.load_state_dict(state["ratio_embed"], strict=False)
            if "head" in state:
                self.model.head.load_state_dict(state["head"], strict=False)
            # If checkpoint has "trunk" or "head", load it.
            # If `state` is just the model state dict (no "model_state" key), clean it and load.
            if not ("tokenizer" in state or "model_state" in state):
                 # Assume whole dict is state
                 self.model.load_state_dict(state, strict=False)


    def forward(self, x: torch.Tensor) -> int:
        """
        Forward pass.
        Args:
            x: Input features (B, seq, dim) or (seq, dim).
        Returns:
            int: K value.
        """
        x = x.to(self.device).to(self.model_dtype)
        
        with torch.no_grad():
            outputs = self.model.forward_from_embeds(x)
            # outputs shape: (B, seq_len(1), num_classes) usually
            
            if outputs.ndim == 3 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            
            # Now outputs shape: (B, num_classes)
            # Softmax
            probs = torch.softmax(outputs, dim=-1)
            
            # Take prob of class 1 (solvable/positive)
            # Assuming batch size 1
            if probs.ndim >= 2:
                assert probs.shape[0] == 1
                # (B, num_classes)
                prob_val = probs[0, 1].item()
            elif probs.ndim == 1:
                # (num_classes,)
                prob_val = probs[1].item()
            else:
                prob_val = probs.item()
                
        # Map to K
        return self._map_prob_to_k(prob_val)

    @staticmethod
    def _map_prob_to_k(prob: float) -> int:
        p = max(0.0, min(float(prob), 1.0))

        # 1) Give extra samples to the empirically hardest band
        if 0.02 <= p < 0.34:
            return 5

        # 2) Keep your high-K region, but slightly narrower to pay for (1)
        if 0.34 <= p < 0.60:
            return 7

        # 3) Moderate region (shorter than before to keep avg K ~ constant)
        if 0.60 <= p < 0.737:
            return 5

        if 0.737 <= p < 0.837:
            return 3

        # 4) Everything else: cheap
        return 1

if __name__ == "__main__":
    pass
