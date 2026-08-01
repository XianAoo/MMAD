import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiTBlock(nn.Module):
    """
    DiT Block: Self-Attention + FeedForward + AdaLN (Time conditioning)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim) 
        )

    def forward(self, x, c, src_key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Self-Attention
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_mask = ~src_key_padding_mask.bool() if src_key_padding_mask is not None else None
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class DiT1D(nn.Module):
    def __init__(
        self,
        in_channels=2,      # IAT, Size
        dim=128,
        depth=6,
        heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=32,
        num_classes=None,    # Class Condition
        class_dropout_prob=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = in_channels 
        self.channels = in_channels
        self.self_condition = False
        self.num_classes = num_classes if num_classes is not None else 0
        self.x_embed = nn.Linear(in_channels, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))

        # 2. Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # 4. Final Layer
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )
        self.final_linear = nn.Linear(dim, in_channels)
        self.initialize_weights()

        # if num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes + 1, dim)

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x, time, x_self_cond=None, y=None, mask=None):
        """
        x: (B, C, L) -> (B, 2, 32)
        """
        x = x.transpose(1, 2)
        
        # Embedding
        x = self.x_embed(x) + self.pos_embed[:, :x.shape[1], :]
        
        # Time Embedding
        t = self.time_embed(time) # (B, dim)
        
        # if y is not None and hasattr(self, 'label_emb'):
        #     l = self.label_emb(y)
        #     c = t + l 
        # else:
        c = t

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, c, src_key_padding_mask=mask)
            # x = block(x, t, src_key_padding_mask=mask)
            
        # Final Layer
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)
        
        x = x.transpose(1, 2)
        return x
