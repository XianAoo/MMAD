import torch
import torch.nn as nn
import torch.nn.functional as F
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

class TimeDependentClassifier(nn.Module):
    """Time-conditioned classifier base."""
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

# ================= 1. MLP-based Classifier =================
class MLPClassifier(TimeDependentClassifier):
    def __init__(self, input_dim=96, time_dim=64, hidden_dim=256, num_classes=2):
        super().__init__(time_dim)
        
        # input_dim = 2 * 32 = 64
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim), # Concat time embedding
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, t):
        # x: (B, 2, 32)
        B = x.shape[0]
        x = x.reshape(B, -1) # Flatten -> (B, 64)
        
        x_emb = self.input_proj(x) # (B, hidden)
        t_emb = self.time_mlp(t)   # (B, time_dim)
        
        h = torch.cat([x_emb, t_emb], dim=1)
        return self.net(h)

# ================= 2. CNN-based Classifier (1D) =================
class CNNClassifier(TimeDependentClassifier):
    def __init__(self, in_channels=3, time_dim=64, num_classes=2):
        super().__init__(time_dim)
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.time_proj = nn.Linear(time_dim, 256) 
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, t):
        # x: (B, 2, 32)
        t_emb = self.time_mlp(t) # (B, time_dim)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # (B, 256, 32)
        
        x = self.pool(x).squeeze(-1) # (B, 256)
        
        t_proj = self.time_proj(t_emb)
        x = x + t_proj
        
        return self.fc(x)

# ================= 3. LSTM-based Classifier =================
class LSTMClassifier(TimeDependentClassifier):
    def __init__(self, in_channels=3, time_dim=64, hidden_dim=128, num_classes=2):
        super().__init__(time_dim)
        
        self.lstm = nn.LSTM(
            input_size=in_channels + time_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes) # Bidirectional

    def forward(self, x, t):
        x = x.transpose(1, 2)
        B, L, C = x.shape
        
        t_emb = self.time_mlp(t) # (B, time_dim)
        
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, L, -1)
        
        x_in = torch.cat([x, t_emb_expanded], dim=-1)
        
        # LSTM
        # output: (B, L, 2*H), (h_n, c_n)
        output, _ = self.lstm(x_in)
        
        last_hidden = output[:, -1, :]
        
        return self.fc(last_hidden)

# ================= 4. Transformer-based Classifier =================
class TransformerClassifier(TimeDependentClassifier):
    def __init__(self, in_channels=3, sequence_length=32, time_dim=64, model_dim=128, num_heads=4, depth=3, num_classes=2):
        super().__init__(time_dim)
        
        self.input_proj = nn.Linear(in_channels, model_dim)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        self.time_proj = nn.Linear(time_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x, t):
        # x: (B, 2, 32) -> (B, 32, 2)
        x = x.transpose(1, 2)
        
        x = self.input_proj(x) # (B, 32, dim)
        t_emb = self.time_proj(self.time_mlp(t)) # (B, dim)
        
        # Add Position and Time
        x = x + self.pos_emb + t_emb.unsqueeze(1)
        
        x = self.transformer(x) # (B, 32, dim)
        
        # Global Pooling -> (B, dim)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        
        return self.fc(x)

def get_classifier(name, device='cuda', sequence_length=32, in_channels=3):
    if name == 'mlp':
        return MLPClassifier(input_dim=in_channels * sequence_length).to(device)
    elif name == 'cnn':
        return CNNClassifier(in_channels=in_channels).to(device)
    elif name == 'lstm':
        return LSTMClassifier(in_channels=in_channels).to(device)
    elif name == 'transformer':
        return TransformerClassifier(
            in_channels=in_channels, sequence_length=sequence_length
        ).to(device)
    else:
        raise ValueError(f"Unknown classifier type: {name}")
