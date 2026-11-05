import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-10
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True)
        LN = self.scale * ((x - mean) / torch.sqrt(var + self.eps))
        return LN
    
        
class GELU(nn.Module):
    def foward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi)**0.5 * (x + 0.044715 * x ** 3)))
    

class FeedFoward (nn.Module):
    def __init__(self, emb_dim, drop_rate = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(drop_rate)
        )
    def forward(self, x):
        x = self.net(x)
        return x