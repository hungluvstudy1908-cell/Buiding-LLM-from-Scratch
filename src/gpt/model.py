import torch
import torch.nn as nn
from layer import LayerNorm as LN
from layer import FeedForward as FF
from attention import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiheadAttention(d_in = cfg["emb"],
                                      d_out = cfg["emb_dim"],
                                      context_length = cfg["context_length"],
                                      num_head = cfg["n_head"])
        
        self.LN1 = LN(emb_dim = cfg["emb_dim"])
        self.LN2 = LN(emb_dim = cfg["emb_dim"])
        self.FF = FF(emb_dim = cfg["emb_dim"])
        self.drop_out = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        
        x = self.LN1(x)
        x = self.mha(x)
        x = self.drop_out(x)
        x = x + shortcut
        
        shortcut = x
        x = self.LN2(x)
        x = self.FF(x)
        x = self.drop_out(x)
        x = shortcut + x
        
        return x
        
        

class GPTmodel(nn.Module):
    def __init__ (self, cfg):
        super().__init__(self)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.TranformerBlocks = nn.Sequetital (*[Transformer(cfg) for _ in range (cfg["n_layer"])])
        self.Final_norm = LN(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
        
    def Foward(self, in_idx):
        batch, seq_length = in_idx.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = tok_emb + pos_emb
        
        x= self.drop_emb(x)
        x = self.TranformerBlock(x)
        x = self.Final_norm(x)
        logits = self.out_head(x)
        
        return logits
        
        