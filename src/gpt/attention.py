import torch
import torch.nn as nn




class MuitiheadAttentio(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_head, dropout_rate ,qkv_bias = False):
        super().__init__()
        assert d_out % num_head == 0, "d_out % num_head phai == 0"
        
        self.num_head = num_head
        self.Query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Value = nn.Linear(d_in, d_out, bias = qkv_bias)
        
        self.head_dim = d_out // num_head
        
        self.Dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length),diagonal = 1))
        self.out_proj = nn.Linear(d_out, d_out)
        
    def forward(self, x):
        batch, num_token, d_in = x.shape
        num_head = self.num_head
        head_dim = self.head_dim
        d_out = num_head * head_dim
        
        Query = self.Query(x) # [Batch, num_token, d_out]
        Key = self.Key(x)
        Value = self.Value(x)
        
        # Process muitilhead
        Query = Query.view (batch, num_token, num_head, head_dim)
        Key = Key.view (batch, num_token, num_head, head_dim)
        Value = Value.view(batch, num_token, num_head, head_dim)
        
        Query = Query.transpose(1, 2) # Batch, num_head, num_token, head_dim
        Key = Key.transpose(1, 2)
        Value = Value.transpose(1, 2)
        
        attn_scores = Query @ Key.transpose(2, 3) #attn_scores [batch, num_head, head_dim, head_dim]
        
        
        masked_bool = self.mask.bool()[:num_token,:num_token]
        
        attn_scores.masked_fill_(masked_bool, -torch.inf)
        
        
        attn_weights = torch.softmax((attn_scores)/ head_dim**0.5, dim = -1)
        
        #attn_weight [batch, num_head, head_dim, head_dim]
        attn_weights = self.Dropout(attn_weights)
        
        context_vector = (attn_weights @ Value)
        context_vector = context_vector.contiugous().view(batch,num_token, d_out)
        
        context_vector = self.out_proj(context_vector) 
        
        return context_vector