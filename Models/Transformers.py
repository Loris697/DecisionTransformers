import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def repeat_integers(integers):
    # Repeat each integer three times
    repeated_integers = torch.repeat_interleave(integers, 3)
    # Trim the tensor to the maximum length
    repeated_integers = repeated_integers
    return repeated_integers

def set_elements_to_zero(row, index):
    # Create a mask tensor
    mask = torch.zeros_like(row)
    mask[index:] = 1
    # Set elements before the index to zero
    row = row * mask
    return torch.nan_to_num(row, neginf = -float('inf'))

def step_masking(step_len):
    seq_len = step_len * 3
    attention_mask = torch.full((seq_len, seq_len), -float('inf'), dtype=torch.float32)
    for step in range(1, step_len + 1):
        row_set = step - 1
        for sequence_element in [row_set * 3, row_set * 3 + 1, row_set * 3 + 2]:
            attention_mask[sequence_element] = set_elements_to_zero(attention_mask[sequence_element], step * 3)
    
    return attention_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 512, position: torch.Tensor = None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if position is None:
            position = torch.arange(max_len)

        pe = torch.zeros(1, max_len, d_model)  # Shape: [1, max_len, d_model]
            
        position = position.float().unsqueeze(1)  # Shape: [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0) 
        argument = position @ div_term

        pe[:, :, 0::2] = torch.sin(argument)
        pe[:, :, 1::2] = torch.cos(argument)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # Instead of adding positional encodings along the sequence dimension,
        # we add them along the batch dimension.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerArchitecture(nn.Module):
    def __init__(self, d_model=256, n_head=8, n_layer=24, 
                 d_ff=1024, max_step_len=512, dropout = 0.0, batch_first = True):
        super(TransformerArchitecture, self).__init__()
        self.dropout = dropout
        self.activation = nn.GELU()

        positions = repeat_integers(torch.arange(0, max_step_len))
        
        self.positional_embedding = PositionalEncoding(d_model,max_len = max_step_len*3, position = positions)
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_head, batch_first = batch_first)
            for _ in range(n_layer)
        ])
        self.linear1 = nn.ModuleList([
            nn.Linear(d_model, d_ff)
            for _ in range(n_layer)
        ])
        self.linear2 = nn.ModuleList([
            nn.Linear(d_ff, d_model)
            for _ in range(n_layer)
        ])

        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        
    def forward(self, x):
        # Each step is made by reward, observation, action
        step_len = x.size(1) // 3 
        
        x = self.positional_embedding(x)
        
        # Masking
        attention_mask = step_masking(step_len).to(device=x.device, dtype=torch.float32)
            
        for i in range(len(self.attentions)):
            # Multi-head Attention
            residual = x
            x, _ = self.attentions[i](x, x, x, attn_mask=attention_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x  # Residual connection
            x = self.layer_norms1[i](x)
            
            # Feed-forward network
            residual = x
            x = self.activation(self.linear1[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.linear2[i](x)
            x = residual + x  # Residual connection
            x = self.layer_norms2[i](x)
            
        return x