import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in the paper
    "Attention Is All You Need" by Vaswani et al.

    This module computes sinusoidal positional encodings for each position
    in a sequence of maximum length `max_len`. The positional encodings are
    added to input features which could be outputs from an embedding layer.

    Args:
        d_model (int): The dimension of the embeddings (also the feature size of the model).
        dropout (float): Dropout value to be applied after adding positional encodings.
        max_len (int): Maximum length of the input sequences.
        position (torch.Tensor, optional): Tensor of positions; if not provided,
                                           positions from 0 to `max_len-1` are used.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.0, 
                 max_len: int = 512, 
                 position: torch.Tensor = None
                ):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        if position is None:
            position = torch.arange(max_len).float()

        position = position.reshape(-1, 1)

        # Precompute the divisors for the arguments of the sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ).unsqueeze(0)
        pe = torch.zeros(max_len, 1, d_model)

        # Compute the positional encodings once
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # Transpose to shape [1, max_len, d_model]

        # Register the positional encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to each sequence in the input batch, followed by dropout.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: The resulting tensor with the same shape as input `x`.
        """
        # Add positional encodings to the input tensor `x`
        x = x + self.pe[:, :x.size(1)]
        
        # Apply dropout to the resulting tensor
        return self.dropout(x)


class EmbeddingPositionalEncoding(nn.Module):
    """
    Implements embedding-based positional encoding.

    This module uses a learnable embedding to provide positional encodings for input sequences.
    The embeddings are added to the input tensor which typically comes from an earlier embedding layer
    that encodes the input tokens.

    Args:
        d_model (int): The dimension of the embeddings and the expected feature size of the model.
        dropout (float): Dropout value to be applied after adding positional encodings.
        max_len (int): Maximum length of the input sequences.
        position (torch.Tensor, optional): Tensor of predefined positions; if not provided,
                                           positions from 0 to `max_len-1` are used.
    """
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.0, 
                 max_len: int = 512, 
                 position: torch.Tensor = None
                ):
        super(EmbeddingPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the position vector if not provided
        if position is None:
            position = torch.arange(max_len).long()  # Ensuring index type is long for embedding

        # Create an embedding layer for the positional encodings
        self.pe = nn.Embedding(max_len, d_model)

        # Store the positions
        self.position = position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to each sequence in the input batch, followed by dropout.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor: The resulting tensor with the same shape as input `x`.
        """
        # Get the positional encodings for each index in the input sequence lengths
        # Using .expand_as to ensure it matches the batch size of x
        positional_encodings = self.pe(self.position[:x.size(1)]).expand_as(x)
        
        # Add the positional encodings to the input tensor
        x = x + positional_encodings
        
        # Apply dropout to the resulting tensor
        return self.dropout(x)