import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerArchitecture(nn.Module):
    """
    Implements a simplified version of the Transformer architecture as described in
    "Attention is All You Need" by Vaswani et al. This class allows for customizable
    configuration of the transformer model with varying number of layers, heads,
    and dimensionality.

    Args:
        positional_embedding (nn.Module): The positional embedding module to be used.
        positions (torch.Tensor, optional): Tensor specifying the positions; if None, positions
                                            are generated up to `max_step_len`.
        activation (nn.Module, optional): Activation to be used in feed-forward networks.
        d_model (int): Dimensionality of the model and embeddings.
        n_head (int): Number of attention heads.
        n_layer (int): Number of stacked transformer layers.
        d_ff (int): Dimension of the feed-forward network.
        max_step_len (int): Maximum length of the input sequences allowed.
        dropout (float): Dropout rate for regularization.
        batch_first (bool): Whether the first dimension of the input tensor represents batch size.

    Attributes:
        dropout (float): Dropout rate.
        activation (nn.Module): Activation function used in feed-forward networks.
        positions (torch.Tensor): Positions tensor used for positional encoding.
        positional_embedding (nn.Module): Instance of the positional embedding module.
        attentions (nn.ModuleList): List of multi-head attention layers.
        linear1 (nn.ModuleList): First linear transformation in the feed-forward network.
        linear2 (nn.ModuleList): Second linear transformation in the feed-forward network.
        layer_norms1 (nn.ModuleList): Layer normalization after the attention layers.
        layer_norms2 (nn.ModuleList): Layer normalization after the feed-forward network.
    """
    def __init__(self, 
                 positional_embedding: nn.Module, 
                 positions: torch.Tensor = None,
                 activation: nn.Module = nn.GELU(),
                 d_model: int = 256, 
                 n_head: int = 8, 
                 n_layer: int = 24, 
                 d_ff: int = 1024,
                 max_step_len: int = 512, 
                 dropout: float = 0.0, 
                 batch_first: bool = True
                ):
        super(TransformerArchitecture, self).__init__()

        self.dropout = dropout
        self.activation = activation
        self.max_step_len = max_step_len
        self.d_model = d_model

        if positions is None:
            positions = torch.arange(max_step_len)
        self.positions = positions

        self.positional_emb_callable = positional_embedding
        self.positional_embedding = self.positional_emb_callable(
            d_model=d_model,
            max_len=max_step_len,
            position=self.positions
        )

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
            for _ in range(n_layer)
        ])

        self.linear1 = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(n_layer)])
        self.linear2 = nn.ModuleList([nn.Linear(d_ff, d_model) for _ in range(n_layer)])

        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, d_model].
            attention_mask (torch.Tensor, optional): Optional mask for the attention layers to ignore certain positions.

        Returns:
            torch.Tensor: Output tensor of the transformer model with the same shape as `x`.
        """
        x = self.positional_embedding(x)

        for i in range(len(self.attentions)):
            # Multi-head Attention layer
            residual = x
            x, _ = self.attentions[i](x, x, x, attn_mask=attention_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.layer_norms1[i](x)

            # Feed-forward network
            residual = x
            x = self.activation(self.linear1[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.linear2[i](x)
            x = residual + x
            x = self.layer_norms2[i](x)

        return x

    def setPosition(self, positions):
        """
        Update the positional embeddings with new positions.

        Args:
            positions (torch.Tensor): New positions tensor to update the model's positional encodings.
        """
        self.positions = positions
        self.positional_embedding = self.positional_emb_callable(
            d_model=self.d_model,
            max_len=self.max_step_len,
            position=self.positions
        )