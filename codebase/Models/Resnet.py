import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_out_conv_layer(in_h, in_w, ker, pad=0, dil=1, stri=1):
    """
    Calculate the output height and width of a convolutional layer.

    Args:
        in_h (int): Input height.
        in_w (int): Input width.
        ker (int): Kernel size.
        pad (int, optional): Padding size. Defaults to 0.
        dil (int, optional): Dilation. Defaults to 1.
        stri (int, optional): Stride. Defaults to 1.

    Returns:
        tuple: Output height and output width.
    """
    out_h = (in_h + 2 * pad - dil * (ker - 1) - 1) // stri + 1
    out_w = (in_w + 2 * pad - dil * (ker - 1) - 1) // stri + 1
    return out_h, out_w

class CustomResNet(nn.Module):
    """
    A custom Residual Network (ResNet) style architecture with configurable layers and dimensions.
    This implementation is designed for image processing tasks with configurable convolutional layers.

    Args:
        observation_space (gym.spaces): Environment observation space.
        features_dim (int, optional): Dimension of the output feature vector. Defaults to 256.
        hidden_channels (int, optional): Number of channels in hidden layers. Defaults to 32.
        n_cnn_layers (int, optional): Number of convolutional layers. Defaults to 3.
        stride (int, optional): Stride for convolutional layers. Defaults to 1.
        dropout (float, optional): Dropout rate for dropout layers. Defaults to 0.2.
    """
    def __init__(self, observation_space_shape, features_dim=256, hidden_channels=32, n_cnn_layers=3, stride=1, dropout=0.2):
        super(CustomResNet, self).__init__()
        self.dropout = dropout
        n_input_channels = observation_space_shape[2]
        image_h, image_w = observation_space_shape[:2]

        # Initialize the first convolutional layer
        self.first_cnn_layer = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden_channels, kernel_size=3, stride=stride, padding=1),
            nn.LayerNorm([hidden_channels, image_h, image_w]),
            nn.GELU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate output dimensions after the first layer
        out_h, out_w = calc_out_conv_layer(image_h, image_w, 3, pad=1, stri=stride)
        out_h //= 2
        out_w //= 2

        # Initialize subsequent convolutional layers
        self.cnn_layers = nn.ModuleList()
        for _ in range(n_cnn_layers - 1):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1),
                nn.LayerNorm([hidden_channels, out_h, out_w]),
                nn.GELU(),
                nn.Dropout2d(p=self.dropout),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

            # Calculate new dimensions after adding each layer
            out_h, out_w = calc_out_conv_layer(out_h, out_w, 3, pad=1, stri=stride)
            out_h //= 2
            out_w //= 2

        # Flatten layer to prepare for linear layer
        self.flatten = nn.Flatten()

        # Number of flattened features
        n_flatten = hidden_channels * out_h * out_w

        # Final linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim),
        )

        # Print the number of learnable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CustomResNet model.

        Args:
            observations (torch.Tensor): Input tensor, typically images with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Processed tensor, resulting in a feature vector per observation.
        """
        cnn_output = observations
        for layer in [self.first_cnn_layer] + list(self.cnn_layers):
            cnn_output = layer(cnn_output)
        flattened_output = self.flatten(cnn_output)
        return self.linear(flattened_output)