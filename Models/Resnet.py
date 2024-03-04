import torch
import torch.nn as nn

def calc_out_conv_layer(in_h, in_w, ker, pad = 0, dil = 1, stri = 1):
    out_h = in_h
    out_w = in_w
    
    out_h = (out_h + 2*pad - dil * (ker-1) - 1)//stri + 1
    out_w = (out_w + 2*pad - dil * (ker-1) - 1)//stri + 1

    return out_h, out_w

class CustomResNet(nn.Module):
    def __init__(self, observation_space, features_dim: int = 256, 
                 hidden_channels: int = 32, n_cnn_layers: int = 3, 
                 stride: int = 1, dropout : float = 0.2):
        super(CustomResNet, self).__init__()
        self.dropout =  dropout   
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[2]
        image_h = observation_space.shape[0]
        image_w = observation_space.shape[1]
        out_h_first_2, out_w_first_2 = calc_out_conv_layer(image_h, image_w, n_input_channels, stri = stride)
        #Now the size is hidden_channels x out_h x out_w
        out_h_first = out_h_first_2 // 2
        out_w_first = out_w_first_2 // 2
        
        self.first_cnn_layer = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden_channels, kernel_size=3, stride=stride),
            nn.LayerNorm([hidden_channels, out_h_first_2, out_w_first_2]),  # Add Layer Normalization
            nn.GELU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        ##Now the size is hidden_channels x out_h_first x out_w_first
        out_h_2, out_w_2 = calc_out_conv_layer(out_h_first, out_w_first, 3, stri = stride)
        
        self.second_cnn_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride),
            nn.LayerNorm([hidden_channels, out_h_2, out_w_2]),  # Add Layer Normalization
            nn.GELU(),
            nn.Dropout2d(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #Now the size is hidden_channels x out_h x out_w
        out_h = out_h_2 // 2
        out_w = out_w_2 // 2
        
        # Define a list to hold the CNN layers
        self.cnn_layers = nn.ModuleList()
        
        # Add the specified number of convolutional layers
        for index in range(n_cnn_layers - 2):
            modules = []
            modules.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding="same"))
            modules.append(nn.LayerNorm([hidden_channels, out_h, out_w]))  # Add Layer Normalization
            modules.append(nn.GELU())  # Use GELU activation function after LN
            #modules.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Add MaxPooling layer
            modules.append(nn.Dropout2d(p=0.2))  # Add Dropout (DropBlock) layer
            self.cnn_layers.append(nn.Sequential(*modules))

        # Define the Flatten layer
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Define a dummy input tensor
            dummy_input = torch.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()
            # Perform a forward pass through the CNN layers
            cnn_output = dummy_input
            cnn_output = self.first_cnn_layer(cnn_output)
            cnn_output = self.second_cnn_layer(cnn_output)
            for layer in self.cnn_layers:
                cnn_output = cnn_output + layer(cnn_output)
            # Compute the shape after flattening
            n_flatten = self.flatten(cnn_output).shape[1]

        # Define the linear layer
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.GELU())
        # Print the number of learnable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters for the CNN:", num_params)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Perform a forward pass through the CNN layers
        cnn_output = self.first_cnn_layer(observations)
        cnn_output = self.second_cnn_layer(cnn_output)
        for index, layer in enumerate(self.cnn_layers):
            layer_output = layer(cnn_output)
            cnn_output = cnn_output + layer_output
        # Flatten the output
        cnn_output = self.flatten(cnn_output)
        # Pass the flattened output through the linear layer
        return self.linear(cnn_output)