import torch

__all__ = ["CustomCNN"]


class CustomCNN(torch.nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_size=(28, 28),
                 output_size=10,
                 conv_layers=None,
                 fc_layers=None,
                 activation='relu',
                 use_batchnorm=False,
                 dropout_prob=0.0,
                 extractor=False):
        """
        Customizable CNN class.

        Parameters:
        - input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        - output_size (int): Number of output classes for classification.
        - conv_layers (list of dict): List of convolutional layer configurations.
          Each dict should contain:
            - 'out_channels' (int): Number of output channels.
            - 'kernel_size' (int or tuple): Size of the kernel.
            - 'stride' (int or tuple, optional): Stride of the convolution (default: 1).
            - 'padding' (int or tuple, optional): Padding added to all sides (default: 0).
        - fc_layers (list of int): List of fully connected layer sizes.
        - activation (str): Activation function ('relu', 'sigmoid', 'tanh').
        - use_batchnorm (bool): Whether to use Batch Normalization.
        - dropout_prob (float): Dropout probability (default: 0.0).
        """
        super(CustomCNN, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.activation = self._get_activation_function(activation)
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.extractor = extractor

        # Build convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        in_channels = input_channels
        tmp_dim = input_size
        for layer_cfg in conv_layers or []:
            conv_layer = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg.get('stride', 1),
                padding=layer_cfg.get('padding', 0)
            )
            self.conv_layers.append(conv_layer)
            tmp_dim = [(tmp_dim[0] - layer_cfg['kernel_size'] + 2 * layer_cfg.get('padding', 0)) // layer_cfg.get('stride', 1) + 1,
                          (tmp_dim[1] - layer_cfg['kernel_size'] + 2 * layer_cfg.get('padding', 0)) // layer_cfg.get('stride', 1) + 1]

            if use_batchnorm:
                self.conv_layers.append(torch.nn.BatchNorm2d(layer_cfg['out_channels']))

            self.conv_layers.append(self.activation)

            if dropout_prob > 0:
                self.conv_layers.append(torch.nn.Dropout2d(p=dropout_prob))

            in_channels = layer_cfg['out_channels']

        latent_dim = tmp_dim[0]*tmp_dim[1]*in_channels
        # Build fully connected layers
        self.fc_layers = torch.nn.ModuleList()
        in_features = latent_dim  # Will be determined after forward pass of conv layers
        for out_features in fc_layers or []:
            self.fc_layers.append(torch.nn.Linear(in_features, out_features))
            if use_batchnorm:
                self.fc_layers.append(torch.nn.BatchNorm1d(out_features))
            self.fc_layers.append(self.activation)
            if dropout_prob > 0:
                self.fc_layers.append(torch.nn.Dropout(p=dropout_prob))
            in_features = out_features

        # Final classification layer
        if fc_layers:
            self.fc_layers.append(torch.nn.Linear(fc_layers[-1], output_size))
        else:
            self.fc_layers.append(torch.nn.Linear(in_features, output_size))

        if self.extractor:
            self.output_size = latent_dim
        else:
            self.output_size = output_size

    def forward(self, x):
        if torch.isnan(x).all() and self.extractor:
            return torch.zeros(x.shape[0], self.output_size).to(x.device)

        for layer in self.conv_layers:
            x = layer(x)

        x = torch.flatten(x, start_dim=1)  # Flatten for FC layers

        if self.extractor:
            return x

        for layer in self.fc_layers:
            x = layer(x)

        return x

    @staticmethod
    def _get_activation_function(name):
        if name == 'relu':
            return torch.nn.ReLU()
        elif name == 'sigmoid':
            return torch.nn.Sigmoid()
        elif name == 'tanh':
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")


if __name__ == "__main__":
    pass
