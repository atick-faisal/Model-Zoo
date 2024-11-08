import torch
import torch.nn as nn


class FusionGenerator(nn.Module):
    def __init__(self):
        super(FusionGenerator, self).__init__()

        # First layer: 5x5 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Second layer: 5x5 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Third layer: 3x3 conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Fourth layer: 3x3 conv
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Fifth layer: 1x1 conv with tanh activation
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the generator
        Args:
            x: Input tensor of shape (batch_size, 2, H, W) containing concatenated IR and visible images
        Returns:
            Fused image tensor of shape (batch_size, 1, H-16, W-16)
        """
        x = self.conv1(x)  # Output: (batch_size, 64, H-4, W-4)
        x = self.conv2(x)  # Output: (batch_size, 128, H-8, W-8)
        x = self.conv3(x)  # Output: (batch_size, 256, H-10, W-10)
        x = self.conv4(x)  # Output: (batch_size, 128, H-12, W-12)
        x = self.conv5(x)  # Output: (batch_size, 1, H-16, W-16)
        return x


def calculate_output_size(input_size):
    """
    Calculate the output size given an input size
    Args:
        input_size: tuple of (H, W) representing input dimensions
    Returns:
        tuple of (H, W) representing output dimensions
    """
    H, W = input_size
    # Each conv layer reduces dimensions by (kernel_size - 1)
    # Layer 1 and 2: 5x5 kernel -> reduce by 4 each
    # Layer 3 and 4: 3x3 kernel -> reduce by 2 each
    # Layer 5: 1x1 kernel -> no reduction
    H_out = H - (4 + 4 + 2 + 2 + 0)
    W_out = W - (4 + 4 + 2 + 2 + 0)
    return (H_out, W_out)


# Example usage
if __name__ == "__main__":
    # Create the generator
    generator = FusionGenerator()

    # Example input size (batch_size, channels, height, width)
    batch_size = 4
    input_height = 256
    input_width = 256

    # Calculate output size
    output_height, output_width = calculate_output_size((input_height, input_width))
    print(f"Input size: {input_height}x{input_width}")
    print(f"Output size: {output_height}x{output_width}")

    # Create example input (batch_size, 2, H, W)
    # 2 channels: one for IR image, one for visible image
    x = torch.randn(batch_size, 2, input_height, input_width)

    # Forward pass
    output = generator(x)
    print(f"Output shape: {output.shape}")