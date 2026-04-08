import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Every step in the U-Net consists of two consecutive 3x3 convolutions,
    each followed by a Batch Normalization and a ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # padding=1 ensures the spatial dimensions (height/width) don't shrink during convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. ENCODER (Downsampling path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. BOTTLENECK (The bottom of the "U")
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2) # 512 -> 1024

        # 3. DECODER (Upsampling path)
        for feature in reversed(features):
            # Up-convolution (Transposed Convolution) to double the spatial dimensions
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # After upsampling, concatenate the skip connection and pass through DoubleConv
            self.ups.append(DoubleConv(feature * 2, feature))

        # 4. FINAL OUTPUT LAYER
        # Maps the 64 feature channels down to the requested number of output channels (1 for binary mask)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Pass through Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # Save features for the skip connections
            x = self.pool(x)

        # Pass through Bottleneck
        x = self.bottleneck(x)

        # Reverse the skip connections list to easily match with the Decoder layers
        skip_connections = skip_connections[::-1]

        # Pass through Decoder
        # Step by 2 because self.ups contains both the ConvTranspose2d and the DoubleConv
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Upsample
            
            skip_connection = skip_connections[idx//2] # Grab corresponding skip connection
            
            # Failsafe: if the image size isn't perfectly divisible by 16, resize to match.
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            # Concatenate the skip connection and the upsampled image along the channels dimension (dim=1)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Pass through DoubleConv
            x = self.ups[idx+1](concat_skip)

        # Note: We output raw logits here. We will apply the Sigmoid activation automatically 
        # inside the loss function (BCEWithLogitsLoss) during training for better numerical stability.
        return self.final_conv(x)


# --- Quick Test to Verify Architecture ---
if __name__ == "__main__":
    # Create a dummy image tensor: Batch Size of 1, 3 Channels (RGB), 256x256 Height/Width
    dummy_input = torch.randn((1, 3, 256, 256))
    
    # Initialize the model
    model = UNet(in_channels=3, out_channels=1)
    
    # Pass the dummy image through the model
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}  -> (Batch, Channels, Height, Width)")
    print(f"Output shape: {output.shape}  -> (Batch, Classes, Height, Width)")
    
    # The output shape must exactly match the input's height and width, but with 1 channel!
    if output.shape == (1, 1, 256, 256):
        print("Success! The U-Net architecture is correctly built and shapes match perfectly.")
    else:
        print("Error: Output shape does not match expected dimensions.")