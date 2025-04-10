import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.conv_last = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder pathway
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder pathway
        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))

        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))

        return self.conv_last(dec1)

# Helper function
def get_segmentation_model(in_channels=4, num_classes=4):
    model = UNet3D(in_channels, num_classes)
    return model
