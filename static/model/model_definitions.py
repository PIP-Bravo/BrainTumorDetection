import torch
import torch.nn as nn
import timm

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # After concatenation, halve the channel count
        if use_skip:
            conv_in_channels = out_channels * 2  # after concat
            conv_out_channels = out_channels     # halve channels back
        else:
            conv_in_channels = out_channels
            conv_out_channels = out_channels

        self.conv = DoubleConv(conv_in_channels, conv_out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class SwinSegmentationModel(nn.Module):
    def __init__(self, backbone='swin_tiny_patch4_window7_224', pretrained=True, num_classes=1, debug=False):
        super(SwinSegmentationModel, self).__init__()

        self.debug = debug

        # Encoder from Swin Transformer
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        feat_channels = self.encoder.feature_info.channels()  # [C0, C1, C2, C3]

        # Bottleneck after the deepest encoder feature
        self.bottleneck = DoubleConv(feat_channels[3], feat_channels[3])

        # Decoder
        self.up3 = UpsampleBlock(feat_channels[3], feat_channels[2], use_skip=True)
        self.up2 = UpsampleBlock(feat_channels[2], feat_channels[1], use_skip=True)
        self.up1 = UpsampleBlock(feat_channels[1], feat_channels[0], use_skip=True)
        self.up0 = UpsampleBlock(feat_channels[0], 64, use_skip=False)

        # Final output convolution
        self.up_final = UpsampleBlock(64, 32, use_skip=False)
        #self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        if (self.debug):
            print(f"Input: {x.shape}")  # Original input shape

        feats = self.encoder(x)
        if (self.debug):
            for i, f in enumerate(feats):
                print(f"Encoder stage {i}: {f.shape}")

        f0, f1, f2, f3 = [f.permute(0, 3, 1, 2) for f in feats]

        b3 = self.bottleneck(f3)

        d3 = self.up3(b3, skip=f2)

        d2 = self.up2(d3, skip=f1)

        d1 = self.up1(d2, skip=f0)

        d0 = self.up0(d1)

        if (self.debug):
            print(f"After bottleneck: {b3.shape}")
            print(f"After up3: {d3.shape}")
            print(f"After up2: {d2.shape}")
            print(f"After up1: {d1.shape}")
            print(f"After up0: {d0.shape}")

        out = self.up_final(d0)
        if (self.debug):
            print(f"After final upsampling: {out.shape}")

        out = self.final_conv(out)
        if (self.debug):
            print(f"Final conv output: {out.shape}")

        return torch.sigmoid(out)