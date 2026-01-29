import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        # convolution with ones for mask update
        self.register_buffer(
            "conv_all_ones",
            torch.ones(1, 1, kernel_size, kernel_size),
        )

        self.window_size = kernel_size * kernel_size

    def forward(self, x, mask):
        # apply mask
        x = x * mask

        # standard convolution
        x = self.conv(x)

        # update mask
        with torch.no_grad():
            mask_sum = F.conv2d(
                mask,
                self.conv_all_ones,
                stride=self.stride,
                padding=self.padding,
            )
            mask_new = (mask_sum > 0).float()

        # normalization
        x = x * (self.window_size / (mask_sum + 1e-8))
        x = x * mask_new

        return x, mask_new
    

class PartialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.pconv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.relu(x)
        return x, mask


class PartialUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ========= Encoder =========
        self.enc1 = PartialBlock(3, 32, stride=1)     # 256×256
        self.enc2 = PartialBlock(32, 64, stride=2)    # 128×128
        self.enc3 = PartialBlock(64, 128, stride=2)   # 64×64
        self.enc4 = PartialBlock(128, 128, stride=2)  # 32×32

        # ========= Decoder =========
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = PartialBlock(128 + 128, 64, stride=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = PartialBlock(64 + 64, 32, stride=1)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.final = PartialConv2d(32 + 32, 3, kernel_size=3, stride=1)

    def forward(self, x, mask):
        # ========= Encoder =========
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)

        # ========= Decoder =========
        d1 = self.up1(e4)
        m4u = self.up1(m4)
        d1, md1 = self.dec1(
            torch.cat([d1, e3], dim=1),
            torch.max(m4u, m3),
        )

        d2 = self.up2(d1)
        md1u = self.up2(md1)
        d2, md2 = self.dec2(
            torch.cat([d2, e2], dim=1),
            torch.max(md1u, m2),
        )

        d3 = self.up3(d2)
        md2u = self.up3(md2)

        out, _ = self.final(
            torch.cat([d3, e1], dim=1),
            torch.max(md2u, m1),
        )

        return torch.sigmoid(out)
    

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = PartialUNet().to(device)
    model.eval()

    # dummy input
    x = torch.randn(1, 3, 256, 256).to(device)   # image
    mask = torch.ones(1, 1, 256, 256).to(device) # full valid mask

    # forward pass
    with torch.no_grad():
        y = model(x, mask)

    # print shapes
    print("Input image shape :", x.shape)
    print("Input mask shape  :", mask.shape)
    print("Output image shape:", y.shape)




