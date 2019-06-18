# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_input_channels, 64, 3)
        self.down1 = down(64, 128, 3, stride=4)
        self.down2 = down(128, 128, 3, stride=4)
        self.up1 = up(256, 64, conv_size=3, bilinear=False)
        self.up2 = up(128, 64, conv_size=3, bilinear=False)
        self.outc = outconv(64, n_output_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)