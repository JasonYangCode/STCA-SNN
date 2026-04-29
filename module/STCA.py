import torch.nn as nn
from module.SCAU import Three_D_Coordinate_Attention_Unit

class Temporal_Attention_Unit(nn.Module):
    def __init__(self, T, ratio=2):
        super(Temporal_Attention_Unit, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(T, T // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(T // ratio, T, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        out1 = self.sharedMLP(avg)
        return out1

class Spatio_Temporal_Coordinate_Attention(nn.Module):
    def __init__(self, T, out_channels, ratio_1=8, ratio_2=16):
        super().__init__()
        self.TAU = Temporal_Attention_Unit(T=T, ratio=ratio_1)
        self.TDCAU = Three_D_Coordinate_Attention_Unit(inp=out_channels, oup=out_channels,
                                                       reduction=ratio_2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        TAU = self.TAU(x_seq)
        SCA = self.TDCAU(x_seq)
        out = self.sigmoid(TAU + SCA)
        y_seq = out * spikes
        return y_seq