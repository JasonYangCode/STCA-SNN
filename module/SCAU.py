# module/SCAU.py
import torch
import torch.nn as nn


class Three_D_Coordinate_Attention_Unit(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Three_D_Coordinate_Attention_Unit, self).__init__()
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_c_avg = nn.AdaptiveAvgPool2d((1, 1))

        mip = inp // reduction
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_c = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)

        identity = x

        x_c_avg = self.pool_c_avg(x)

        x_c_expanded = x_c_avg.repeat(1, 1, C, 1)

        x_h_avg = self.pool_h_avg(x)

        x_w_avg = self.pool_w_avg(x).permute(0, 1, 3, 2)

        y = torch.cat([x_c_expanded, x_h_avg, x_w_avg], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_c, x_h, x_w = torch.split(y, [C, H, W],
                                    dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        a_c = self.conv_c(
            x_c.mean(dim=2, keepdim=True)).sigmoid()

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = a_w * a_h * a_c

        out = out.reshape(B, T, C, H, W)

        return out

