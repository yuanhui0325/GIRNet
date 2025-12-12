import torch
import torch.nn as nn

def make_3d_network(args):
    return Network_3d(args)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(16, 16 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(16 // 16, 16, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # y1= self.avg_pool(x)
        # y2 = self.fc1(y1)
        # y3 = self.relu1(y2)
        # avg_out = self.fc2(y3)
        #
        # y11= self.max_pool(x)
        # y22 = self.fc1(y11)
        # y33 = self.relu1(y22)
        # max_out = self.fc2(y33)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Network_3d(nn.Module):

    def __init__(self, downsample=None):
        super(Network_3d, self).__init__()

        planes =16
        self.K, S, P, OP = (3, [2, 4, 4], 1, [ 0, 3, 3])

        self.base = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=planes, kernel_size= 3,padding=1),
            nn.PReLU(),
        )
        self.next = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size= 3,padding=1),
            nn.PReLU(),
        )


        self.r = nn.PReLU()

        self.last = nn.Conv3d(16, 3, kernel_size=3, padding=1)


        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        # self.downsample = downsample
        # self.stride = stride

        self.upFrame = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=self.K, stride=S, padding=P, output_padding=OP),
            nn.PReLU(),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        a = 7
        # b, c, n, h, w = x.shape
        old = x
        # old = x[:, 1:2, :, :]
        ##Block1
        out = self.base(x)
        out = self.next(out)

        for i in range(a):
            out = self.next(out)
            out = self.next(out)
            out = self.ca(out) * out
            out = self.sa(out) * out

            out = self.r(out)

        out = self.upFrame(out)


        # b, c, n, h, w = x.shape
        # out_Ay = torch.zeros(b, c ,n,h,w).cuda()
        # for i in range(c):
        #     x = out[:, i, :, :, :]
        #     y = self.ca(x) * x
        #     Ay = self.sa(y) * y
        #     out_Ay[:, i, :, :, :] = Ay



        # if self.downsample is not None:
        #     residual = self.downsample(x)


        ##Block2
        # out = self.base1(out)
        #
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        #
        # out = self.r(out)


        # out = self.last(out)
        # out = out +old
        # out += residual
        # out = self.relu(out)

        return out
