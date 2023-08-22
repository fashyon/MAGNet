import torch
from torch import nn
import torch.nn.functional as F
import settings
import numpy as np
from math import log


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    ZeroPad2d = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    img = ZeroPad2d(input_data)
    col = torch.zeros([N, C, filter_h,filter_w, out_h,out_w]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride,]

    col = col.reshape(N, C, filter_h*filter_w, out_h*out_w)

    return col


def col2im(col, orisize, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = orisize
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, C, filter_h, filter_w,out_h, out_w)
    img = torch.zeros([N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1]).cuda()
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class NLWT(nn.Module):
    def __init__(self):
        super(NLWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return NLWT_init(x)

def NLWT_init(x):
    U1ImulP1I = np.array([[-0.5000,   -0.5000,   -0.5000,    0.5000],
                          [0.5000,   -0.5000,    0.5000,    0.5000],
                          [-0.5000,   -0.5000,    0.5000,   -0.5000],
                          [-0.5000,    0.5000,    0.5000,    0.5000]])
    U2ImulP2Imul4 = np.array([[-1,    1,    -1,     1],
                              [-1,    -1,    1,     1],
                              [1,     1,     1,     1],
                              [-1,    1,     1,    -1]])
    U1ImulP1I = torch.cuda.FloatTensor(U1ImulP1I).unsqueeze(0).unsqueeze(0)
    U2ImulP2Imul4 = torch.cuda.FloatTensor(U2ImulP2Imul4).unsqueeze(0).unsqueeze(0)

    b, c, h, w = x.size()
    orisize = x.size()

    xT_col = im2col(x, 2, 2, stride=2, pad=0);
    x1 = U1ImulP1I @ xT_col;

    h1 = h // 2
    w1 = w // 2
    T2 = x1[:, :, 1, :].reshape(b, c, h1, w1);
    T3 = x1[:, :, 2, :].reshape(b, c, h1, w1);
    T4 = x1[:, :, 3, :].reshape(b, c, h1, w1);

    T22 = torch.roll(T2, shifts=-1, dims=2)
    T32 = torch.roll(T3, shifts=-1, dims=3)
    T42 = torch.roll(T4, shifts=(-1, -1), dims=(2, 3))

    x1[:, :, 1, :] = T22.flatten(2);
    x1[:, :, 2, :] = T32.flatten(2);
    x1[:, :, 3, :] = T42.flatten(2);

    x2 = U2ImulP2Imul4 @ x1;

    A_low0 = x2[:, :, 0, :].reshape(b, c, h1, w1);
    B_high1 = x2[:, :, 1, :].reshape(b, c, h1, w1);
    C_high2 = x2[:, :, 2, :].reshape(b, c, h1, w1);
    D_high3 = x2[:, :, 3, :].reshape(b, c, h1, w1);

    return A_low0, B_high1, C_high2, D_high3, orisize

class INLWT(nn.Module):
    def __init__(self):
        super(INLWT, self).__init__()
        self.requires_grad = False

    def forward(self, A_low0, B_high1, C_high2, D_high3,orisize):
        return INLWT_init(A_low0, B_high1, C_high2, D_high3, orisize)

def INLWT_init(A_low0,B_high1,C_high2,D_high3,orisize):
    P2mulU2 = np.array([[-1,   -1,     1,    -1],
                        [1,    -1,     1,     1],
                        [-1,    1,     1,     1],
                        [1,     1,     1,    -1]])
    P1mulU1div4 = np.array([[-0.1250,    0.1250,   -0.1250,   -0.1250],
                            [-0.1250,   -0.1250,   -0.1250,    0.1250],
                            [-0.1250,    0.1250,    0.1250,    0.1250],
                            [0.1250,     0.1250,   -0.1250,    0.1250]])
    P2mulU2 = torch.cuda.FloatTensor(P2mulU2).unsqueeze(0).unsqueeze(0)
    P1mulU1div4 = torch.cuda.FloatTensor(P1mulU1div4).unsqueeze(0).unsqueeze(0)

    b, c, h1, w1 = A_low0.size()

    A = A_low0.reshape(b, c, 1, h1 * w1);
    B = B_high1.reshape(b, c, 1, h1 * w1);
    C = C_high2.reshape(b, c, 1, h1 * w1);
    D = D_high3.reshape(b, c, 1, h1 * w1);

    Y1 = torch.cat([A, B, C, D], dim=2)
    Y2 = P2mulU2 @ Y1;
    t2 = Y2[:, :, 1, :].reshape(b, c, h1, w1);
    t3 = Y2[:, :, 2, :].reshape(b, c, h1, w1);
    t4 = Y2[:, :, 3, :].reshape(b, c, h1, w1);

    t22 = torch.roll(t2, shifts=1, dims=2)
    t32 = torch.roll(t3, shifts=1, dims=3)
    t42 = torch.roll(t4, shifts=(1, 1), dims=(2, 3))

    Y2[:, :, 1, :] = t22.flatten(2)
    Y2[:, :, 2, :] = t32.flatten(2)
    Y2[:, :, 3, :] = t42.flatten(2)

    Y3 = P1mulU1div4 @ Y2;
    rst = col2im(Y3, orisize, 2, 2, stride=2, pad=0);

    return rst

class NLWT_CatOne(nn.Module):
    def __init__(self):
        super(NLWT_CatOne, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        A_low0, B_high1, C_high2, D_high3, orisize = NLWT_init(x)
        out_catone = torch.cat([A_low0, B_high1, C_high2, D_high3], dim=1)
        return out_catone, orisize

class INLWT_CatOne(nn.Module):
    def __init__(self):
        super(INLWT_CatOne, self).__init__()
        self.requires_grad = False

    def forward(self, decoder_one,orisize):
        out_channel = orisize[1]
        A_low0 = decoder_one[:, 0:out_channel, :, :]
        B_high1 = decoder_one[:, out_channel:out_channel * 2, :, :]
        C_high2 = decoder_one[:, out_channel * 2:out_channel * 3, :, :]
        D_high3 = decoder_one[:, out_channel * 3:out_channel * 4, :, :]
        rst = INLWT_init(A_low0, B_high1, C_high2, D_high3, orisize)
        return rst


class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Sequential(nn.Conv2d(inputchannel, outchannel, kernel_size, stride), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(self.padding(x))
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class ECA(nn.Module):
    def __init__(self, channel,gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

#Scale_guide_Progressive_Fusion_Module
class SPFM(nn.Module):
    def __init__(self, channel):
        super(SPFM, self).__init__()
        self.channel = channel
        self.scale1_block1 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                           ECA(self.channel),
                                           nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.scale2_block1 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.scale2_block2 = nn.Sequential(ECA(self.channel),
                                           nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.scale3_block1 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.scale3_block2 = nn.Sequential(ECA(self.channel),
                                           nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.scale1_downfuse1 = convd(self.channel * 4, self.channel, 3, 1)
        self.scale1_downfuse2 = convd(self.channel * 4, self.channel, 3, 1)
        self.scale2_downfuse1 = convd(self.channel * 4, self.channel, 3, 1)
        self.scale2_downfuse2 = convd(self.channel * 4, self.channel, 3, 1)
        self.scale1_convfuse1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.scale2_convfuse1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.scale2_convfuse2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.scale3_convfuse1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.ifuse1 = convd(self.channel, self.channel * 4, 3, 1)
        self.ifuse2 = convd(self.channel, self.channel * 4, 3, 1)
        self.CBAM = CBAM(self.channel)
        self.res = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.NLWT_CatOne = NLWT_CatOne()
        self.INLWT_CatOne = INLWT_CatOne()

    def forward(self, x):
        # Scale-guide process
        x_scale1 = x
        x_scale1_NLWT,scale1_size = self.NLWT_CatOne(x_scale1)
        x_scale2 = self.scale1_downfuse1(x_scale1_NLWT)
        x_scale2_NLWT,scale2_size = self.NLWT_CatOne(x_scale2)
        x_scale3 = self.scale2_downfuse1(x_scale2_NLWT)
        x_scale1 = self.scale1_block1(x_scale1)
        x_scale2 = self.scale2_block1(x_scale2)
        x_scale3 = self.scale3_block1(x_scale3)
        x_scale1_block1_down = self.scale1_downfuse2(self.NLWT_CatOne(x_scale1)[0])
        x_scale2_guide = self.scale2_block2(self.scale2_convfuse1(torch.cat([x_scale1_block1_down, x_scale2], dim=1)))
        x_scale2_guide_down = self.scale2_downfuse2(self.NLWT_CatOne(x_scale2_guide)[0])
        x_scale3_guide = self.scale3_block2(self.scale3_convfuse1(torch.cat([x_scale2_guide_down, x_scale3], dim=1)))

        # Progressive fusion process
        x_scale3_guide_up = self.INLWT_CatOne(self.ifuse1(x_scale3_guide), scale2_size)
        x_scale2_guide = self.scale2_convfuse2(torch.cat([x_scale3_guide_up, x_scale2_guide], dim=1))
        x_scale2_guide_up = self.INLWT_CatOne(self.ifuse2(x_scale2_guide), scale1_size)
        x_scale1 = self.scale1_convfuse1(torch.cat([x_scale2_guide_up, x_scale1], dim=1))
        x_scale1 = self.CBAM(x_scale1)
        res = self.res(x_scale1)
        output = res + x
        return output

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return y

class GFM_2(nn.Module):
    def __init__(self,channel):
        super(GFM_2, self).__init__()
        self.channel = channel
        self.ca = CALayer(2*self.channel)

    def forward(self,x1,x2):
        w = self.ca(torch.cat([x1, x2], dim=1))
        w = w.view(-1, 2, self.channel)[:,:,:,None,None]
        out = w[:, 0, ::] * x1 + w[:, 1, ::] * x2
        return out

class GFM_3(nn.Module):
    def __init__(self,channel):
        super(GFM_3, self).__init__()
        self.channel = channel
        self.ca = CALayer(3*self.channel)

    def forward(self,x1,x2,x3):
        w = self.ca(torch.cat([x1, x2, x3], dim=1))
        w = w.view(-1, 3, self.channel)[:, :, :, None, None]
        out = w[:, 0, ::] * x1 + w[:, 1, ::] * x2 + w[:, 2, ::] * x3
        return out

class Multi_Aggregation_Network(nn.Module):
    def __init__(self, in_channel=3, mid_channel=settings.channel, out_channel=3):
        super(Multi_Aggregation_Network, self).__init__()
        self.channel = mid_channel
        self.convert = nn.Sequential(nn.Conv2d(in_channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.encoder1_level0 = SPFM(self.channel)
        self.encoder2_level0 = SPFM(self.channel)
        self.encoder1_level1 = SPFM(self.channel)
        self.encoder2_level1 = SPFM(self.channel)
        self.encoder1_level2 = SPFM(self.channel)
        self.encoder2_level2 = SPFM(self.channel)
        self.encoder1_level3 = SPFM(self.channel)
        self.decoder1_level3 = SPFM(self.channel)
        self.decoder1_level2 = SPFM(self.channel)
        self.decoder2_level2 = SPFM(self.channel)
        self.decoder1_level1 = SPFM(self.channel)
        self.decoder2_level1 = SPFM(self.channel)
        self.decoder1_level0 = SPFM(self.channel)
        self.decoder2_level0 = SPFM(self.channel)
        self.GFM1_level0 = GFM_3(self.channel)
        self.GFM2_level0 = GFM_3(self.channel)
        self.GFM1_level1 = GFM_2(self.channel)
        self.GFM2_level1 = GFM_2(self.channel)
        self.GFM3_level1 = GFM_3(self.channel)
        self.GFM4_level1 = GFM_3(self.channel)
        self.GFM1_level2 = GFM_2(self.channel)
        self.GFM2_level2 = GFM_2(self.channel)
        self.GFM3_level2 = GFM_3(self.channel)
        self.GFM4_level2 = GFM_3(self.channel)
        self.GFM1_level3 = GFM_2(self.channel)
        self.GFM2_level3 = GFM_2(self.channel)
        self.fuse1 = convd(self.channel * 4, self.channel, 3, 1)
        self.fuse2 = convd(self.channel * 4, self.channel, 3, 1)
        self.fuse3 = convd(self.channel * 4, self.channel, 3, 1)
        self.fuse4 = convd(self.channel * 4, self.channel, 3, 1)
        self.fuse5 = convd(self.channel * 4, self.channel, 3, 1)
        self.fuse6 = convd(self.channel * 4, self.channel, 3, 1)
        self.ifuse1 = convd(self.channel, self.channel * 4, 3, 1)
        self.ifuse2 = convd(self.channel, self.channel * 4, 3, 1)
        self.ifuse3 = convd(self.channel, self.channel * 4, 3, 1)
        self.ifuse4 = convd(self.channel, self.channel * 4, 3, 1)
        self.ifuse5 = convd(self.channel, self.channel * 4, 3, 1)
        self.ifuse6 = convd(self.channel, self.channel * 4, 3, 1)
        self.NLWT_CatOne = NLWT_CatOne()
        self.INLWT_CatOne = INLWT_CatOne()
        self.out = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(self.channel, out_channel, 1, 1))

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 32
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]

    def forward(self, x):
        x_check, ori_size = self.check_image_size(x)

        # Encoder
        convert = self.convert(x_check)
        en1_le0 = self.encoder1_level0(convert)
        en2_le0 = self.encoder2_level0(en1_le0)
        en1_le0_NLWT, level0_size = self.NLWT_CatOne(en1_le0)
        en1_le0_down = self.fuse1(en1_le0_NLWT)
        en2_le0_down = self.fuse2(self.NLWT_CatOne(en2_le0)[0])

        en1_le1 = self.encoder1_level1(self.GFM1_level1(en1_le0_down, en2_le0_down))
        en2_le1 = self.encoder2_level1(self.GFM2_level1(en1_le1, en2_le0_down))
        en1_le1_NLWT, level1_size = self.NLWT_CatOne(en1_le1)
        en1_le1_down = self.fuse3(en1_le1_NLWT)
        en2_le1_down = self.fuse4(self.NLWT_CatOne(en2_le1)[0])

        en1_le2 = self.encoder1_level2(self.GFM1_level2(en1_le1_down, en2_le1_down))
        en2_le2 = self.encoder2_level2(self.GFM2_level2(en1_le2, en2_le1_down))
        en1_le2_NLWT, level2_size = self.NLWT_CatOne(en1_le2)
        en1_le2_down = self.fuse5(en1_le2_NLWT)
        en2_le2_down = self.fuse6(self.NLWT_CatOne(en2_le2)[0])
        en1_le3 = self.encoder1_level3(self.GFM1_level3(en1_le2_down, en2_le2_down))

        # Decoder
        de1_le3 = self.decoder1_level3(self.GFM2_level3(en1_le3, en2_le2_down))
        en1_le3_up = self.INLWT_CatOne(self.ifuse1(en1_le3),level2_size)
        de1_le3_up = self.INLWT_CatOne(self.ifuse2(de1_le3),level2_size)

        de1_le2 = self.decoder1_level2(self.GFM3_level2(en1_le3_up, de1_le3_up, en2_le2))
        de2_le2 = self.decoder2_level2(self.GFM4_level2(de1_le2, de1_le3_up, en1_le2))
        de1_le2_up = self.INLWT_CatOne(self.ifuse3(de1_le2),level1_size)
        de2_le2_up = self.INLWT_CatOne(self.ifuse4(de2_le2),level1_size)
        
        de1_le1 = self.decoder1_level1(self.GFM3_level1(de1_le2_up, de2_le2_up, en2_le1))
        de2_le1 = self.decoder2_level1(self.GFM4_level1(de1_le1, de2_le2_up, en1_le1))
        de1_le1_up = self.INLWT_CatOne(self.ifuse5(de1_le1),level0_size)
        de2_le1_up = self.INLWT_CatOne(self.ifuse6(de2_le1),level0_size)

        de1_le0 = self.decoder1_level0(self.GFM1_level0(de1_le1_up, de2_le1_up, en2_le0))
        de2_le0 = self.decoder2_level0(self.GFM2_level0(de1_le0, de2_le1_up, en1_le0))
        y = self.restore_image_size(self.out(de2_le0), ori_size)
        out = x - y
        return out