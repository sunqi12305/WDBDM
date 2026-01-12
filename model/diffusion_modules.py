# WDBDM trainer class.
# This part builds heavily on https://github.com/arpitbansal297/Cold-Diffusion-Models.
import torch
from fontTools.subset import prune_hints
from torch import nn
import torchvision
import math
import cv2
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from utils.measure import *
from einops.layers.torch import Rearrange

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    YHH =  torch.cat((x_HL, x_LH, x_HH), 1)
    return x_LL, YHH


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    # print("x", x.shape)
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_ch):
        super(MultiScaleFeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.conv3 = nn.Conv2d(in_ch, in_ch // 2, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, in_ch // 2, 5, padding=2)
        self.fuse = nn.Conv2d(in_ch // 2 * 3, in_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        out = torch.cat([x1, x3, x5], dim=1)
        return self.fuse(out)

class Attenup(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attenup, self).__init__()
        # self.MSFF = MultiScaleFeatureFusion(in_ch=in_ch)
        self.deconv_layer = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1
        )
    def forward(self, xl, xh):
        x = torch.cat([xl, xh], dim=1)
        # x1 = self.MSFF(x)
        # x =  x + x1
        x = self.deconv_layer(x)

        return x
# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result
#
# class Attenup(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Attenup, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(in_ch, reduction=8)
#         self.pa = PixelAttention(in_ch)
#         # self.conv = nn.Conv2d(in_ch, in_ch, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#         # self.MSFF = MSFF(in_ch,in_ch)
#         self.deconv_layer = nn.ConvTranspose2d(
#             in_ch*2, out_ch, kernel_size=4, stride=2, padding=1
#         )
#     def forward(self, xl, xh):
#         x = torch.cat([xl, xh], dim=1)
#
#         cattn = self.ca(x)
#         sattn = self.sa(x)
#         pattn1 = sattn + cattn  #[4.4.256.256]
#         pattn2 = self.sigmoid(self.pa(x, pattn1)) #([4, 4, 256, 256]
#         result = torch.cat([x, pattn2], dim=1)
#         x = self.deconv_layer(result)
#
#         return x


class FrequencyWeighting(nn.Module):
    def __init__(self):
        super(FrequencyWeighting, self).__init__()
        # 初始化低频和高频的可学习权重参数
        self.alpha_low = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 低频权重
        self.alpha_high = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 高频权重

    def forward(self, low_freq, high_freq):
        # 将权重约束在 [0, 1] 范围内，防止权重过大或为负
        alpha_low = torch.sigmoid(self.alpha_low)
        alpha_high = torch.sigmoid(self.alpha_high)

        # 直接输出低频和高频的加权值
        weighted_low_freq = alpha_low * low_freq
        weighted_high_freq = alpha_high * high_freq

        return weighted_low_freq, weighted_high_freq, alpha_low, alpha_high


def extract(a, t, x_shape):
    # print("extract t ", t)
    b, *_ = t.shape
    # print("extract a", a)
    # print("extract a", a[1])
    out = a.gather(-1, t)
    # print("extract out", out)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_alpha_schedule(timesteps):
    steps = timesteps
    alphas_cumprod = 1 - torch.linspace(0, steps, steps) / timesteps
    return torch.clip(alphas_cumprod, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(self,
        denoise_fn1 = None,
        denoise_fn2 = None,
        image_size = 256, #512
        channels = 1,
        timesteps = 10,
        context=True,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = 256  #image_size
        self.denoise_fn1 = denoise_fn1
        self.denoise_fn2 = denoise_fn2
        # 定义自定义模块和反卷积层
        # self.down_wt = Down_wt(in_ch=1, out_ch=1).cuda()
        # self.deconv_layer = IWT().cuda()
        self.deconv_layer1 = Attenup(in_ch=4, out_ch=1).cuda()
        self.deconv_layer2 = Attenup(in_ch=4, out_ch=1).cuda()
        # self.w = FrequencyWeighting()
        # self.w1 = FrequencyWeighting()
        # self.deconv_layer2 = Attenup(in_ch=4, out_ch=1).cuda()
        self.dwt = DWT()
        # self.iwt1 = IWT()
        # self.iwt2 = IWT()
        self.num_timesteps = int(timesteps)
        self.context = context

        alphas_cumprod = linear_alpha_schedule(timesteps)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('one_minus_alphas_cumprod', 1. - alphas_cumprod)


    #  mean-preserving degradation operator
    def q_sample(self, x_start, x_end, t):
        # print(f"alphas_cumprod at t={t}: {extract(self.alphas_cumprod, t, x_start.shape)}")
        # print(f"one_minus_alphas_cumprod at t={t}: {extract(self.one_minus_alphas_cumprod, t, x_start.shape)}")
        # print(f"x_start: {x_start.mean()}, x_end: {x_end.mean()}")
        return (
                extract(self.alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )


    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    def transfer_calculate_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
        return img

    @torch.no_grad()
    def sample(self, batch_size=4,y=None, img=None, t=None, sampling_routine='ddim', n_iter=1, start_adjust_iter=1):
        self.denoise_fn1.eval()
        self.denoise_fn2.eval()
        self.deconv_layer1.eval()
        self.deconv_layer2.eval()
        # self.w.eval()
        t2 = t
        if t == None:
            t = self.num_timesteps
        if t2 == None:
            t2 = self.num_timesteps
        dwt = DWT()
        iwt = IWT()
        x = img
        yl, yh = dwt(y)
        if self.context:
            xl1 = img[:, 1].unsqueeze(1)
            xl0 = img[:, 0].unsqueeze(1)
            xl2 = img[:, 2].unsqueeze(1)
            img, xh11 = dwt(xl1)
            up_img, xh00 = dwt(xl0)
            down_img, xh22 = dwt(xl2)
            # up_img = img[:, 0].unsqueeze(1)
            # down_img = img[:, 2].unsqueeze(1)
            # img = img[:, 1].unsqueeze(1)

        noise = img
        x1_bar = img
        direct_recons = []
        imstep_imgs = []
        if sampling_routine == 'x0_step_down':
            # print("############run x0_step_down#################")
            while (t):
                # print("****************t*****************",t)
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)

                if self.context:
                    full_img = torch.cat((up_img, img, down_img), dim=1)
                else:
                    full_img = img

                if t == self.num_timesteps:
                    adjust = False
                else:
                    adjust = True
                # print("############adjust#########33", adjust)
                # x1_bar = self.denoise_fn1(full_img, step, x1_bar, noise, adjust=adjust)
                x1_bar = self.denoise_fn1(full_img, step, x1_bar, xh11, adjust=adjust)

                x2_bar = noise

                xt_bar = x1_bar
                if t != 0:
                    # print(" if t != 0:")
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    # print("if t - 1 != 0:")
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                img = img - xt_bar + xt_sub1_bar

                direct_recons.append(x1_bar)
                imstep_imgs.append(img)
                # print("****************t*****************", t)
                t = t - 1

        if self.context:
            xl1 = x[:, 1].unsqueeze(1)
            xl0 = x[:, 0].unsqueeze(1)
            xl2 = x[:, 2].unsqueeze(1)
            img1, img2 = dwt(xl1)
            up_img1, up_img = dwt(xl0)
            down_img1, down_img = dwt(xl2)
            # up_img = img[:, 0].unsqueeze(1)
            # down_img = img[:, 2].unsqueeze(1)
            # img = img[:, 1].unsqueeze(1)
        noise = img2
        x1_bar = img2
        direct_recons = []
        imstep_imgs = []
        if sampling_routine == 'x0_step_down':
            # print("############run x0_step_down#################")
            while (t2):
                # print("****************t*****************",t2)
                step = torch.full((batch_size,), t2 - 1, dtype=torch.long, device=img2.device)

                if self.context:
                    full_img = torch.cat((up_img, img2, down_img), dim=1)
                else:
                    full_img = img2

                if t2 == self.num_timesteps:
                    adjust = False
                else:
                    adjust = True
                # print("############adjust#########33", adjust)
                # x1_bar = self.denoise_fn2(full_img, step, x1_bar, noise, adjust=adjust)
                x1_bar = self.denoise_fn2(full_img, step, x1_bar, img1, adjust=adjust)

                x2_bar = noise

                xt_bar = x1_bar
                if t2 != 0:
                    # print(" if t != 0:")
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t2 - 1 != 0:
                    # print("if t - 1 != 0:")
                    step2 = torch.full((batch_size,), t2 - 2, dtype=torch.long, device=img2.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)
                # output = img.detach().cpu().numpy()[0, 0]  # 转为 numpy 格式
                # output_normalized = (output - output.min()) / (output.max() - output.min())  # 归一化
                # output_normalized = (output_normalized * 255).astype(np.uint8)  # 转换为 0-255 的范围
                #
                # # 使用 OpenCV 显示图像
                # cv2.imshow("Output", output_normalized)
                # cv2.waitKey(1)  # 延迟 1ms 以便实时更新
                img2 = img2 - xt_bar + xt_sub1_bar
                direct_recons.append(x1_bar)
                # imstep_imgs.append(img)
                # print("****************t*****************", t2)
                t2 = t2 - 1

            # output = torch.cat([img, img2], dim=1)
            # output = iwt(output)
            # weighted_low, weighted_high, alpha_low, alpha_high = self.w(img, img2)
            # output = iwt(torch.cat([weighted_low, weighted_high], dim=1))
            # print(f"Learned low frequency weight (alpha_low): {alpha_low.item()}")
            # print(f"Learned high frequency weight (alpha_high): {alpha_high.item()}")
            # output = torch.cat([img, img2], dim=1)
            # output = iwt(output)
            output = self.deconv_layer2(img, img2)
            direct_recons.append(x1_bar)
            imstep_imgs.append(img)
        # print("img, img2, yl, yh", img.shape, img2.shape, yl.shape, yh.shape)
        return output.clamp(0., 1.), img, img2, yl, yh


    def forward(self, x, y, n_iter, only_adjust_two_step=False, start_adjust_iter=1):
        '''
        :param x: low dose image
        :param y: ground truth image
        :param n_iter: trainging iteration
        :param only_adjust_two_step: only use the EMM module in the second stage. Default: True
        :param start_adjust_iter: the number of iterations to start training the EMM module. Default: 1
        '''
        dwt = DWT()
        xl1 = x[:, 1].unsqueeze(1)
        xl0 = x[:, 0].unsqueeze(1)
        xl2 = x[:, 2].unsqueeze(1)
        xl11, xh11 = dwt(xl1)
        xl00, xh00 = dwt(xl0)
        xl22, xh22 = dwt(xl2)
        ll_low_dose = torch.cat((xl00, xl11, xl22), dim=1)  # [1, 3, 512, 512]
        # print("ll_low_dose", ll_low_dose.shape)
        lh_low_dose = torch.cat((xh00, xh11, xh22), dim=1)  # ([1, 9, 256, 256]
        # print("lh_low_dose", lh_low_dose.shape)
        yl, yh = dwt(y)
        # print("yh", yh.shape) #[1, 3, 256, 256]

        b, c, h, w, device, img_size, = *y.shape, y.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()

        t = t_single.repeat((b,))

        #ll
        # print("lllllll###########################t",t)
        if self.context:
            x_end = ll_low_dose[:,1].unsqueeze(1)#[2, 1, 512, 512] [0 1 2] middle
            x_mix = self.q_sample(x_start=yl, x_end=x_end, t=t)
            x_mix = torch.cat((ll_low_dose[:,0]. unsqueeze(1), x_mix, ll_low_dose[:,2].unsqueeze(1)), dim=1)
        else:
            x_end = ll_low_dose
            x_mix = self.q_sample(x_start=yl, x_end=x_end, t=t)

        # stage I
        if only_adjust_two_step or n_iter < start_adjust_iter:#start_adjust_iter=1
            # print("run  if")
            x_recon = self.denoise_fn1(x_mix, t, yl, x_end, adjust=False)#[1, 1, 256, 256])
            # print("x_recon", x_recon.shape)
        else:
            # print("run else")
            if t[0] == self.num_timesteps - 1:
                adjust = False
            else:
                adjust = True
            x_recon = self.denoise_fn1(x_mix, t, yl, x_end, adjust=adjust)

        # stage II
        if n_iter >= start_adjust_iter and t_single.item() >= 1:
            t_sub1 = t - 1
            t_sub1[t_sub1 < 0] = 0

            if self.context:
                x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)
                x_mix_sub1 = torch.cat((ll_low_dose[:, 0].unsqueeze(1), x_mix_sub1, ll_low_dose[:, 2].unsqueeze(1)), dim=1)
            else:
                x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)

            # x_recon_sub1 = self.denoise_fn1(x_mix_sub1, t_sub1, x_recon, x_end, adjust=True)
            x_recon_sub1 = self.denoise_fn1(x_mix_sub1, t_sub1, x_recon, xh11, adjust=True)
        else:
            # print("sun stage2 else")
            x_recon_sub1, x_mix_sub1 = x_recon, x_mix

        #hh
        # print("hhhhhhhhhhhhh###########################t", t)
        if self.context:
            # x_end_h = lh_low_dose[:, 1].unsqueeze(1)  # [2, 1, 512, 512] [0 1 2] middle
            x_end_h =  lh_low_dose[:, 3:6, :, :]
            # print("x_end_h", x_end_h.shape)
            x_mix_h = self.q_sample(x_start=yh, x_end=x_end_h, t=t)
            x_mix_h = torch.cat((lh_low_dose[:, :3, :, :], x_mix_h, lh_low_dose[:, 6:9, :, :]), dim=1)#[1, 9, 256, 256]
            # print("x_mix_h", x_mix_h.shape)
        else:
            x_end_h = lh_low_dose
            x_mix_h = self.q_sample(x_start=yh, x_end=x_end_h, t=t)

        # stage I
        if only_adjust_two_step or n_iter < start_adjust_iter:  # start_adjust_iter=1
            # print("run  if")
            x_reconh_h = self.denoise_fn2(x_mix_h, t, yh, x_end_h, adjust=False)
            # print("x_reconh_h", x_reconh_h.shape)
        else:
            # print("run else")
            if t[0] == self.num_timesteps - 1:
                adjust = False
            else:
                adjust = True
            x_reconh_h = self.denoise_fn2(x_mix_h, t, yh, x_end_h, adjust=adjust)

        # stage II
        if n_iter >= start_adjust_iter and t_single.item() >= 1:
            t_sub1_h = t - 1
            t_sub1_h[t_sub1_h < 0] = 0

            if self.context:
                x_mix_sub1_h = self.q_sample(x_start=x_reconh_h, x_end=x_end_h, t=t_sub1_h)
                # print("x_mix_sub1_h", x_mix_sub1_h.shape)
                x_mix_sub1_h = torch.cat((lh_low_dose[:, :3, :, :], x_mix_sub1_h, lh_low_dose[:, 6:9, :, :]),
                                         dim=1)
            else:
                x_mix_sub1_h = self.q_sample(x_start=x_reconh_h, x_end=x_end_h, t=t_sub1_h)

            # x_recon_sub1_h = self.denoise_fn2(x_mix_sub1_h, t_sub1_h, x_reconh_h, x_end_h, adjust=True)
            x_recon_sub1_h = self.denoise_fn2(x_mix_sub1_h, t_sub1_h, x_reconh_h, xl11, adjust=True)
        else:
            # print("sun stage2 else")
            x_recon_sub1_h, x_mix_sub1_h = x_reconh_h, x_mix_h

        # iwt = IWT()
        # output1 = torch.cat([x_recon, x_reconh_h], dim=1)
        # output2 = torch.cat([x_recon_sub1, x_recon_sub1_h], dim=1)
        # output1 = iwt(output1)
        # output2 = iwt(output2)
        output1 = self.deconv_layer1(x_recon, x_reconh_h)
        output2 = self.deconv_layer2(x_recon_sub1, x_recon_sub1_h)

        return x_recon, x_recon_sub1, x_reconh_h, x_recon_sub1_h, output1, output2, yl, yh