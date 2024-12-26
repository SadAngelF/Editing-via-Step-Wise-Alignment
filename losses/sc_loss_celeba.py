import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as tvu
from resnet import Resnet18

from torch.nn import BatchNorm2d
import cv2
from PIL import Image
import numpy as np
import os 


def get_image(image_path, row, col, image_size=256, grid_width=1, device = 'cuda'):
    left_point = grid_width + (image_size + grid_width) * (col)
    up_point = grid_width + (image_size + grid_width) * (row)
    right_point = left_point + image_size
    down_point = up_point + image_size
    image = Image.open(image_path)
    croped_image = image.crop((left_point, up_point, right_point, down_point))
    # croped_image = transforms.ToTensor()(croped_image)
    # croped_image = croped_image.unsqueeze(0).to(device)
    return croped_image

def paste_image(output_image, image, row, col, image_size=256, grid_width=1):
    left_point = grid_width + (image_size + grid_width) * (col)
    up_point = grid_width + (image_size + grid_width) * (row)
    output_image.paste(image, (left_point, up_point))
    return output_image

def create_image(row, col, image_size=256, grid_width=1):
    image = Image.new("RGB", (image_size*(col)+grid_width*(col+1), image_size*(row)+grid_width*(row+1)), "white")
    return image


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im#, vis_parsing_anno

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


    # # Save result or not
    # if save_im:
    #     cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
    #     cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# class ConvBNReLU(nn.Module):

#     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(in_chan,
#                 out_chan,
#                 kernel_size = ks,
#                 stride = stride,
#                 padding = padding,
#                 bias = False)
#         self.bn = BatchNorm2d(out_chan)
#         self.relu = nn.ReLU(inplace=True)
#         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)


# class UpSample(nn.Module):

#     def __init__(self, n_chan, factor=2):
#         super(UpSample, self).__init__()
#         out_chan = n_chan * factor * factor
#         self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
#         self.up = nn.PixelShuffle(factor)
#         self.init_weight()

#     def forward(self, x):
#         feat = self.proj(x)
#         feat = self.up(feat)
#         return feat

#     def init_weight(self):
#         nn.init.xavier_normal_(self.proj.weight, gain=1.)


# class BiSeNetOutput(nn.Module):

#     def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
#         super(BiSeNetOutput, self).__init__()
#         self.up_factor = up_factor
#         out_chan = n_classes
#         self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
#         self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
#         self.up = nn.Upsample(scale_factor=up_factor,
#                 mode='bilinear', align_corners=False)
#         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.conv_out(x)
#         x = self.up(x)
#         return x

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, nn.modules.batchnorm._BatchNorm):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params


# class AttentionRefinementModule(nn.Module):
#     def __init__(self, in_chan, out_chan, *args, **kwargs):
#         super(AttentionRefinementModule, self).__init__()
#         self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
#         self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
#         self.bn_atten = BatchNorm2d(out_chan)
#         #  self.sigmoid_atten = nn.Sigmoid()
#         self.init_weight()

#     def forward(self, x):
#         feat = self.conv(x)
#         atten = torch.mean(feat, dim=(2, 3), keepdim=True)
#         atten = self.conv_atten(atten)
#         atten = self.bn_atten(atten)
#         #  atten = self.sigmoid_atten(atten)
#         atten = atten.sigmoid()
#         out = torch.mul(feat, atten)
#         return out

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)


# class ContextPath(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(ContextPath, self).__init__()
#         self.resnet = Resnet18()
#         self.arm16 = AttentionRefinementModule(256, 128)
#         self.arm32 = AttentionRefinementModule(512, 128)
#         self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#         self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#         self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
#         self.up32 = nn.Upsample(scale_factor=2.)
#         self.up16 = nn.Upsample(scale_factor=2.)

#         self.init_weight()

#     def forward(self, x):
#         feat8, feat16, feat32 = self.resnet(x)

#         avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
#         avg = self.conv_avg(avg)

#         feat32_arm = self.arm32(feat32)
#         feat32_sum = feat32_arm + avg
#         feat32_up = self.up32(feat32_sum)
#         feat32_up = self.conv_head32(feat32_up)

#         feat16_arm = self.arm16(feat16)
#         feat16_sum = feat16_arm + feat32_up
#         feat16_up = self.up16(feat16_sum)
#         feat16_up = self.conv_head16(feat16_up)

#         return feat16_up, feat32_up # x8, x16

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, nn.modules.batchnorm._BatchNorm):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params


# class SpatialPath(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(SpatialPath, self).__init__()
#         self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
#         self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
#         self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
#         self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
#         self.init_weight()

#     def forward(self, x):
#         feat = self.conv1(x)
#         feat = self.conv2(feat)
#         feat = self.conv3(feat)
#         feat = self.conv_out(feat)
#         return feat

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, nn.modules.batchnorm._BatchNorm):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params


# class FeatureFusionModule(nn.Module):
#     def __init__(self, in_chan, out_chan, *args, **kwargs):
#         super(FeatureFusionModule, self).__init__()
#         self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
#         ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
#         self.conv = nn.Conv2d(out_chan,
#                 out_chan,
#                 kernel_size = 1,
#                 stride = 1,
#                 padding = 0,
#                 bias = False)
#         self.bn = nn.BatchNorm2d(out_chan)
#         #  self.conv1 = nn.Conv2d(out_chan,
#         #          out_chan//4,
#         #          kernel_size = 1,
#         #          stride = 1,
#         #          padding = 0,
#         #          bias = False)
#         #  self.conv2 = nn.Conv2d(out_chan//4,
#         #          out_chan,
#         #          kernel_size = 1,
#         #          stride = 1,
#         #          padding = 0,
#         #          bias = False)
#         #  self.relu = nn.ReLU(inplace=True)
#         self.init_weight()

#     def forward(self, fsp, fcp):
#         fcat = torch.cat([fsp, fcp], dim=1)
#         feat = self.convblk(fcat)
#         atten = torch.mean(feat, dim=(2, 3), keepdim=True)
#         atten = self.conv(atten)
#         atten = self.bn(atten)
#         #  atten = self.conv1(atten)
#         #  atten = self.relu(atten)
#         #  atten = self.conv2(atten)
#         atten = atten.sigmoid()
#         feat_atten = torch.mul(feat, atten)
#         feat_out = feat_atten + feat
#         return feat_out

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     nowd_params.append(module.bias)
#             elif isinstance(module, nn.modules.batchnorm._BatchNorm):
#                 nowd_params += list(module.parameters())
#         return wd_params, nowd_params


# class BiSeNetV1(nn.Module):

#     def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
#         super(BiSeNetV1, self).__init__()
#         self.cp = ContextPath()
#         self.sp = SpatialPath()
#         self.ffm = FeatureFusionModule(256, 256)
#         self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
#         self.aux_mode = aux_mode
#         if self.aux_mode == 'train':
#             self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
#             self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
#         self.init_weight()

#     def forward(self, x):
#         H, W = x.size()[2:]
#         feat_cp8, feat_cp16 = self.cp(x)
#         feat_sp = self.sp(x)
#         feat_fuse = self.ffm(feat_sp, feat_cp8)

#         feat_out = self.conv_out(feat_fuse)
#         if self.aux_mode == 'train':
#             feat_out16 = self.conv_out16(feat_cp8)
#             feat_out32 = self.conv_out32(feat_cp16)
#             return feat_out, feat_out16, feat_out32
#         elif self.aux_mode == 'eval':
#             return feat_out,
#         elif self.aux_mode == 'pred':
#             feat_out = feat_out.argmax(dim=1)
#             return feat_out
#         else:
#             raise NotImplementedError

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
#         for name, child in self.named_children():
#             child_wd_params, child_nowd_params = child.get_params()
#             if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
#                 lr_mul_wd_params += child_wd_params
#                 lr_mul_nowd_params += child_nowd_params
#             else:
#                 wd_params += child_wd_params
#                 nowd_params += child_nowd_params
#         return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

# class BiSeNet(nn.Module):
#     def __init__(self, n_classes, *args, **kwargs):
#         super(BiSeNet, self).__init__()
#         self.cp = ContextPath()
#         ## here self.sp is deleted
#         self.ffm = FeatureFusionModule(256, 256)
#         self.conv_out = BiSeNetOutput(256, 256, n_classes)
#         self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
#         self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
#         self.init_weight()

#     def forward(self, x):
#         H, W = x.size()[2:]
#         feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
#         feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
#         feat_fuse = self.ffm(feat_sp, feat_cp8)

#         feat_out = self.conv_out(feat_fuse)
#         feat_out16 = self.conv_out16(feat_cp8)
#         feat_out32 = self.conv_out32(feat_cp16)

#         feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
#         feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
#         feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
#         return feat_out, feat_out16, feat_out32

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def get_params(self):
#         wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
#         for name, child in self.named_children():
#             child_wd_params, child_nowd_params = child.get_params()
#             if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
#                 lr_mul_wd_params += child_wd_params
#                 lr_mul_nowd_params += child_nowd_params
#             else:
#                 wd_params += child_wd_params
#                 nowd_params += child_nowd_params
#         return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



class SemanticConsistencyLoss(torch.nn.Module):

    def __init__(self, device='cuda', num_classes = 19,ckpt_path='./checkpoint/79999_iter.pth'):
        super(SemanticConsistencyLoss, self).__init__()

        self.device = device
        self.model = BiSeNet(num_classes).to(self.device)
        print('pretrained semantic segmentation model load')
        self.model.load_state_dict((torch.load(ckpt_path)))
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



        
    def forward(self, x1, x2):

        H, W = x1.size()[2], x1.size()[3]

        img1 = x1.to(self.device)
        img2 = x2.to(self.device)

        out1 = self.model(img1)[0]
        out2 = self.model(img2)[0]

        out1 = out1.cpu().detach().numpy().argmax(1)
        out2 = out2.cpu().detach().numpy().argmax(1)

        loss = (out1 == out2).sum((1,2))/(H * W)

        return loss


def run_church_department_store():

    device = 'cuda'

    ckpt_path = "./checkpoint/model_final_v1_city_new.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)
    
    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    save_path = 'vis_results/church_department_store'

    target_idx = [14]

    # church_department
    for i in range(32):

        if not i  in target_idx:
            continue

        ours_file_i = (i // 32) * 32 + 31
        ours_image_col = i % 32
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_js/runs_find_clip_loss_coeff/church/church_department_store/999_church_department_store_clip_loss_0.8_l1_loss_3_train_from_random_noise_LC_church_outdoor_t999_ninv40_ngen40/test_images/test_{ours_file_i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_church/output_department_ED_church_outdoor_t500_ninv40_ngen6_church_department_t500/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,ours_image_col)
        ours_image = get_image(ourspath, 1,ours_image_col)
        diffusionclip_image = Image.open(diffusionclippath)
        
        #save images
        result_image = create_image(1,3)

        result_image = paste_image(result_image, origin_image,0,0)
        result_image = paste_image(result_image, ours_image,0,1)
        result_image = paste_image(result_image, diffusionclip_image,0,2)

        result_image.save(f"{save_path}{i}.png")

        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0).to(device)
        ours_image = transforms.ToTensor()(ours_image).unsqueeze(0).to(device)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image).unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        

        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)



        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( "church_department_store", loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count


def run_church_wooden():

    device = 'cuda'

    ckpt_path = "./checkpoint/model_final_v1_city_new.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)

    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    save_path = 'vis_results/church_wooden'

    # church_department
    for i in range(32):
        ours_file_i = (i // 32) * 32 + 31
        ours_image_col = i % 32
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_js/runs_find_clip_loss_coeff/church/church_wooden_church/999_church_wooden_church_clip_loss_0.8_l1_loss_3_train_from_random_noise_LC_church_outdoor_t999_ninv40_ngen40/test_images/test_{ours_file_i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_church/output_wooden_ED_church_outdoor_t500_ninv40_ngen6_church_wooden_t500/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,ours_image_col)
        origin_image = get_image(ourspath, 0,ours_image_col)
        ours_image = get_image(ourspath, 1,ours_image_col)
        diffusionclip_image = Image.open(diffusionclippath)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image)
        diffusionclip_image = diffusionclip_image.unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        

        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)


        x_list = [origin_image, ours_image, diffusionclip_image]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        # tvu.save_image(grid, os.path.join(folder_dir, f'{file_name}_ngen{self.args.n_train_step}.png'), normalization=True)
        tvu.save_image(grid,os.path.join(save_path, f'{i}.png'), normalization=True)

        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( "church_wooden", loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count


def run_celeba_neanderthal():
    device = 'cuda'

    ckpt_path = "./checkpoint/face_parsing.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)
    
    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    exp_name = "celeba_neanderthal"

    import os

    save_path = os.path.join("vis_results", exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # target_idx = [14]

    for i in range(32):
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_mingi/Diffusion/runs_celeba_figure/neanderthal_clip_loss_1.2_l1_loss_3_train_from_precomputed_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/1000/test_{i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_celeba/neanderthal_ED_CelebA_HQ_t601_ninv40_ngen6_human_neanderthal_t601/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,0)
        ours_image = get_image(ourspath, 1,0)
        diffusionclip_image = Image.open(diffusionclippath)
        
        #save images
        result_image = create_image(1,3)

        result_image = paste_image(result_image, origin_image,0,0)
        result_image = paste_image(result_image, ours_image,0,1)
        result_image = paste_image(result_image, diffusionclip_image,0,2)

        result_image.save(f"{save_path}/{i}.png")

        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0).to(device)
        ours_image = transforms.ToTensor()(ours_image).unsqueeze(0).to(device)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image).unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)



        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( exp_name, loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count


def run_celeba_pixar():
    device = 'cuda'

    ckpt_path = "./checkpoint/face_parsing.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)
    
    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    exp_name = "celeba_pixar"

    import os

    save_path = os.path.join("vis_results", exp_name)

    # target_idx = [14]


    for i in range(32):

        # if not i  in target_idx:
        #     continue
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_mingi/Diffusion/runs_celeba_figure/pixar_clip_loss_0.8_l1_loss_3_train_from_random_noise_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/1000/test_{i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_celeba/pixar_ED_CelebA_HQ_t601_ninv40_ngen6_human_pixar_t601/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,0)
        ours_image = get_image(ourspath, 1,0)
        diffusionclip_image = Image.open(diffusionclippath)
        
        #save images
        result_image = create_image(1,3)

        result_image = paste_image(result_image, origin_image,0,0)
        result_image = paste_image(result_image, ours_image,0,1)
        result_image = paste_image(result_image, diffusionclip_image,0,2)

        result_image.save(f"{save_path}/{i}.png")

        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0).to(device)
        ours_image = transforms.ToTensor()(ours_image).unsqueeze(0).to(device)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image).unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        

        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)



        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( exp_name, loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count


def run_celeba_smiling():
    device = 'cuda'

    ckpt_path = "./checkpoint/face_parsing.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)
    
    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    exp_name = "celeba_smiling"

    import os

    save_path = os.path.join("vis_results", exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # target_idx = [14]

    for i in range(32):

        ours_file_i = (i // 4) * 4 + 3
        ours_image_col = i % 4
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_mingi/Diffusion/runs_celeba_figure/smiling_clip_loss_0.8_l1_loss_3_train_from_precomputed_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/1000/test_{ours_file_i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_celeba/smiling_ED_CelebA_HQ_t301_ninv40_ngen6_smiling_FT_CelebA_HQ_smiling_t301_ninv40_ngen6_id1.0_l11.0_lr8e-06_smiling_face-4/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,ours_image_col)
        ours_image = get_image(ourspath, 1,ours_image_col)
        diffusionclip_image = Image.open(diffusionclippath)
        
        #save images
        result_image = create_image(1,3)

        result_image = paste_image(result_image, origin_image,0,0)
        result_image = paste_image(result_image, ours_image,0,1)
        result_image = paste_image(result_image, diffusionclip_image,0,2)

        result_image.save(f"{save_path}/{i}.png")

        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0).to(device)
        ours_image = transforms.ToTensor()(ours_image).unsqueeze(0).to(device)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image).unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        

        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)



        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( exp_name, loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count


def run_celeba_sad():
    device = 'cuda'

    ckpt_path = "./checkpoint/face_parsing.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)
    
    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    exp_name = "celeba_sad"

    import os

    save_path = os.path.join("vis_results", exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # target_idx = [14]

    # church_department
    for i in range(32):


        ours_file_i = (i // 4) * 4 + 3
        ours_image_col = i % 4
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_mingi/Diffusion/runs_celeba_figure/sad_clip_loss_0.8_l1_loss_3_train_from_precomputed_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/1000/test_{ours_file_i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_celeba/sad_ED_CelebA_HQ_t301_ninv40_ngen6_sad_FT_CelebA_HQ_sad_t301_ninv40_ngen6_id1.0_l11.0_lr8e-06_sad_face-4/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,ours_image_col)
        ours_image = get_image(ourspath, 1,ours_image_col)
        diffusionclip_image = Image.open(diffusionclippath)
        
        #save images
        result_image = create_image(1,3)

        result_image = paste_image(result_image, origin_image,0,0)
        result_image = paste_image(result_image, ours_image,0,1)
        result_image = paste_image(result_image, diffusionclip_image,0,2)

        result_image.save(f"{save_path}/{i}.png")

        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0).to(device)
        ours_image = transforms.ToTensor()(ours_image).unsqueeze(0).to(device)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image).unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        


        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)



        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( exp_name, loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count


def run_celeba_tanned():
    device = 'cuda'

    ckpt_path = "./checkpoint/face_parsing.pth"
    
    sc_loss_fn = SemanticConsistencyLoss(ckpt_path=ckpt_path).to(device)
    
    
    image_count = 0
    loss_sc_ours = 0
    loss_sc_diffusionclip = 0

    exp_name = "celeba_tanned"

    import os

    save_path = os.path.join("vis_results", exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # target_idx = [14]

    for i in range(100):
        ours_file_i = i
        ours_image_col = 0
        ourspath = f"/home/mingi/Diffusion_Datacenter_experiments/Diffusion_from_datacenter_mingi/Diffusion/runs_celeba_figure/tanned_clip_loss_0.7_l1_loss_3_train_from_precomputed_LC_CelebA_HQ_t999_ninv40_ngen40/test_images/1000/test_{ours_file_i}_0_ngen40.png"
        diffusionclippath = f"/home/mingi/DiffusionCLIP_origin_not_ours/DiffusionCLIP_not_ours/runs_celeba/output_tanned_ED_CelebA_HQ_t201_ninv40_ngen6_human_tanned_t201/image_samples/test_{i}_2_clip_ngen40_mrat1.png"

        origin_image = get_image(ourspath, 0,ours_image_col)
        ours_image = get_image(ourspath, 1,ours_image_col)
        diffusionclip_image = Image.open(diffusionclippath)
        #save images
        result_image = create_image(1,3)

        result_image = paste_image(result_image, origin_image,0,0)
        result_image = paste_image(result_image, ours_image,0,1)
        result_image = paste_image(result_image, diffusionclip_image,0,2)

        result_image.save(f"{save_path}/{i}.png")

        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0).to(device)
        ours_image = transforms.ToTensor()(ours_image).unsqueeze(0).to(device)
        diffusionclip_image = transforms.ToTensor()(diffusionclip_image).unsqueeze(0).to(device)

        loss_sc_ours += sc_loss_fn(ours_image, origin_image)
        loss_sc_diffusionclip += sc_loss_fn(diffusionclip_image, origin_image)
        
        


        #visualize
        origin_seg = sc_loss_fn.model(origin_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        ours_seg = sc_loss_fn.model(ours_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()
        diffusionclip_seg = sc_loss_fn.model(diffusionclip_image)[0].cpu().detach().squeeze(0).permute(1,2,0).numpy()

        origin_seg_anno = np.argmax(origin_seg, axis=2)
        ours_seg_anno = np.argmax(ours_seg, axis=2)
        diffusionclip_seg_anno = np.argmax(diffusionclip_seg, axis=2)



        origin_seg_anno = vis_parsing_maps(origin_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), origin_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_origin.jpg')
        ours_seg_anno = vis_parsing_maps(ours_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), ours_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_ours.jpg')
        diffusionclip_seg_anno = vis_parsing_maps(diffusionclip_image.cpu().detach().squeeze(0).permute(1,2,0).numpy(), diffusionclip_seg_anno, 1)#, save_im=True, save_path=save_path+f'parsing_map_on_im{i}_diffusionclip.jpg')
        
        #numpy to tensor
        origin_seg_anno = transforms.ToTensor()(origin_seg_anno).unsqueeze(0)
        ours_seg_anno = transforms.ToTensor()(ours_seg_anno).unsqueeze(0)
        diffusionclip_seg_anno = transforms.ToTensor()(diffusionclip_seg_anno).unsqueeze(0)

        #make grid
        x_list = [origin_seg_anno, ours_seg_anno, diffusionclip_seg_anno]
        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=3, padding=1)

        tvu.save_image(grid, os.path.join(save_path, f'{i}_seg.png'), normalization=True)
        
        image_count += 1

    print( exp_name, loss_sc_ours / image_count, loss_sc_diffusionclip / image_count)
    # print(loss)
    print('done')

    return loss_sc_ours / image_count, loss_sc_diffusionclip / image_count    


if __name__ == '__main__':

    # mode ='church'
    mode = 'celeba'

    # run_celeba()
    ours_loss = 0
    diffusionclip_loss = 0

    if mode =="church":
    
        results = run_church_department_store()
        
        ours_loss += results[0]
        diffusionclip_loss += results[1]

        # results = run_church_wooden()

        ours_loss += results[0]
        diffusionclip_loss += results[1]

        print(ours_loss/2, diffusionclip_loss/2)
    
    elif mode =="celeba":
        # results = run_celeba_neanderthal()
        # ours_loss += results[0]
        # diffusionclip_loss += results[1]

        # results = run_celeba_pixar()
        # ours_loss += results[0]
        # diffusionclip_loss += results[1]

        # results = run_celeba_smiling()
        # ours_loss += results[0]
        # diffusionclip_loss += results[1]

        # results = run_celeba_sad()
        # ours_loss += results[0]
        # diffusionclip_loss += results[1]

        results =run_celeba_tanned()
        ours_loss += results[0]
        diffusionclip_loss += results[1]


        # print('avg: ',ours_loss/4, diffusionclip_loss/4)

