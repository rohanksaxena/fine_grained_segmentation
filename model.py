import torch
import torch.nn as nn

from torchvision.models import vgg16, VGG16_Weights
from lib.ssn.ssn import ssn_iter, sparse_ssn_iter


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class SSNModel(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            conv_bn_relu(5, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64 * 3 + 5, feature_dim - 5, 3, padding=1),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pixel_f = self.feature_extract(x)

        if self.training:
            return *ssn_iter(pixel_f, self.nspix, self.n_iter), pixel_f
        else:
            return *sparse_ssn_iter(pixel_f, self.nspix, self.n_iter), pixel_f

    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)


class SSN_VGG(nn.Module):
    def __init__(self, layer_number, nspix, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.layer_number = layer_number
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.relu_layer_indices = (1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29)
        self.pooling_layer_indices = (4, 9, 16, 23, 30)

    def forward(self, x):

        input = x
        output = {}
        # Use VGG backbone for feature extraction
        num_children_modules = len(list(self.features.children()))
        for i in range(self.layer_number + 1):
            x = self.features[i](x)
            if i in self.relu_layer_indices:
                output[f'layer_{i}'] = x

        for key, value in output.items():
            output[key] = nn.functional.interpolate(value, size=input.shape[-2:], mode='bilinear', align_corners=False)

        pixel_f = torch.cat([*output.values(), input], 1)
        # pixel_f = self.features(x)

        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)
