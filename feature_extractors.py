import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

import op_utils
from op_utils import scale_features, reflect_pad
from PyTorchSteerablePyramid import extract_steerable_features


class Colors:
    def __init__(self, post_scale=False):
        self.post_scale = post_scale

    def __call__(self, image: torch.tensor) -> torch.tensor:
        if self.post_scale:
            return scale_features(image)
        else:
            return image


class RandomKernels:
    def __init__(self, input_c=3, projections=((256, 1),), patch_size=7, device=None):
        self.ms_kernels = [self.build_kernels(input_c, num_proj, patch_size, device) for num_proj, _ in projections]
        self.scales = [scale for _, scale in projections]

    @staticmethod
    def build_kernels(c, num_proj, patch_size, device):
        kernels = torch.randn(num_proj, c * patch_size ** 2, device=device)
        kernels = kernels / torch.norm(kernels, dim=1, keepdim=True)
        kernels = kernels.reshape(num_proj, c, patch_size, patch_size)
        return kernels

    def __call__(self, image: torch.tensor) -> torch.tensor:
        parts = []
        for kernels, scale in zip(self.ms_kernels, self.scales):
            scaled_im = F.interpolate(image, scale_factor=scale, mode='bilinear')
            scaled_f = F.conv2d(reflect_pad(scaled_im, patch_size=kernels.shape[-1]), kernels)
            parts.append(F.interpolate(scaled_f, size=image.shape[-2:], mode='bilinear'))
        return torch.cat(parts, dim=1)


class NeuralExtractor:
    def __init__(self, post_scale=True, device=None):
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(-1, 1, 1)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(-1, 1, 1)
        self.post_scale = post_scale
        self.device = device

    def normalize(self, image: torch.tensor):
        return (image - self.normalization_mean) / self.normalization_std

    def post(self, features):
        if self.post_scale:
            return scale_features(features)
        return features


class VggExtractor(NeuralExtractor):
    def __init__(self, layers=('conv_1', 'conv_2', 'conv_3'), post_scale=True, device=None):
        super(VggExtractor, self).__init__(post_scale, device)
        self.cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.layers = layers

    def get_vgg_features(self, image: torch.tensor) -> torch.tensor:
        out = self.normalize(image)
        vgg_features = []
        i = 1
        for layer in self.cnn.children():
            out = layer(out)
            if isinstance(layer, torch.nn.Conv2d) and f'conv_{i}' in self.layers:
                vgg_features.append(out.clone().detach())
                i += 1

        return vgg_features

    @staticmethod
    def union_features(vgg_features):
        hw = vgg_features[0].shape[-2:]
        same_size = [F.interpolate(f, size=hw, mode='bilinear') for f in vgg_features]
        return torch.cat(same_size, dim=1)

    @torch.no_grad()
    def __call__(self, image: torch.tensor) -> torch.tensor:
        vgg_features = self.get_vgg_features(image)
        union = self.union_features(vgg_features)
        return self.post(union)


class WideResnetExtractor(NeuralExtractor):
    def __init__(self, post_scale=True, device=None):
        super(WideResnetExtractor, self).__init__(post_scale, device)
        cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
        self.model = torch.nn.Sequential(*list(cnn.children())[:6])

    @torch.no_grad()
    def __call__(self, image: torch.tensor) -> torch.tensor:
        features = self.model(self.normalize(image))
        return self.post(features)


class SteerableExtractor:
    def __init__(self, post_scale=True, o=4, m=1, scale_factor=2):
        self.post_scale = post_scale
        self.o = o
        self.m = m
        self.scale_factor = scale_factor

    def __call__(self, image: torch.tensor) -> torch.tensor:
        features = extract_steerable_features(image, self.o, self.m, self.scale_factor)
        if self.post_scale:
            features = scale_features(features)
        return features


class LawTextureEnergyMeasure:
    def __init__(self, mean_patch_size=15, device=None):
        self.mean_patch_size = mean_patch_size
        # noinspection SpellCheckingInspection
        lesr = torch.tensor([[1, 4, 6, 4, 1],
                             [-1, -2, 0, 2, 1],
                             [-1, 0, 2, 0, -1],
                             [1, -4, 6, -4, 1]], dtype=torch.float32, device=device)
        outers = torch.einsum('ni,mj->nmij', lesr, lesr)
        self.kernels = outers.reshape(outers.shape[0] * outers.shape[1], 1, *outers.shape[-2:])  # 16 x 1 x 5 x 5

    def __call__(self, image: torch.tensor) -> torch.tensor:
        image = torch.mean(image, dim=1, keepdim=True)  # Grayscale B x 1 x H x W
        if self.mean_patch_size != 0:
            image = image - op_utils.blur(image, self.mean_patch_size, sigma=20)
        energy_maps = F.conv2d(reflect_pad(image, patch_size=5), self.kernels) / 10
        features = torch.stack([
            energy_maps[:, 5],  # E5E5
            energy_maps[:, 10],  # S5S5
            energy_maps[:, 15],  # R5R5
            (energy_maps[:, 1] + energy_maps[:, 4]) / 2.0,  # L5E5 + E5L5
            (energy_maps[:, 2] + energy_maps[:, 8]) / 2.0,  # L5S5 + S5L5
            (energy_maps[:, 3] + energy_maps[:, 12]) / 2.0,  # L5R5 + R5L5
            (energy_maps[:, 6] + energy_maps[:, 9]) / 2.0,  # E5S5 + S5E5
            (energy_maps[:, 7] + energy_maps[:, 13]) / 2.0,  # E5R5 + R5E5
            (energy_maps[:, 11] + energy_maps[:, 14]) / 2.0,  # S5R5 + R5S5
        ], dim=1)
        return features
