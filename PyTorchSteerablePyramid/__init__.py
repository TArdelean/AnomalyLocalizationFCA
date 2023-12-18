import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))

from PyTorchSteerablePyramid.steerable.SCFpyr_PyTorch import SCFpyr_PyTorch


def extract_steerable_features(image, o=4, m=1, scale_factor=2):
    # image of shape 1 x 3 x H x W
    assert image.shape[0] == 1
    im_batch_torch = image.permute(1, 0, 2, 3)
    pyr = SCFpyr_PyTorch(height=m + 2, nbands=o, scale_factor=scale_factor, device=image.device)
    coeff = pyr.build(im_batch_torch)

    h, w = im_batch_torch.shape[-2:]
    real_features = [torch.stack(c)[..., 0] for c in coeff[1:-1]]
    real_features = [torch.nn.functional.interpolate(f, size=(h, w), mode='bilinear') for f in real_features]
    real_features = torch.cat(real_features, dim=0).reshape(image.shape[0], -1, h, w)

    return real_features
