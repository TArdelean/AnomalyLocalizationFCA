import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np

def scale_features(tensor: torch.tensor) -> torch.tensor:
    # tensor shape B x C x H x W
    tf = tensor.flatten(start_dim=2)
    mini = tf.min(dim=-1).values[..., None, None]
    maxi = tf.max(dim=-1).values[..., None, None]
    div = (maxi - mini + 1e-8)
    return (tensor - mini) / div

# noinspection PyProtectedMember,PyUnresolvedReferences
def get_gaussian_kernel(device, tile_size, s: float):
    return torchvision.transforms._functional_tensor._get_gaussian_kernel2d(tile_size, [s, s], torch.float32, device)

def blur(image, kernel_size=7, sigma=None):
    if sigma is None:
        sigma = kernel_size / 4
    shape = image.shape
    im_b = image[(None,) * (4 - len(shape))]
    # noinspection PyUnresolvedReferences
    return torchvision.transforms.functional.gaussian_blur(im_b, kernel_size, sigma=sigma).view(shape)


def load_image_tensor(path, image_size, device=None):
    input_image = cv2.imread(str(path))
    if image_size == "original":
        image_size = input_image.shape[:2]
    else:
        image_size = tuple(image_size)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    return F.interpolate(image_to_tensor(input_image, device=device), size=image_size, mode='bilinear')


def image_to_tensor(image, device):
    image = torch.tensor(image, device=device, dtype=torch.float32)
    image = image.permute(2, 0, 1)[None] / 255
    return image


def tensor_to_image(image: torch.tensor):
    out = image.detach().cpu()
    out = (out.squeeze(0).permute(1, 2, 0) * 255)
    return out.numpy().astype(np.uint8)


def reflect_pad(image, patch_size=7):
    p = patch_size // 2
    return torch.nn.functional.pad(image, (p, p, p, p), mode='reflect')
