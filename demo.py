import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from feature_extractors import WideResnetExtractor
from sc_methods import ScFCA


def main():
    device = torch.device('cuda')
    feature_extractor = WideResnetExtractor(device=device)
    s = ScFCA((9, 9), sigma_p=3.0, sigma_s=1.0)

    image = torch.tensor(np.asarray(Image.open("example/000.png")), device=device).permute(2, 0, 1)[None] / 255.0
    features = feature_extractor(image)
    a = s(features)

    pad = 10
    np_im = a.cpu().numpy()[pad:-pad, pad:-pad]
    np_im = (np_im - np_im.min()) / (np_im.max() - np_im.min())
    cma = cm.Reds(np_im, bytes=True)
    result = Image.fromarray(cma, 'RGBA').convert('RGB')
    result.save('example/000_out.png')
    result.show()


if __name__ == '__main__':
    main()
