import torch
import torch.nn.functional as F
import torchvision

from op_utils import blur, scale_features, reflect_pad, get_gaussian_kernel


def sc_moments(features, tile_size, powers=(1, 2, 3, 4), sigma=6.0):
    assert features.shape[0] == 1  # Only implemented for one image, but easy to extend...
    ps = features.new_tensor(list(powers))
    b, c, h, w = features.shape
    moments = torch.pow(features[:, :, None, :, :], ps[None, None, :, None, None]).reshape(b, c * len(ps), h, w)
    local_moments = blur(moments, kernel_size=tile_size[0], sigma=sigma)[0]  # C*powers x H x W

    def dist(one, many):
        vec_arr = one[:, None, None].expand_as(many)
        loss = torch.mean((vec_arr - many) ** 2, dim=0)
        del vec_arr
        return loss

    representative = torch.mean(local_moments.flatten(start_dim=-2), dim=-1)  # C*powers
    return dist(representative, local_moments)


def sc_hist(features, tile_size, bins=10, sigma=6.0):
    assert features.shape[0] == 1  # Only implemented for one image, but easy to extend...
    scaled = scale_features(features)
    indices = torch.floor(0.999 * bins * scaled).type(torch.int64)
    one_hot = F.one_hot(indices)
    hist = one_hot.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2).type(torch.float32)
    aggregated_hist = blur(hist, tile_size[0], sigma=sigma)[0]  # (C*bins) x H x W
    aggregated_hist = aggregated_hist.view(features.shape[1], bins, *features.shape[-2:])  # C x bins x H x W

    def dist(one, many):
        # Computes EMD distance between two 1D histograms
        vec_arr = one[:, None, None].expand_as(many)
        c1 = torch.cumsum(vec_arr, dim=0)
        c2 = torch.cumsum(many, dim=0)
        loss = torch.mean(torch.abs(c1 - c2), dim=0)
        del vec_arr
        return loss

    representative = torch.mean(aggregated_hist.flatten(start_dim=-2), dim=-1)  # C x bins
    return torch.stack([dist(c_rep, c_hist) for (c_rep, c_hist) in zip(representative, aggregated_hist)]).mean(dim=0)


def get_reference(reference_type, tile_size, num_ref=1):
    def reference_median(features):
        unf = F.unfold(features, tile_size, stride=tile_size)[0].T.reshape(-1, features.shape[1], *tile_size)
        val, _ = torch.sort(unf.view(-1, features.shape[1], tile_size[0] * tile_size[1]), dim=-1)
        return torch.median(val, dim=0).values  # C x tile**2

    def reference_random(features):
        h, w = features.shape[-2:]
        h0 = torch.randint(0, h - tile_size[0], (num_ref,), device=features.device)
        w0 = torch.randint(0, w - tile_size[1], (num_ref,), device=features.device)
        grid = torch.meshgrid(torch.arange(tile_size[0], device=features.device),
                              torch.arange(tile_size[1], device=features.device), indexing='ij')

        x_ind = (grid[0].reshape(-1)[None] + h0[:, None]).view(-1)
        y_ind = (grid[1].reshape(-1)[None] + w0[:, None]).view(-1)
        refs = features[..., x_ind, y_ind].reshape(*features.shape[:-2], num_ref, -1)
        refs = torch.sort(refs[0].permute(1, 0, 2), dim=-1).values  # num_ref x C x tile**2
        assert num_ref == 1  # Current public code only allows 1 (random) reference
        return refs[0]

    if reference_type == 'median':
        return reference_median
    if reference_type == 'random':
        return reference_random
    else:
        raise ValueError()


class ScSWW:
    def __init__(self, tile_size, chunk_size=8, sigma=6.0, reference_selection='median'):
        self.tile_size = tuple(tile_size)
        self.chunk_size = chunk_size
        self.sigma = sigma
        self.ref_selection = get_reference(reference_selection, self.tile_size)
        self.gaussian_kernel = None

    def generate_all_sets(self, features):
        b, c, h, w = features.shape
        padded = reflect_pad(features, self.tile_size[0])
        unf = torch.nn.functional.unfold(padded, (self.tile_size[0], padded.shape[-1]), stride=(1, 1))
        unf = unf[0].T.reshape(h, c, self.tile_size[0], padded.shape[-1])
        for i in range(0, len(unf), self.chunk_size):
            chunk = unf[i:i + self.chunk_size]
            unf_2 = torch.nn.functional.unfold(chunk, self.tile_size, stride=(1, 1))
            unf_2 = unf_2.transpose(1, 2).reshape(chunk.shape[0], w, c, -1)
            yield unf_2

    def __call__(self, features):
        r_set = self.ref_selection(features)
        generator = self.generate_all_sets(features)
        if self.gaussian_kernel is None:
            self.gaussian_kernel = get_gaussian_kernel(features.device, self.tile_size, self.sigma).reshape(-1)
        parts = []
        for f_set in generator:
            fvalues, ind = torch.sort(f_set, dim=-1)  # h x W x C x tile**2
            vec_arr = r_set[None, None].expand_as(fvalues)
            weights = torch.gather(self.gaussian_kernel.expand_as(fvalues), dim=-1, index=ind)
            loss = (F.l1_loss(fvalues, vec_arr, reduction='none') * weights).sum(dim=-1).mean(dim=-1)
            del vec_arr
            parts.append(loss)
        p = torch.cat(parts, dim=0)
        return p


class ScFCA(ScSWW):
    def __init__(self, tile_size, chunk_size=8, sigma_p=3.0, reference_selection='median', k_s=5, sigma_s=1.0):
        super(ScFCA, self).__init__(tile_size, chunk_size, sigma_p, reference_selection)
        assert tile_size[0] == tile_size[1]  # Only implemented for square patches
        self.p_size = (tile_size[0] // 2)
        if sigma_s is not None:
            self.local_blur = torchvision.transforms.GaussianBlur(k_s, sigma=sigma_s)
        else:
            self.local_blur = None

    def __call__(self, features):
        r_set = self.ref_selection(features)
        wp = features.shape[-1] + 2 * self.p_size
        generator = self.generate_all_sets(features)
        if self.gaussian_kernel is None:
            self.gaussian_kernel = get_gaussian_kernel(features.device, self.tile_size, self.sigma).reshape(-1)
        parts = []
        for f_set in generator:
            fvalues, ind = torch.sort(f_set, dim=-1)  # h x W x C x tile**2
            vec_arr = r_set[None, None].expand_as(fvalues)
            diff = F.l1_loss(fvalues, vec_arr, reduction='none')
            diff_re = torch.gather(diff, dim=-1, index=torch.argsort(ind)).mean(dim=2, keepdim=True)  # h x W x 1 x t**2
            if self.local_blur is not None:
                diff_re = self.local_blur(diff_re.view(-1, 1, *self.tile_size)).reshape(diff_re.shape)
            diff_re = diff_re * self.gaussian_kernel  # h x W x 1 x tile**2
            diff_re = diff_re.permute(0, 2, 3, 1).reshape(f_set.shape[0], -1, features.shape[-1])  # h x 1*tile**2 x W
            c_fold = F.fold(diff_re, (self.tile_size[0], wp), kernel_size=self.tile_size)  # h x C x tile x WP
            parts.append(c_fold)
        combined = torch.cat(parts, dim=0)  # H x 1 x tile x WP
        folded = F.fold(combined.permute(1, 2, 3, 0).reshape(1, -1, features.shape[-2]),
                        output_size=(wp, wp), kernel_size=(self.tile_size[0], wp))
        folded = folded[0, 0, self.p_size:-self.p_size, self.p_size:-self.p_size]  # Remove extra pad -> 1 x 1 x H x W

        return folded


def sc_aota(features, tile_size, sigma=100.0, k=400):
    def one_to_many_dist(values, dist_f):
        distances = []
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                distances.append(dist_f(values[:, i, j], values))
        return values.new_tensor(distances).reshape(values.shape[-2:])

    local_moments = blur(features, kernel_size=tile_size[0], sigma=sigma)[0]  # C x H x W

    def dist(one, many):
        vec_arr = one[:, None, None].expand_as(many)
        loss = ((vec_arr - many) ** 2).sum(dim=0).view(-1)
        del vec_arr
        loss = torch.topk(loss, k=k, largest=False).values.mean()
        return loss

    result = one_to_many_dist(local_moments, dist_f=dist)
    return blur(F.interpolate(result[None, None], (320, 320)), kernel_size=25, sigma=4.0)[0, 0]
