import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def add_gaussian_noise(epsilon: float) -> callable:
    """Returns a function that adds Gaussian noise to an input tensor."""
    def fwd(x, return_latents=False):
        z = torch.randn_like(x).to(x.device)
        x_n = x + epsilon * z
        if return_latents: 
            return x_n, z
        else: return x_n
    return fwd


def random_mask_image(mask_ratio: float, epsilon: float) -> callable:
    """Returns a function that randomly masks out a fraction of pixels in an image."""

    def fwd(image: torch.Tensor, return_latents=False):
        """
        Args:
            image: a 3-D tensor of shape (C, H, W) or
                   a 4-D batched tensor of shape (N, C, H, W).
        Returns:
            masked_image: same shape as `image`, with masked pixels set to 0.
            mask:         same shape, with 1.0 for *kept* pixels and 0.0 for masked pixels.
        """
        if image.dim() == 3:
            # Single image
            C, H, W = image.shape
            # sample a mask of shape (H, W)
            prob = torch.rand(H, W, device=image.device)
            single_mask = (prob > mask_ratio).float()        # 1=keep, 0=mask
            mask = single_mask.unsqueeze(0)#.expand(C, H, W)  # (C, H, W)
        elif image.dim() == 4:
            # Batch of images
            N, C, H, W = image.shape
            prob = torch.rand(N, H, W, device=image.device)
            batch_mask = (prob > mask_ratio).float()            # (N, H, W)
            mask = batch_mask.unsqueeze(1)#.expand(N, C, H, W)   # (N, C, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")

        x_n = image * mask
        z = torch.randn_like(image).to(image.device)
        x_n += z * epsilon

        if return_latents: 
            return x_n, mask
        else:
            return x_n

    return fwd



# def noise_and_mask_image(eps: float, mask_ratio: float) -> callable:
#     """Returns a function that randomly masks out a fraction of pixels in an image."""

#     def fwd(image: torch.Tensor):
#         """
#         Args:
#             image: a 3-D tensor of shape (C, H, W) or
#                    a 4-D batched tensor of shape (N, C, H, W).
#         Returns:
#             masked_image: same shape as `image`, with masked pixels set to 0.
#             mask:         same shape, with 1.0 for *kept* pixels and 0.0 for masked pixels.
#         """
#         z = torch.randn_like(image).to(image.device)
#         image  = image + eps*z

#         if image.dim() == 3:
#             # Single image
#             C, H, W = image.shape
#             # sample a mask of shape (H, W)
#             prob = torch.rand(H, W, device=image.device)
#             single_mask = (prob > mask_ratio).float()        # 1=keep, 0=mask
#             mask = single_mask.unsqueeze(0).expand(C, H, W)  # (C, H, W)
#         elif image.dim() == 4:
#             # Batch of images
#             N, C, H, W = image.shape
#             prob = torch.rand(N, H, W, device=image.device)
#             batch_mask = (prob > mask_ratio).float()            # (N, H, W)
#             mask = batch_mask.unsqueeze(1).expand(N, C, H, W)   # (N, C, H, W)
#         else:
#             raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")

#         masked_image = image * mask
        
#         return masked_image

#     return fwd


def gaussian_blur(sigma: float, epsilon: float) -> callable:
    """Returns a function that applies Gaussian blur to an input tensor."""        
    kernel_size = int(2 * math.ceil(3*sigma) + 1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def fwd(x, return_latents=False):
        x_b = gaussian_blur(x)
        z = torch.randn_like(x_b).to(x.device)
        x_b += epsilon * z
        if return_latents:
            return x_b, z
        else:
            return x_b

    return fwd


def random_block_mask(block_size, epsilon):
    """
    Randomly masks out `num_blocks` rectangular regions of size `block_size` in `image`.

    Args:
        image: Tensor of shape (C, H, W) or (N, C, H, W).
        block_size: int or (bh, bw). Size of each block to mask.
        num_blocks: how many blocks to place in each image.
        fill_value: value to assign inside each masked block (default 0.0).

    Returns:
        masked: Tensor same shape as `image`, with blocks set to `fill_value`.
        mask:   FloatTensor same shape, with 1.0 where kept, 0.0 where masked.
    """
    # normalize block_size to a tuple
    if isinstance(block_size, float):
        bh = bw = int(block_size)
    else:
        bh, bw = int(block_size[0]), int(block_size[1])

    def fwd(image, return_latents=False):
        # handle single image vs batch
        if image.ndim == 3:
            img = image.unsqueeze(0)
            squeeze_after = True
        elif image.ndim == 4:
            img = image*1.
            squeeze_after = False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.ndim}D")

        N, C, H, W = img.shape
        device = img.device

        # start with a mask of ones
        mask = torch.ones((N, 1, H, W), dtype=torch.float32, device=device)

        for n in range(N):
            top  = torch.randint(0, H - bh + 1, (1,)).item()
            left = torch.randint(0, W - bw + 1, (1,)).item()
            mask[n, :, top:top+bh, left:left+bw] = 0.0

        # apply
        img *= mask
        z = torch.randn_like(img).to(device)
        img += z * epsilon
        if squeeze_after:
            img = img.squeeze(0)
        
        if return_latents:
            return img, mask
        else:
            return img    
    return fwd



def motion_blur(kernel_size, angle, epsilon):
    """
    img: Tensor[C,H,W] or [N,C,H,W], float or double
    kernel_size: odd int
    angle: degrees
    """
    # normalize block_size to a tuple
    # 1) Create horizontal kernel on CPU
    kernel_size = int(kernel_size)
    k = torch.zeros(kernel_size, kernel_size)
    k[kernel_size // 2, :] = 1.0
    # rotate via grid_sample
    theta = torch.tensor([
        [ torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0],
        [ torch.sin(torch.deg2rad(torch.tensor(angle))),  torch.cos(torch.deg2rad(torch.tensor(angle))), 0]
    ]).unsqueeze(0)  # 1×2×3

    # need N=1,C=1,H=k,W=k
    k = k.unsqueeze(0).unsqueeze(0)  # 1×1×k×k
    grid = F.affine_grid(theta, k.size(), align_corners=False)
    k = F.grid_sample(k, grid, align_corners=False)
    k = k / k.sum()

    def fwd(img, return_latents=False):
        # 2) Convolve
        # ensure img is batched
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)
        # pad so output same size
        pad = kernel_size // 2
        out = F.conv2d(img, weight=k.expand(img.size(1), -1, -1, -1).to(img.device),
                    padding=pad, groups=img.size(1))
        out = out.squeeze(0) if was_3d else out
        z = torch.randn_like(img).to(img.device)
        out += z * epsilon

        if return_latents:
            return out, k
        else:
            return out   
    return fwd


corruption_dict = {
    'gaussian_noise': add_gaussian_noise,
    'random_mask': random_mask_image,
    'noise_and_mask': random_mask_image, 
    'gaussian_blur': gaussian_blur,
    'block_mask': random_block_mask,
    'motion_blur': motion_blur,
}
