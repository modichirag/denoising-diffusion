import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def add_gaussian_noise(epsilon: float) -> callable:
    """Returns a function that adds Gaussian noise to an input tensor."""

    def fwd(x, return_latents=False, generator=None):
        z = torch.randn(x.shape, generator=generator).to(x.device)
        x_n = x + epsilon * z
        if return_latents: 
            return x_n, z
        else: return x_n
    return fwd


def random_mask_image(mask_ratio: float, epsilon: float) -> callable:
    """Returns a function that randomly masks out a fraction of pixels in an image."""

    def fwd(image: torch.Tensor, return_latents=False, generator=None):
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
            prob = torch.rand(H, W, device=image.device, generator=generator)
            single_mask = (prob > mask_ratio).float()        # 1=keep, 0=mask
            mask = single_mask.unsqueeze(0)#.expand(C, H, W)  # (C, H, W)
        elif image.dim() == 4:
            # Batch of images
            N, C, H, W = image.shape
            prob = torch.rand(N, H, W, device=image.device, generator=generator)
            batch_mask = (prob > mask_ratio).float()            # (N, H, W)
            mask = batch_mask.unsqueeze(1)#.expand(N, C, H, W)   # (N, C, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")

        x_n = image * mask
        z = torch.randn(image.shape, generator=generator).to(image.device)
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

    def fwd(x, return_latents=False, generator=None):
        x_b = gaussian_blur(x)
        z = torch.randn(x_b.shape, generator=generator).to(x.device)
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
    if isinstance(block_size, float) or isinstance(block_size, int):
        bh = bw = int(block_size)
    else:
        bh, bw = int(block_size[0]), int(block_size[1])

    def fwd(image, return_latents=False,  generator=None):
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
            top  = torch.randint(0, H - bh + 1, (1,), generator=generator).item()
            left = torch.randint(0, W - bw + 1, (1,), generator=generator).item()
            mask[n, :, top:top+bh, left:left+bw] = 0.0

        # apply
        img *= mask
        z = torch.randn(img.shape, generator=generator).to(device)
        img += z * epsilon
        if squeeze_after:
            img = img.squeeze(0)
            mask = mask.squeeze(0)
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
    assert kernel_size % 2 == 1, "kernel_size must be odd"
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

    def fwd(img, return_latents=False, generator=None):
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
        
        z = torch.randn(img.shape, generator=generator).to(img.device)
        out += z * epsilon
        if was_3d:
            out = out.squeeze(0)
        if return_latents:
            return out, k
        else:
            return out   
    return fwd


def random_motion(kernel_size, epsilon):
    """
    img: Tensor[C,H,W] or [N,C,H,W], float or double
    kernel_size: odd int
    """
    # normalize block_size to a tuple
    # 1) Create horizontal kernel on CPU
    kernel_size = int(kernel_size)
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    k = torch.zeros(kernel_size, kernel_size)
    k[kernel_size // 2, :] = 1.0
    k = k.unsqueeze(0).unsqueeze(0)  # 1×1×k×k
    pad = kernel_size // 2

    def fwd(img, return_latents=False, generator=None):
        # Ensure img is batched
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)  # Add batch dimension
        batch_size = img.size(0)
        N, C, H, W = img.shape

        # sample angle and rotate via grid_sample
        angles = (torch.rand(batch_size, generator=generator) - 0.5) * 360.
        angles = torch.deg2rad(angles).to(img.device)  # Convert to radians
        thetas = []
        for angle in angles:
            theta = torch.tensor([
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle),  torch.cos(angle), 0]
            ])
            thetas.append(theta)
        thetas = torch.stack(thetas).to(img.device)  # Shape: [batch_size, 2, 3]
 
        # 3) Rotate kernels in batch via grid_sample
        #    Expand k to (N,1,Kh,Kw) for grid_sample
        k_batch = k.expand(batch_size, 1, kernel_size, kernel_size).to(img.device)
        grid = F.affine_grid(thetas, k_batch.size(), align_corners=False)
        ka = F.grid_sample(k_batch, grid, align_corners=False)
        # Normalize each kernel
        ka = ka / ka.flatten(1).sum(dim=1).view(N, 1, 1, 1)

        # 4) Depthwise conv with per-sample kernels using grouped conv
        #    Reshape input to (1, N*C, H, W) and kernels to (N*C, 1, Kh, Kw)
        img_reshaped = img.reshape(1, N * C, H, W)
        # Repeat each rotated kernel C times for each channel
        weight = ka.repeat_interleave(C, dim=0)  # (N*C,1,Kh,Kw)
        # Convolve with groups=N*C and reshape back to (N,C,H,W)
        out = F.conv2d(img_reshaped, weight=weight, padding=pad, groups=N*C)
        out = out.view(N, C, H, W)     

        z = torch.randn(img.shape, generator=generator).to(img.device)
        out += z * epsilon
        out = out.squeeze(0) if was_3d else out

        if return_latents:
            latent = (angles / torch.pi).unsqueeze(1)
            latent = latent.squeeze(0) if was_3d else latent
            return out, latent
        else:
            return out   
    return fwd


def random_motion2(kernel_size, epsilon):
    """
    img: Tensor[C,H,W] or [N,C,H,W], float or double
    kernel_size: odd int
    """
    # normalize block_size to a tuple
    # 1) Create horizontal kernel on CPU
    kernel_size = int(kernel_size)
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    kernel = torch.zeros(kernel_size, kernel_size)
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1×1×k×k
    pad = kernel_size // 2

    def direction_map_projection(cos, sin, H: int, W: int):
        """
        Returns a single-channel map (H,W) where each pixel's
        value = (x, y)·(cos(theta), sin(theta)), with x,y in [-1,+1].
        """
        # vx, vy = cos, sin    
        # build normalized coordinate grids from -1 to +1
        xs = torch.linspace(-1.0, 1.0, W, device=cos.device).view(1, W).expand(H, W)
        ys = torch.linspace(-1.0, 1.0, H, device=sin.device).view(H, 1).expand(H, W)
        # project each (x,y) onto the direction vector (vx, vy)
        proj = cos * xs + sin * ys   # shape (H, W)
        return proj

    
    def fwd_single(img, cos, sin):

        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)

        # angle = torch.deg2rad(angle).to(img.device)  # Convert to radians
        k = kernel.to(img.device)
        # cos = torch.cos(angle)                    # (N,)
        # sin = torch.sin(angle)                    # (N,)
        zeros = torch.zeros_like(cos)           # (N,)
        
        # 3) pack into the batch of 2×3 matrices
        row1 = torch.stack([ cos, -sin, zeros ], dim=0)  # (N, 3)
        row2 = torch.stack([ sin,  cos, zeros ], dim=0)  # (N, 3)
        theta = torch.stack([row1, row2], dim=0).unsqueeze(0)         # (N, 2, 3)

        # need N=1,C=1,H=k,W=k
        grid = F.affine_grid(theta, k.size(), align_corners=False)
        k = F.grid_sample(k, grid, align_corners=False)
        k = k / k.sum()
            
        # pad so output same size
        pad = kernel_size // 2
        out = F.conv2d(img, weight=k.expand(img.size(1), -1, -1, -1).to(img.device),
                    padding=pad, groups=img.size(1))
        out = out.squeeze(0) if was_3d else out

        return out

    def fwd(img, angles=None, return_latents=False, generator=None):
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)  # Add batch dimension
        batch_size = img.size(0)
        N, C, H, W = img.shape

        # sample angle and rotate via grid_sample
        if angles is  None:
            angles = (torch.rand(batch_size, generator=generator) - 0.5) * 360.  
        angles = angles.to(img.device)
        rads = torch.deg2rad(angles)
        cos, sin = torch.cos(rads), torch.sin(rads)
        out = torch.vmap(fwd_single, in_dims=(0, 0, 0), out_dims=(0))(img, cos, sin)

        z = torch.randn(img.shape, generator=generator).to(img.device)
        out += z*epsilon
        out = out.squeeze(0) if was_3d else out

        if return_latents:
            latent = torch.vmap(direction_map_projection, in_dims=(0, 0, None, None), out_dims=(0))  (cos, sin, H, W)
            latent = latent.unsqueeze(1)
            latent = latent.squeeze(0) if was_3d else latent
            return out, latent
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
    'random_motion': random_motion,
    'random_motion2': random_motion2,
}

def parse_latents(corruption, D):
    """Parse the corruption function and return the latent dimensions."""
    if 'mask' in corruption:
        use_latents = True
        latent_dim = [1, D, D]
    elif corruption == 'random_motion':
        use_latents = True
        latent_dim = [1]
    elif corruption == 'random_motion2':
        use_latents = True
        latent_dim = [1, D, D]
    else:
        use_latents = False
        latent_dim = None
    return use_latents, latent_dim
