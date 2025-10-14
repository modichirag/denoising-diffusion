import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from utils import infinite_dataloader


def add_gaussian_noise(epsilon: float) -> callable:
    """Returns a function that adds Gaussian noise to an input tensor."""

    def fwd(x, return_latents=False, generator=None):
        z = torch.randn(x.shape, generator=generator).to(x.device)
        x_n = x + epsilon * z
        # x_n = x + z * (torch.rand(x.shape, generator=generator).to(x.device) * epsilon + epsilon/2)
        if return_latents:
            return x_n, z
        else: return x_n
    return fwd


def random_mask_image(mask_ratio: float, epsilon: float, noise_mask=0.) -> callable:
    """Returns a function that randomly masks out a fraction of pixels in an image."""

    def fwd(image: torch.Tensor, return_latents=False, generator=None, latents=None):
        """
        Args:
            image: a 3-D tensor of shape (C, H, W) or
                   a 4-D batched tensor of shape (N, C, H, W).
        Returns:
            masked_image: same shape as `image`, with masked pixels set to 0.
            mask:         same shape, with 1.0 for *kept* pixels and 0.0 for masked pixels.
        """
        if latents is not None:
            mask = latents
        else:            
            if image.dim() == 3:
                # Single image
                C, H, W = image.shape
                # sample a mask of shape (H, W)
                prob = torch.rand(H, W, device=image.device, generator=generator)
                single_mask = (prob > mask_ratio).unsqueeze(0)  # (1, H, W)
                expanded_mask = single_mask.expand(C, H, W)
                mask = single_mask.float()#.expand(C, H, W)
            elif image.dim() == 4:
                # Batch of images
                N, C, H, W = image.shape
                prob = torch.rand(N, H, W, device=image.device, generator=generator)
                batch_mask = (prob > mask_ratio).unsqueeze(1)   # (N, 1, H, W)
                expanded_mask = batch_mask.expand(N, C, H, W)
                mask = batch_mask.float()#.expand(N, C, H, W)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")

        x_n = image * mask
        z = torch.randn(image.shape, generator=generator).to(image.device)
        x_n += z * epsilon
        if noise_mask > epsilon:
            noise = torch.randn(image.shape).to(image.device)*noise_mask
            noise[expanded_mask] = 0
            x_n[~expanded_mask] = 0
            # noise[mask.expand(-1, C, -1, -1) == 1] = 0
            # x_n[mask.expand(-1, C, -1, -1) == 0] = 0
            x_n += noise

        if return_latents:
            return x_n, mask
        else:
            return x_n

    return fwd


def gaussian_blur(sigma: float, epsilon: float) -> callable:
    """Returns a function that applies Gaussian blur to an input tensor."""
    kernel_size = int(2 * math.ceil(3*sigma) + 1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def fwd(x, return_latents=False, generator=None, latents=None):
        x_b = gaussian_blur(x)
        z = torch.randn(x_b.shape, generator=generator).to(x.device)
        x_b += epsilon * z
        if return_latents:
            return x_b, z
        else:
            return x_b

    return fwd


def gaussian_blur_pnoise(sigma: float, rate: float) -> callable:
    """Returns a function that applies Gaussian blur to an input tensor."""
    kernel_size = int(2 * math.ceil(3*sigma) + 1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def fwd(x, return_latents=False, generator=None, latents=None):
        x_b = gaussian_blur(x)
        rate_tensor = torch.ones(x_b.shape, device=x.device)*rate
        noise = torch.poisson(rate_tensor, generator=generator)
        x_b += noise
        if return_latents:
            return x_b, noise
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

    def fwd(img, return_latents=False, generator=None, latents=None):
        # Ensure img is batched
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)  # Add batch dimension
        batch_size = img.size(0)
        N, C, H, W = img.shape

        # sample angle and rotate via grid_sample
        if latents is not None:
            angles = (latents *  torch.pi).squeeze(1).to(img.device)
            if was_3d:
                angles = angles.unsqueeze(0)
        else:        
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



def jpeg_compression(min_quality=int(1), max_quality=int(100), epsilon=0.01):

    # Standard JPEG quant tables
    _QY = torch.tensor([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99],
    ], dtype=torch.float32)
    _QC = torch.tensor([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
    ], dtype=torch.float32)

    # Precompute the 8×8 orthonormal DCT transform matrix
    def _make_dct8(device):
        N=8
        M = torch.zeros((N,N), dtype=torch.float32, device=device)
        for k in range(N):
            # alpha = torch.sqrt(1/N) if k==0 else torch.sqrt(2/N)
            alpha = math.sqrt(1/N) if k==0 else math.sqrt(2/N)
            for n in range(N):
                M[k,n] = alpha * np.cos(torch.pi*(2*n+1)*k/(2*N))
        return M


    def compress(img: torch.Tensor, quality) -> torch.Tensor:
        """
        img: (3,H,W) in [0,1], float32
        quality: 1 (lowest)–100 (highest)
        returns: (3,H,W) in [0,1], float32
        """
        device = img.device
        C,H,W = img.shape
        # 1) RGB → [0,255] → YCbCr
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img.device).view(-1, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=img.device).view(-1, 1, 1)
        img = img * std + mean  # Now in [0, 1]
        x = img * 255.0
        R, G, B = x[0], x[1], x[2]
        Y  =  0.299   * R + 0.587   * G + 0.114   * B
        Cb = -0.168736* R - 0.331264* G + 0.5     * B + 128
        Cr =  0.5     * R - 0.418688* G - 0.081312* B + 128
        channels = [Y, Cb, Cr]

        # 2) Build quant tables for this quality
        q = quality
        # scale = 5000//q if q<50 else 200 - 2*q
        scale = torch.where(q < 50, 5000//q, 200 - 2*q)
        def make_Q(base):
            Q = ((base * scale + 50) / 100).floor().clamp(min=1, max=255)
            return Q.to(device)
        QY = make_Q(_QY)
        QC = make_Q(_QC)

        # 3) get DCT matrix
        M = _make_dct8(device)
        Mt = M.t()

        out_ch = []
        for idx,ch in enumerate(channels):
            Q  = QY if idx==0 else QC
            # shift to [-128,127]
            c = ch - 128.0
            # pad to multiple of 8
            H8 = ((H+7)//8)*8
            W8 = ((W+7)//8)*8
            pad = (0, W8-W, 0, H8-H)
            c_p = F.pad(c.unsqueeze(0).unsqueeze(0), pad, mode='constant', value=0)
            # unfold into [1,64,L]
            blocks = F.unfold(c_p, kernel_size=8, stride=8)  # shape = 1×64×L
            L = blocks.shape[-1]
            blocks = blocks.view(1,64,L).permute(2,0,1).view(-1,8,8)  # [L,8,8]

            # DCT -> quantize
            D = torch.matmul(M, torch.matmul(blocks, Mt))
            qD = (D / Q).round()

            # dequant + IDCT
            Dq = qD * Q
            rec = torch.matmul(Mt, torch.matmul(Dq, M))

            # fold back
            rec = rec.view(L,1,64).permute(1,2,0)  # 1×64×L
            c_rec = F.fold(rec, output_size=(H8,W8), kernel_size=8, stride=8)
            c_rec = c_rec[0,0,:H,:W] + 128.0
            out_ch.append(c_rec)

        # stack and convert back RGB
        Yr, Cbr, Crr = out_ch
        r = Yr + 1.402   * (Crr - 128)
        g = Yr - 0.344136* (Cbr - 128) - 0.714136*(Crr - 128)
        b = Yr + 1.772   * (Cbr - 128)
        out = torch.stack((r,g,b), dim=0).clamp(0,255) / 255.0
        out = (out - mean) / std
        return out

    def fwd(img, return_latents=False, generator=None):
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)  # Add batch dimension

        # sample angle and rotate via grid_sample
        quality = torch.randint(int(min_quality), int(max_quality+1), (img.shape[0],), \
                                generator=generator)
        out = torch.vmap(compress, in_dims=(0, 0), out_dims=(0))(img, quality)
        z = torch.randn(img.shape, generator=generator).to(img.device)
        out += z*epsilon
        out = out.squeeze(0) if was_3d else out

        if return_latents:
            latent = quality.to(img.device).float().unsqueeze(1)
            latent = latent.squeeze(0) if was_3d else latent
            # return out, quality.to(img.device).float().unsqueeze(1)
            return out, latent
        else:
            return out

    return fwd


def mri_pix1d(r, epsilon, downscale_factor=4, mode='same_rate', \
              noise_masked=False, interpolate_masked=False):

    from mri_data import MRI_Subsampling_Pixel
    mri = MRI_Subsampling_Pixel(r=r, epsilon=epsilon, downscale_factor=downscale_factor,\
                                expand_latents=False, mode=mode, \
                                noise_masked=noise_masked, \
                                interpolate_masked=interpolate_masked)

    def fwd(img, return_latents=False, generator=None):
        return mri(img, return_latents=return_latents, generator=generator)
    
    return fwd


def mri_pix3d(r, epsilon, downscale_factor=4, mode='same_rate', \
            noise_masked=False, interpolate_masked=False):

    from mri_data import MRI_Subsampling_Pixel
    mri = MRI_Subsampling_Pixel(r=r, epsilon=epsilon, downscale_factor=downscale_factor,\
                                expand_latents=True, mode=mode, \
                                noise_masked=noise_masked, \
                                interpolate_masked=interpolate_masked)

    def fwd(img, return_latents=False, generator=None):
        return mri(img, return_latents=return_latents, generator=generator)
    
    return fwd


# Helper function to compute Ax (coefficients y = Ax)
def compute_Ax(A_matrix: torch.Tensor, x_vector: torch.Tensor) -> torch.Tensor:
    """
    Computes Ax using einsum.
    A_matrix: shape (..., M, N)  e.g., (Batch, dim_out, dim_in)
    x_vector: shape (..., N)    e.g., (Batch, dim_in)
    Returns: shape (..., M)   e.g., (Batch, dim_out)
    """
    return torch.einsum('...ij,...j->...i', A_matrix, x_vector)

# Helper function to compute A^T y (projection from coefficients p = A^T y)
def compute_At_y(A_matrix: torch.Tensor, y_vector: torch.Tensor) -> torch.Tensor:
    """
    Computes A^T y using einsum.
    A_matrix: shape (..., M, N) e.g., (Batch, dim_out, dim_in)
    y_vector: shape (..., M)   e.g., (Batch, dim_out)
    Returns: shape (..., N)   e.g., (Batch, dim_in)
    """
    # Einsum for A^T y: sum over the M dimension (index i for A, index i for y)
    # ...ij corresponds to A, ...i corresponds to y.
    # Sum over i (dim_out), output dimension is j (dim_in).
    return torch.einsum('...ij,...i->...j', A_matrix, y_vector)

# Helper function to compute A^TA x
def compute_AtA_x(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes A^T(Ax) efficiently using a single einsum call.
    Equivalent to p_i = sum_j sum_k A_ji A_jk x_k

    A: shape (..., M, N) - The matrix defining the subspace basis vectors (rows)
                            (Indices represented as ...jk)
    x: shape (..., N)    - The vector (or batch of vectors) to project
                            (Indices represented as ...k)
    Returns: shape (..., N) - The projection vector p = A^T A x
                            (Indices represented as ...i)
"""
    # A (...ji) - effectively A^T
    # A (...jk) - the regular A
    # x (...k)  - the vector x
    # Sum over j (size M) and k (size N). Output indexed by i (size N).
    return torch.einsum('...ji,...jk,...k->...i', A, A, x)

def random_projection_coeff(dim_out: float, epsilon: float) -> callable:
    dim_out = int(dim_out)
    def fwd(x: torch.Tensor, return_latents=False, generator=None):
        """
        Args:
            x: a 2-D tensor of shape (B, dim_in)
        """
        N, dim_in = x.shape
        A = torch.randn(N, dim_out, dim_in, device=x.device)
        A = A / torch.linalg.norm(A, dim=-1, keepdim=True)
        # Below is random 2 * 2 rotation matrix for test
        # theta = torch.rand(N, device=x.device,) * 2 * torch.pi
        # cos_theta = torch.cos(theta)
        # sin_theta = torch.sin(theta)
        # A = torch.zeros(N, 2, 2, device=x.device)
        # A[:, 0, 0] = cos_theta
        # A[:, 0, 1] = -sin_theta
        # A[:, 1, 0] = sin_theta
        # A[:, 1, 1] = cos_theta

        x_n = compute_Ax(A, x)
        z = torch.randn(x_n.shape, generator=generator).to(x.device)
        x_n += z * epsilon
        padded = torch.randn(N, dim_in - dim_out, device=x.device)
        x_n = torch.cat([x_n, padded], dim=-1)

        if return_latents:
            return x_n, A
        else:
            return x_n
    return fwd

def random_projection_vec(dim_out: float, epsilon: float) -> callable:
    dim_out = int(dim_out)
    # for testing on fixed A
    A_base = torch.randn(1, dim_out, 2, device='cuda')
    A_base = A_base / torch.linalg.norm(A_base, dim=-1, keepdim=True)

    def fwd(x: torch.Tensor, return_latents=False, generator=None):
        """
        Args:
            x: a 2-D tensor of shape (B, dim_in)
        """
        N, dim_in = x.shape
        # A = A_base.repeat(N, 1, 1).to(x.device)
        A = torch.randn(N, dim_out, dim_in, device=x.device)
        A = A / torch.linalg.norm(A, dim=-1, keepdim=True)
        # A = (2.0*torch.rand([N, 1, dim_in], device=x.device)-1.0)

        x_n = compute_AtA_x(A, x)
        # add mixture with ful-rank projection
        # A2 = torch.randn(N, dim_in, dim_in, device=x.device)
        # A2 = A2 / torch.linalg.norm(A2, dim=-1, keepdim=True)
        # x_n2 = project_einsum(A2, x)
        # raw_mask = torch.bernoulli(torch.full((N, 1), 0.5)).to(x.device)
        # x_n = torch.where(raw_mask == 1, x_n, x_n2)
        z = torch.randn(x_n.shape, generator=generator).to(x.device)
        x_n += z * epsilon

        if return_latents:
            return x_n, A
        else:
            return x_n
    return fwd

def random_projection_vec_dataset(dataloader) -> callable:
    N_dl = dataloader.batch_size
    N_total = dataloader.dataset.__len__()
    dl = infinite_dataloader(dataloader)
    def fwd(x: torch.Tensor, return_latents=False, generator=None):
        """
        Args:
            x: a 2-D tensor of shape (B, dim_in)
        """
        N, _ = x.shape
        if N == N_dl:
            A = next(dl).to(x.device)
        else:
            indices = torch.randperm(N_total)[:N]
            A = dataloader.dataset[indices].to(x.device)
        x_n = compute_Ax(A, x)
        z = torch.randn(x_n.shape, generator=generator).to(x.device)
        x_n += z * 0.01
        x_n = compute_At_y(A, x_n)

        if return_latents:
            return x_n, A
        else:
            return x_n
    return fwd

corruption_dict = {
    'gaussian_noise': add_gaussian_noise,
    'random_mask': random_mask_image,
    'noise_and_mask': random_mask_image,
    'gaussian_blur': gaussian_blur,
    'gaussian_blur_pnoise': gaussian_blur_pnoise,
    'block_mask': random_block_mask,
    'motion_blur': motion_blur,
    'random_motion': random_motion,
    'random_motion2': random_motion2,
    'jpeg_compress': jpeg_compression,
    'mri_pix1d': mri_pix1d,
    'mri_pix3d': mri_pix3d,
    'projection_coeff': random_projection_coeff,
    'projection_vec': random_projection_vec,
    'projection_vec_ds': random_projection_vec_dataset,
}

def parse_latents(corruption, D, s=None):
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
    elif corruption == 'mri_pix1d':
        if s is None:
            raise ValueError("For 'mri_pix1d', 's' must be provided.")
        use_latents = True
        latent_dim = [int(s*D)]
    elif corruption == 'mri_pix3d':
        if s is None:
            raise ValueError("For 'mri_pix3d', 's' must be provided.")
        use_latents = True
        latent_dim = [int(s**2), D, D]
    elif corruption == 'jpeg_compress':
        use_latents = True
        latent_dim = [1]
    elif corruption.startswith('projection'):
        use_latents = True
        latent_dim = [1] # artificial, to be corrected later
    else:
        use_latents = False
        latent_dim = None
    return use_latents, latent_dim
