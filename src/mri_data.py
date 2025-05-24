import sys, os
import h5py
import torch
import torch.nn as nn
import numpy as np
import math

def real2complex(x: torch.Tensor) -> torch.Tensor:
    """
    Split the last dim into real+imag and build a complex tensor.
    Expects x.shape = (..., 2*M), returns shape (..., M), dtype=torch.cfloat or cdouble.
    """
    a, b = x.chunk(2, dim=-1)          # each has half the last-dim
    return torch.complex(a, b)         # cfloat if a.dtype is float32, cdouble if float64

def complex2real(x: torch.Tensor) -> torch.Tensor:
    """Concatenate real and imag parts along the last axis."""
    return torch.cat((x.real, x.imag), dim=-1)

def fft2c(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Centered 2D FFT via: ifftshift → fft2 → fftshift on axes -3, -2.
    Works for real or complex inputs.
    """
    # PyTorch 1.8+ has these in torch.fft
    x0 = torch.fft.ifftshift(x, dim=(-3, -2))
    X  = torch.fft.fft2      (x0, dim=(-3, -2), norm=norm)
    return torch.fft.fftshift  (X, dim=(-3, -2))

def ifft2c(k: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """
    Centered 2D inverse FFT via: ifftshift → ifft2 → fftshift on axes -3, -2.
    """
    k0 = torch.fft.ifftshift(k, dim=(-3, -2))
    x  = torch.fft.ifft2      (k0, dim=(-3, -2), norm=norm)
    return torch.fft.fftshift  (x, dim=(-3, -2))

def make_mask(n, w, r, generator = None, device = None, mode='same_rate') -> torch.Tensor:
    """
    Creates a horizontal frequency subsampling mask.
    
    Args:
        n: batch size
        w: width of the mask (in pixels)
        r: downsampling factor (controls the band-stop width)
        generator: optional torch.Generator for deterministic sampling
        device: torch device to create the mask on
    Returns:
        mask: a BoolTensor of shape (1, 320, 1)
    """
    # 1) sample uniform noise in [0,1)
    if generator is None:
        A = torch.rand((n, 1, w, 1), device=device)
    else:
        A = torch.rand((n, 1, w, 1), generator=generator, device=device)

    # 2) threshold to get ~200/(320*r - 120) density
    calibration_region = 120. * (w / 320)
    if mode == 'same_rate':
        target_sample_count = 200. * (w / 320) 
    elif mode == 'same_number':
        target_sample_count = 200.
    thresh = target_sample_count / (w * r - calibration_region)
    # thresh = 200.0 / (320 * r - 120)
    mask = A < thresh            # BoolTensor of shape (1,320,1)

    # 3) force-set the central band to True
    center = w//2
    half = math.ceil(calibration_region / 2. / r)
    mask[:, :, center - half : center + half, :] = True
    return mask


# def make_mask_og(n, w, r, generator = None, device = None) -> torch.Tensor:
#     """
#     Creates a horizontal frequency subsampling mask.
    
#     Args:
#         r: downsampling factor (controls the band-stop width)
#         generator: optional torch.Generator for deterministic sampling
#         device: torch device to create the mask on
#     Returns:
#         mask: a BoolTensor of shape (1, 320, 1)
#     """
#     # 1) sample uniform noise in [0,1)
#     if generator is None:
#         A = torch.rand((n, 1, 320, 1), device=device)
#     else:
#         A = torch.rand((n, 1, 320, 1), generator=generator, device=device)

#     # 2) threshold to get ~200/(320*r - 120) density
#     thresh = 200.0 / (320 * r - 120)
#     mask = A < thresh            # BoolTensor of shape (1,320,1)

#     # 3) force-set the central band to True
#     center = 160
#     half = math.ceil(60 / r)
#     mask[:, :, center - half : center + half, :] = True
#     return mask


# def mri_subsampling(r, epsilon, downscale_factor=4):
#     """
#     Randomly subsample the k-space data of an image.
#     Args:
#         r: downsampling factor (controls the band-stop width)
#         epsilon: noise level
#     """
#     from mri_data import make_mask, fourier_to_pix, pix_to_fourier
#     downsampler = nn.PixelUnshuffle(downscale_factor=downscale_factor)
#     upsampler = nn.PixelShuffle(upscale_factor=downscale_factor)
 
#     def fwd(img, return_latents=False, generator=None):
#         was_3d = (img.dim() == 3)
#         if was_3d:
#             img = img.unsqueeze(0)
#         if img.shape[1] == 2 : # already in Fourier space
#             pass 
#         elif img.shape[1] == 16: # convert to Fourier space
#             img = upsampler(img)  # shape: [1, 1, 320, 320]
#             img = img.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
#             img = pix_to_fourier(img, channel_first=True)  # (N, 320, 320, 2)

#         # move channel last
#         # img = img.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
#         mask = make_mask(n=img.shape[0], r=r, generator=generator).to(img.device) # (N, 1, 320, 1)
#         y = img * mask
#         z = torch.randn(y.shape, generator=generator).to(img.device) 
#         y = y + z*epsilon
#         y = fourier_to_pix(y)
#         # move channel first
#         y = y.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
#         y = downsampler(y)  # shape: [1, 16, 80, 80]

#         if was_3d:
#             y = y.squeeze(0)
        
#         if return_latents:
#             mask = mask.squeeze(3).squeeze(1).to(float)
#             # mask = mask.expand(-1, 320, -1, -1).to(float)  # shape [N, 1, D, D]
#             # mask = mask.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
#             # mask = mask.expand(-1, -1, -1, 320).to(float)  # shape [N, 1, D, D]
#             # mask = mask.repeat(1, img.shape[0], 1, 320).to(float)  # shape [N, 1, D, D]
#             if was_3d:
#                 mask = mask.squeeze(0)
#             return y, mask
#         else:
#             return y

#     return fwd

def make_slices(f):
    with h5py.File(f, "r") as mri:
       slices = mri['reconstruction_rss'][10:41]
       slices = slices / slices.max()  # in [0, 1]
       slices = 4 * slices - 2  # in [-2, 2]
       # slices = slices[..., None]
       slices = np.expand_dims(slices, axis=1)
    return slices


def pix_to_fourier(x, channel_first=True):
    assert len(x.shape) == 4    
    assert x.shape[-1] == 1
    y = complex2real(fft2c(x))
    if channel_first:
        y = y.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
    return y


def fourier_to_pix(y):
    assert len(y.shape) == 4    
    if y.shape[-1] == 2:
        pass
    elif y.shape[1] == 2: #channel first
        y = y.permute(0, 2, 3, 1)
    slices = (ifft2c(real2complex(y))).real
    return slices



def mri_subsampling(r, epsilon, downscale_factor=4, mode='same_rate'):
    """
    Randomly subsample the k-space data of an image.
    Expects data in Fourier space (N, 2, D, D) or scrambled pixel space (N, s^2, D/s, D/s).
    Args:
        r: downsampling factor (controls the band-stop width)
        epsilon: noise level
    """
    from mri_data import make_mask, fourier_to_pix, pix_to_fourier
    downsampler = nn.PixelUnshuffle(downscale_factor=downscale_factor)
    upsampler = nn.PixelShuffle(upscale_factor=downscale_factor)
 
    def fwd(img, return_latents=False, generator=None):
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)
        if img.shape[1] == 2 : # already in Fourier space
            D = img.shape[2]
        elif img.shape[1] == (downscale_factor**2): # convert to Fourier space
            img = upsampler(img)  # shape: [N, 1, D, D]
            assert img.shape[1] == 1
            D = img.shape[2]
            img = img.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
            img = pix_to_fourier(img, channel_first=True)  # (N, 2, D, D)

        # move channel last
        mask = make_mask(n=img.shape[0], w=D,  \
                        r=r, generator=generator, mode=mode ).to(img.device) # (N, 1, D, 1)
        # print(img.shape, mask.shape)
        y = img * mask
        z = torch.randn(y.shape, generator=generator).to(img.device) 
        y = y + z*epsilon
        y = fourier_to_pix(y)
        # move channel first
        y = y.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
        y = downsampler(y)  # shape: [1, 16, 80, 80]

        if was_3d:
            y = y.squeeze(0)
        
        if return_latents:
            mask = mask.expand(-1, -1, -1, D)  # if mask is 1D
            # print(y.shape, mask.shape)
            mask = downsampler(mask).float()
            if was_3d:
                mask = mask.squeeze(0)
            return y, mask
        else:
            return y

    return fwd



def mri_subsampling2(r, epsilon, downscale_factor=4, mode='same_rate', fmri_mean=None, fmri_std=None):
    """
    Randomly subsample the k-space data of an image.
    Excects image in fourier space, stadard or scrambled.
    Args:
        r: downsampling factor (controls the band-stop width)
        epsilon: noise level
    """
    from mri_data import make_mask, fourier_to_pix, pix_to_fourier
    downsampler = nn.PixelUnshuffle(downscale_factor=downscale_factor)
    upsampler = nn.PixelShuffle(upscale_factor=downscale_factor)
    mean = np.load("/mnt/ceph/users/cmodi/ML_data/fastMRI/fourier-sub-mean.npy")
    std = np.load("/mnt/ceph/users/cmodi/ML_data/fastMRI/fourier-sub-std.npy")
    # fmri_mean = torch.from_numpy(mean).to(torch.float32)
    # fmri_std = torch.from_numpy(std).to(torch.float32)

    def fwd(img, return_latents=False, generator=None):
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)
        if img.shape[1] == 2 : # already in Fourier space
            D = img.shape[2]
        elif img.shape[1] == 2*(downscale_factor**2): # convert to Fourier space
            img = upsampler(img)  # shape: [N, 2, D, D]
            assert img.shape[1] == 2
            D = img.shape[2]

        if fmri_mean is not None:
            img = img * fmri_std.to(img.device) + fmri_mean.to(img.device)
        # move channel last
        mask = make_mask(n=img.shape[0], w=D, r=r, generator=generator, mode=mode).to(img.device) # (N, 1, 320, 1)
        mask = mask.expand(-1, -1, -1, D)#.to(float)  # shape [N, 1, D, D]
        # print("img mask shape", img.shape, mask.shape)
        y = img * mask
        z = torch.randn(y.shape, generator=generator).to(img.device) 
        y = y + z*epsilon
                
        if fmri_mean is not None:
            y = (y  - fmri_mean.to(img.device))/ fmri_std.to(img.device)
        y = downsampler(y)  # shape: [N, 2*s^2, D, D]
        # print("y shape",  y.shape)        

        if was_3d:
            y = y.squeeze(0)
        
        if return_latents:
            # mask = mask.expand(-1, -1, -1, D).to(float)  # shape [N, 1, D, D]
            mask = downsampler(mask).to(float)  # shape [N, 1, D, D]
            # print(y.shape, mask.shape)
            if was_3d:
                mask = mask.squeeze(0)
            return y, mask
        else:
            return y

    return fwd

def mri_subsampling3(r, epsilon, downscale_factor=4, mode='same_rate'):
    """
    Randomly subsample the k-space data of an image.
    Expects data in Fourier space (N, 2, D, D) or scrambled pixel space (N, s^2, D/s, D/s).
    Same as mri_subsampling, but returns image as latents.
    Args:
        r: downsampling factor (controls the band-stop width)
        epsilon: noise level
    """
    from mri_data import make_mask, fourier_to_pix, pix_to_fourier
    downsampler = nn.PixelUnshuffle(downscale_factor=downscale_factor)
    upsampler = nn.PixelShuffle(upscale_factor=downscale_factor)
 
    def fwd(img, return_latents=False, generator=None):
        was_3d = (img.dim() == 3)
        if was_3d:
            img = img.unsqueeze(0)
        if img.shape[1] == 2 : # already in Fourier space
            D = img.shape[2]
        elif img.shape[1] == (downscale_factor**2): # convert to Fourier space
            img = upsampler(img)  # shape: [N, 1, D, D]
            assert img.shape[1] == 1
            D = img.shape[2]
            img = img.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
            img = pix_to_fourier(img, channel_first=True)  # (N, 2, D, D)

        # move channel last
        mask = make_mask(n=img.shape[0], w=D,  \
                        r=r, generator=generator, mode=mode).to(img.device) # (N, 1, D, 1)
        # print(img.shape, mask.shape)
        y = img * mask
        z = torch.randn(y.shape, generator=generator).to(img.device) 
        y = y + z*epsilon
        y = fourier_to_pix(y)
        # move channel first
        y = y.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
        y = downsampler(y)  # shape: [1, 16, 80, 80]

        if was_3d:
            y = y.squeeze(0)
        
        if return_latents:
            mask = mask.expand(-1, -1, -1, D)  # if mask is 1D
            # print(y.shape, mask.shape)
            mask = downsampler(mask).float()
            if was_3d:
                mask = mask.squeeze(0)
            return y, y
        else:
            return y

    return fwd



if __name__=="__main__":

    folder_to_read = "/mnt/ceph/users/cmodi/ML_data/fastMRI/knee-singlecoil-train-slices/"
    folder_to_save = "/mnt/ceph/users/cmodi/ML_data/fastMRI/knee-singlecoil-train/"

    files = os.listdir(folder_to_read)
    print(len(files))
    j = 0 
    tosave = []
    for i, f in enumerate(files):
        print(f)
        file = os.path.join(folder_to_read, f)
        slices = np.load(file)
        y = complex2real(fft2c(torch.from_numpy(slices)))
        print(y.shape)
        tosave.append(y.numpy())
        if (i+1) % 40 == 0:
            print("Save")
            tosave = np.concatenate(tosave, axis=0)
            print(tosave.shape)
            print("train", j)
            np.save(os.path.join(folder_to_save, f"train_{j}.npy"), tosave)
            j += 1
            tosave = []
    tosave = np.concatenate(tosave, axis=0)
    print(tosave.shape)
    print("train", j)
    np.save(os.path.join(folder_to_save, f"train_{j}.npy"), tosave)
