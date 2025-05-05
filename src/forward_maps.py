import math
import torch
import torchvision.transforms as transforms

def add_gaussian_noise(epsilon: float) -> callable:
    """Returns a function that adds Gaussian noise to an input tensor."""
    def fwd(x, return_latent=False):
        z = torch.randn_like(x).to(x.device)
        x_n = x + epsilon * z
        if return_latent: 
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
            mask = single_mask.unsqueeze(0).expand(C, H, W)  # (C, H, W)
        elif image.dim() == 4:
            # Batch of images
            N, C, H, W = image.shape
            prob = torch.rand(N, H, W, device=image.device)
            batch_mask = (prob > mask_ratio).float()            # (N, H, W)
            mask = batch_mask.unsqueeze(1).expand(N, C, H, W)   # (N, C, H, W)
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

    def fwd(x):
        x_b = gaussian_blur(x)
        z = torch.randn_like(x_b).to(x.device)
        x_b += epsilon * z
        return x_b

    return fwd

corruption_dict = {
    'gaussian_noise': add_gaussian_noise,
    'random_mask': random_mask_image,
    'noise_and_mask': random_mask_image, 
    'gaussian_blur': gaussian_blur
}
