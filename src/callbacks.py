
import torch
import os
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from utils import grab, push_to_device

## Expected signature of callback_fn
# callback_fn(milestone, b, deconvolver, 
#               dataloader, validation_data, losses, device, results_folder)

def get_samples(b, deconvolver, dataloader, device, validation_data):
    if validation_data is None:
        data, obs, latents = next(dataloader)
    else:
        image, obs, latents = validation_data
    data, obs, latents = push_to_device(data, obs, latents, device=device)
    latents = latents if deconvolver.use_latents else None
    clean = deconvolver.transport(b, obs, latents)
    return data, obs, latents, clean



def save_image(idx, b, deconvolver, dataloader, device, results_folder, losses, validation_data,  **kargs):

    data, obs, latents, clean = get_samples(b, deconvolver, dataloader, device, validation_data)    
    to_show = [data, obs, clean]

    fig, axar = plt.subplots(len(to_show), 8, figsize=(8, 3), sharex=True, sharey=True)
    vmax, vmin = data.max()*1.1, data.min()*0.5
    for i in range(len(to_show)):
        for j in range(8):
            ax = axar[i, j]
            im = ax.imshow(grab(to_show[i][j]).transpose(1, 2, 0), vmax=vmax, vmin=vmin)
    axar[0, 0].set_ylabel('original\nImage')
    axar[1, 0].set_ylabel(f'Corrupted')
    axar[2, 0].set_ylabel(f'Clean')
    for axis in axar.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce spacing
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()



def save_mri_fig(idx, image, corrupted, clean, results_folder, **kargs):

    from mri_data import fourier_to_pix
    shuffler = torch.nn.PixelShuffle(4)
    to_show = [fourier_to_pix(image), \
               shuffler(corrupted).permute(0, 2, 3, 1), \
            shuffler(clean).permute(0, 2, 3, 1)]

    fig, axar = plt.subplots(len(to_show), 8, figsize=(8, 3), sharex=True, sharey=True)
    vmax, vmin = None, None
    for i in range(len(to_show)):
        for j in range(8):
            ax = axar[i, j]
            im = ax.imshow(grab(to_show[i][j]), vmax=vmax, vmin=vmin)
    axar[0, 0].set_ylabel('Original\nImage')
    axar[1, 0].set_ylabel(f'Corrupted\n$\sigma ${epsilon}')
    axar[2, 0].set_ylabel(f'Clean')
    for axis in axar.flatten():
        axis.set_xticks([])
        axis.set_yticks([])
    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce spacing
    # plt.tight_layout()
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()


def save_fig_2dsynt_vec(idx, clean, corrupted, generated, results_folder, **kargs):
    c = '#62508f' # plot color
    push_fwd_func = kargs.get('push_fwd_func', None)
    if push_fwd_func is None:
        fig, axes = plt.subplots(1,3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
    clean = grab(clean)
    corrupted = grab(corrupted)
    generated = grab(generated)

    axes[0].scatter(clean[:,0], clean[:,1], alpha = 0.03, c = c)
    axes[0].set_title(r"Clean samples", fontsize = 18)
    axes[0].set_xlim(-6,6), axes[0].set_ylim(-6,6)
    axes[0].set_xticks([-4,0,4]), axes[0].set_yticks([-4,0,4])

    axes[1].scatter(corrupted[:,0], corrupted[:,1], alpha = 0.03, c = c)
    axes[1].set_title(r"Corrupted samples", fontsize = 18)
    axes[1].set_xlim(-6,6), axes[2].set_ylim(-6,6)
    axes[1].set_xticks([-4,0,4]), axes[2].set_yticks([]);

    axes[2].scatter(generated[:,0], generated[:,1], alpha = 0.03, c = c)
    axes[2].set_title(r"Generated samples ", fontsize = 18)
    axes[2].set_xlim(-6,6), axes[1].set_ylim(-6,6)
    axes[2].set_xticks([-4,0,4]), axes[1].set_yticks([])

    if push_fwd_func is not None:
        generated_corrupted = push_fwd_func(torch.from_numpy(generated)).numpy()
        axes[3].scatter(generated_corrupted[:,0], generated_corrupted[:,1], alpha = 0.03, c = c)
        axes[3].set_title(r"Generated corrupted samples ", fontsize = 18)
        axes[3].set_xlim(-6,6), axes[3].set_ylim(-6,6)
        axes[3].set_xticks([-4,0,4]), axes[3].set_yticks([])

    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce spacing
    # plt.tight_layout()
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()


def save_fig_2dsynt_coeff(idx, clean, corrupted, generated, results_folder, **kargs):
    c = '#62508f' # plot color
    push_fwd_func = kargs.get('push_fwd_func', None)
    latents = kargs.get('latents', None)
    assert latents is not None, "Latents should be provided for this function"
    latents = latents.squeeze()
    assert latents.shape[-1] == 2, "Latents should be 2D for this function"
    angles_rad = grab(torch.atan2(latents[:, 1], latents[:, 0]))
    if push_fwd_func is None:
        fig, axes = plt.subplots(1,3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
    clean = grab(clean)
    corrupted = grab(corrupted)
    generated = grab(generated)

    axes[0].scatter(clean[:,0], clean[:,1], alpha = 0.03, c = c)
    axes[0].set_title(r"Clean samples", fontsize = 18)
    axes[0].set_xlim(-6,6), axes[0].set_ylim(-6,6)
    axes[0].set_xticks([-4,0,4]), axes[0].set_yticks([-4,0,4])

    axes[1].scatter(corrupted[:,0], angles_rad, alpha = 0.03, c = c)
    axes[1].set_title(r"Corrupted samples", fontsize = 18)
    axes[1].set_xlim(-6,6), axes[2].set_ylim(-6,6)
    axes[1].set_xticks([-4,0,4]), axes[2].set_yticks([]);

    axes[2].scatter(generated[:,0], generated[:,1], alpha = 0.03, c = c)
    axes[2].set_title(r"Generated samples ", fontsize = 18)
    axes[2].set_xlim(-6,6), axes[1].set_ylim(-6,6)
    axes[2].set_xticks([-4,0,4]), axes[1].set_yticks([])

    if push_fwd_func is not None:
        generated_corrupted, latents_new = push_fwd_func(torch.from_numpy(generated), return_latents=True)
        generated_corrupted = grab(generated_corrupted)
        latents_new = latents_new.squeeze()
        angles_rad_new = grab(torch.atan2(latents_new[:, 1], latents_new[:, 0]))
        axes[3].scatter(generated_corrupted[:,0], angles_rad_new, alpha = 0.03, c = c)
        axes[3].set_title(r"Generated corrupted samples ", fontsize = 18)
        axes[3].set_xlim(-6,6), axes[3].set_ylim(-6,6)
        axes[3].set_xticks([-4,0,4]), axes[3].set_yticks([])

    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce spacing
    plt.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close()


def save_fig_manifold(idx, clean, corrupted, generated, results_folder, push_fwd_func=None):
    clean = grab(clean)
    corrupted = grab(corrupted)
    generated = grab(generated)
    cmap = plt.get_cmap('Blues')
    colors = [cmap(i) for i in range(16, cmap.N)]
    colors = [(1.0, 1.0, 1.0), *colors]
    cmap = plt.cm.colors.ListedColormap(colors)

    pairplot_grid = seaborn.pairplot(
        data=pd.DataFrame({f'x{i}': xi for i, xi in enumerate((generated).T)}),
        corner=True,
        kind='hist',
        plot_kws={'bins': 64, 'binrange': (-3, 3), 'thresh': None, 'cmap': cmap},
        diag_kws={'bins': 64, 'binrange': (-3, 3), 'element': 'step', 'color': cmap(cmap.N // 2)},
    )
    pairplot_grid.tight_layout(pad=0.5)
    pairplot_grid.savefig(f'{results_folder}/denoising_{idx}.png', dpi=300)
    plt.close(pairplot_grid.fig)


def save_losses_fig(losses, results_folder):
    steps = np.arange(len(losses))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].semilogy(steps, losses, marker='.', linestyle='-', markersize=4, alpha=0.7)
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Loss (log scale)")
    axs[0].set_title("Loss Curve (Semi-Log Y Scale)")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].loglog(steps, losses, marker='.', linestyle='-', markersize=4, alpha=0.7, color='orangered')
    axs[1].set_xlabel("Steps (log scale)")
    axs[1].set_ylabel("Loss (log scale)")
    axs[1].set_title("Loss Curve (Log-Log Scale)")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    print(os.path.join(results_folder, 'losses.png'))
    plt.savefig(os.path.join(results_folder, 'losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
