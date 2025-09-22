import torch
from torch.utils.data import DataLoader
from utils import infinite_dataloader
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import grab
from karras_unet_1d import PixelShuffle1D, PixelUnShuffle1D



def power_law_continuum(wavelength, alpha=1.0, A=1.0):
    """ Generate a power-law continuum for quasar spectra """
    flux = A * (wavelength / 1000) ** (-alpha)  # Assuming wavelength in Angstroms
    return flux

def broken_power_law_continuum(wavelength, alpha1=1.0, alpha2=0.5, lambda_break=2500, A=1.0):
    """ Generate a broken power-law continuum for quasar spectra """
    flux = torch.zeros_like(wavelength)
    flux[wavelength < lambda_break] = A * (wavelength[wavelength < lambda_break] / 1000) ** (-alpha1)
    flux[wavelength >= lambda_break] = A * (wavelength[wavelength >= lambda_break] / 1000) ** (-alpha2)
    return flux

def blackbody_continuum(wavelength, T=15000):
    """ Generate a blackbody continuum for quasar spectra """
    h = 6.62607015e-34  # Planck's constant in J·s
    c = 3.0e8  # Speed of light in m/s
    k = 1.380649e-23  # Boltzmann constant in J/K
    wavelength_meters = wavelength * 1e-10  # Convert from Angstroms to meters
    intensity = (2 * h * c**2) / (wavelength_meters**5) * (1 / (torch.exp((h * c) / (wavelength_meters * k * T)) - 1))
    flux = intensity * 1e-17  # Scaling for flux units (erg/cm^2/s/Angstrom)
    return flux

def empirical_quasar_sed(wavelength):
    """ Generate an empirical quasar SED template for the continuum """
    flux = power_law_continuum(wavelength, alpha=1.0)
    flux += blackbody_continuum(wavelength, T=15000)
    return flux

def generate_quasar_spectrum(wave_range=(1200, 1600), num_points=1000, lines=None, continuum_type='power_law', continuum_params=None):
    """
    Generate a synthetic quasar spectrum with Gaussian emission lines and a continuum model.
    
    Parameters:
        wave_range (tuple): (min_wavelength, max_wavelength) in Angstroms (rest-frame).
        num_points (int): Number of wavelength sample points in the spectrum.
        lines (list of dict, optional): Emission line parameters. Each dict should have 'name', 'wave', 'amp', 'fwhm'.
        continuum_type (str): Type of continuum ('power_law', 'broken_power_law', 'blackbody', 'empirical').
        continuum_params (dict): Parameters specific to the continuum model chosen (e.g., alpha for power-law).
        
    Returns:
        wavelength (torch.Tensor): Wavelength array (Å).
        flux (torch.Tensor): Flux array (arbitrary units).
        metadata (dict): Dictionary with details of lines and generation settings.
    """
    # Create wavelength array (uniform sampling)
    wavelength = torch.linspace(wave_range[0], wave_range[1], num_points)
    
    # Generate continuum flux based on the chosen model
    if continuum_type == 'power_law':
        alpha = continuum_params.get('alpha', 1.0)
        A = continuum_params.get('A', 1.0)
        flux_continuum = power_law_continuum(wavelength, alpha=alpha, A=A)
    
    elif continuum_type == 'broken_power_law':
        alpha1 = continuum_params.get('alpha1', 1.0)
        alpha2 = continuum_params.get('alpha2', 0.5)
        lambda_break = continuum_params.get('lambda_break', 2500)
        A = continuum_params.get('A', 1.0)
        flux_continuum = broken_power_law_continuum(wavelength, alpha1, alpha2, lambda_break, A)
    
    elif continuum_type == 'blackbody':
        T = continuum_params.get('T', 15000)
        flux_continuum = blackbody_continuum(wavelength, T)
    
    elif continuum_type == 'empirical':
        flux_continuum = empirical_quasar_sed(wavelength)
    
    else:
        raise ValueError(f"Unknown continuum type: {continuum_type}")
    
    # Start with the continuum flux
    flux = flux_continuum
    
    # Default emission lines if none provided
    if lines is None:
        lines = [
            {'name': 'Lyα',          'wave': 1215.7, 'amp': 1.0, 'fwhm': 8.0},   # Lyman-alpha
            {'name': 'N V',          'wave': 1240.0, 'amp': 0.3, 'fwhm': 8.0},   # N V doublet (approximate)
            {'name': 'Si IV + O IV]', 'wave': 1400.0, 'amp': 0.5, 'fwhm': 8.0},  # Si IV + O IV] blend
            {'name': 'C IV',         'wave': 1549.0, 'amp': 0.8, 'fwhm': 8.0}    # C IV doublet (approximate)
        ]
    
    # Add each emission line as a Gaussian profile on the continuum
    for line in lines:
        center = line['wave']
        amplitude = line['amp']
        sigma = line['fwhm'] / 2.355  # Convert FWHM to standard deviation
        flux += amplitude * torch.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    
    # Prepare metadata
    metadata = {
        'wave_range': wave_range,
        'num_points': num_points,
        'continuum_type': continuum_type,
        'continuum_params': continuum_params,
        'emission_lines': [line.copy() for line in lines],  # Copy of line parameters
        'corruptions': []  # To record any applied corruptions
    }
    
    return wavelength, flux, metadata

def add_gaussian_noise(wavelength, flux, noise_std=0.1, SNR=None, meta=None):
    """
    Add Gaussian random noise to the flux.
    Parameters:
        noise_std (float): Standard deviation of the Gaussian noise (in flux units).
        meta (dict, optional): Metadata dictionary to update.
    Returns:
        wavelength (torch.Tensor): Same as input.
        new_flux (torch.Tensor): Flux with noise added.
        meta (dict): Updated metadata (if provided).
    """
    # Generate random noise array
    if SNR is not None:
        noise_std = torch.mean(flux) / SNR  + 1e-4 # Estimate noise standard deviation based on SNR
    noise = torch.normal(mean=torch.zeros_like(wavelength), std=noise_std).to(flux.device)
    new_flux = flux + noise
    # Record this corruption in metadata
    if meta is not None:
        meta['corruptions'].append({'type': 'gaussian_noise',
        
        'noise_std': noise_std})
    return wavelength, new_flux, meta


# def degrade_resolution(wavelength, flux, fwhm=3.0, meta=None, radius=10):
#     """
#     Degrade spectral resolution by convolving with a Gaussian kernel.
#     Parameters:
#         fwhm (float): Full-width at half-maximum of the Gaussian kernel (in Å).
#         meta (dict, optional): Metadata dictionary to update.
#     Returns:
#         wavelength (torch.Tensor): Same as input.
#         new_flux (torch.Tensor): Flux after convolution (blurred).
#         meta (dict): Updated metadata.
#     """
#     # Assume uniform wavelength spacing
#     delta_lambda = wavelength[1] - wavelength[0]
#     # Convert FWHM to standard deviation in wavelength space
#     sigma_lambda = fwhm / 2.355
#     # Convert sigma to units of array index spacing
#     sigma_pixels = sigma_lambda / delta_lambda
#     # Build Gaussian kernel (truncate at ~4σ for efficiency)
#     # radius = torch.ceil(4 * sigma_pixels).to(torch.int64)
#     idx = torch.arange(-radius, radius + 1).to(flux.device)
#     kernel = torch.exp(-0.5 * (idx / sigma_pixels) ** 2)
#     kernel /= kernel.sum()  # normalize kernel to preserve flux

#     # Convolve flux with the kernel
#     new_flux = torch.conv1d(flux[None, None, :], kernel[None, None, :], padding=radius).squeeze()
#     # Record in metadata
#     if meta is not None:
#         meta['corruptions'].append({'type': 'resolution_degradation', 'fwhm': fwhm})
#     return wavelength, new_flux, meta

def degrade_resolution(wavelength, flux, dloglambda=3., meta=None, radius=10):
    """
    Degrade spectral resolution by convolving with a Gaussian kernel.
    Parameters:
        fwhm (float): Full-width at half-maximum of the Gaussian kernel (in Å).
        meta (dict, optional): Metadata dictionary to update.
    Returns:
        wavelength (torch.Tensor): Same as input.
        new_flux (torch.Tensor): Flux after convolution (blurred).
        meta (dict): Updated metadata.
    """

    loglamb = torch.log(wavelength)
    delta_lambda = loglamb[1] - loglamb[0]
    sigma_lambda = dloglambda
    sigma_pixels = sigma_lambda / delta_lambda
    idx = torch.arange(-radius, radius + 1).to(flux.device)
    kernel = torch.exp(-0.5 * (idx / sigma_pixels) ** 2)
    kernel /= kernel.sum()  # normalize kernel to preserve flux

    # Convolve flux with the kernel
    new_flux = torch.conv1d(flux[None, None, :], kernel[None, None, :], padding=radius).squeeze()
    # Record in metadata
    if meta is not None:
        meta['corruptions'].append({'type': 'resolution_degradation', 'dloglambda': dloglambda})
    return wavelength, new_flux, meta


def add_absorption_features(wavelength, flux, features=None, n_features=3, meta=None):
    """
    Introduce narrow absorption lines (dips) into the spectrum.
    Parameters:
        features (list of dict, optional): Each dict with keys 'center', 'depth', 'fwhm'.
                                           'depth' is the fractional drop at line center relative to continuum.
        n_features (int): Number of random absorption features to generate if features is None.
        meta (dict, optional): Metadata dictionary to update.
    Returns:
        wavelength (torch.Tensor): Same as input.
        new_flux (torch.Tensor): Flux with absorption features applied.
        meta (dict): Updated metadata.
    """
    new_flux = flux.clone()
    # Determine baseline continuum level for depth reference
    baseline = 1.0
    if meta is not None and 'continuum_level' in meta:
        baseline = meta['continuum_level']
    # If no specific features provided, generate random ones
    if features is None:
        features = []
        wave_min, wave_max = wavelength[0], wavelength[-1]
        for i in range(n_features):
            center = float(torch.rand(1) * (wave_max - wave_min) + wave_min)  # avoid edges by a few Å
            fwhm = float(torch.rand(1) * 1.5 + 0.5)   # FWHM between 0.5 and 2 Å (narrow)
            depth = float(torch.rand(1) * 0.3 + 0.1)  # 10% to 40% absorption depth
            features.append({'center': center, 'fwhm': fwhm, 'depth': depth})
    # Subtract Gaussian absorption profiles
    for feat in features:
        center = feat['center']
        depth = feat['depth']
        sigma = feat['fwhm'] / 2.355
        # Gaussian profile normalized to 1 at center
        profile = torch.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
        # Subtract flux (depth * baseline * profile at each point)
        new_flux -= depth * baseline * profile
    # Update metadata
    if meta is not None:
        meta['corruptions'].append({'type': 'absorption_features', 'features': features})
    return wavelength, new_flux, meta

def add_flux_calibration_error(wavelength, flux, amplitude=0.1, meta=None):
    """
    Apply a smooth polynomial (here linear) distortion to the flux (simulating calibration error).
    Parameters:
        amplitude (float): Fractional variation across the spectrum (e.g., 0.1 for ±10%).
        meta (dict, optional): Metadata dictionary to update.
    Returns:
        wavelength (torch.Tensor): Same as input.
        new_flux (torch.Tensor): Flux after applying calibration distortion.
        meta (dict): Updated metadata.
    """
    lam_min, lam_max = wavelength[0], wavelength[-1]
    lam_mid = 0.5 * (lam_min + lam_max)
    half_range = 0.5 * (lam_max - lam_min)
    # Define a linear polynomial scale: 1 ± amplitude across the range
    scale = 1 + amplitude * ((wavelength - lam_mid) / half_range)
    new_flux = flux * scale
    # Record in metadata
    if meta is not None:
        meta['corruptions'].append({'type': 'flux_calibration_error', 'amplitude': amplitude})
    return wavelength, new_flux, meta

def apply_redshift(wavelength, flux, delta_z=0.001, meta=None):
    """
    Apply a slight redshift/wavelength calibration error to the spectrum.
    Parameters:
        delta_z (float): Fractional shift in wavelength (e.g., 0.001 for 0.1% shift).
        meta (dict, optional): Metadata dictionary to update.
    Returns:
        new_wavelength (torch.Tensor): Shifted wavelength array.
        flux (torch.Tensor): Same flux array (values unchanged).
        meta (dict): Updated metadata.
    """
    new_wavelength = wavelength * (1 + delta_z)
    new_flux = flux  # flux remains the same values, only wavelength scale shifts
    if meta is not None:
        meta['corruptions'].append({'type': 'redshift_perturbation', 'delta_z': delta_z})
    return new_wavelength, new_flux, meta


def qso_model(wavelength, mean_spectra=None, std_spectra=None, downsample_factor=1, \
                   inv_delta_loglambda=20., vary_delta_loglambda=0.05, max_snr=30, min_snr=10):

    loglamb = torch.log(wavelength)
    delta_loglambda = loglamb[1] - loglamb[0]    

    def get_radius(idloglamb):
        sigma_lambda = 1/ idloglamb
        sigma_pixels = sigma_lambda / delta_loglambda
        radius = torch.ceil(4 * sigma_pixels)
        return radius
    radius = get_radius(inv_delta_loglambda)
    print("radius of the kernel corresponding to this resolution: ", radius.item())


    def qso_corruption_single(flux, delta_z, n_features, amp, idloglamb, snr, z_noise, radius):
        # _, flux, meta = apply_redshift(wavelength, flux, delta_z=delta_z)
        # _, flux, meta = add_absorption_features(wavelength, flux, n_features=n_features.item(), meta=meta)
        _, flux, meta = add_flux_calibration_error(wavelength.to(flux.device), flux, amplitude=amp)        
        _, flux, meta = degrade_resolution(wavelength.to(flux.device), flux, dloglambda=1/idloglamb, radius=radius)
        # _, flux, meta = add_gaussian_noise(wavelength.to(flux.device), flux, SNR=snr)        
        noise_stddev = torch.mean(flux) / snr  + 1e-4 # Estimate noise standard deviation based on SNR
        flux = flux + z_noise * noise_stddev
        return flux


    def fwd(flux, return_latents = False, generator=None):

        device = flux.device
        was_2d = False
        if flux.ndim == 2: # Add batch dimension if not present
            was_2d = True
            flux = flux.unsqueeze(0)
        if downsample_factor != 1:
            flux = PixelShuffle1D(downsample_factor)(flux)
        assert flux.shape[1] == 1
        flux = flux.squeeze(1)

        if mean_spectra is not None:
            flux = flux*std_spectra + mean_spectra
        batch_size = flux.shape[0]

        # generate random parameters
        delta_z = torch.rand(batch_size, device=device, generator=generator)*0.01 - 0.005    # U(-0.005,0.005)
        n_features = torch.randint(0, 5, (batch_size, ), device=device, generator=generator)
        amp = torch.rand(batch_size, device=device, generator=generator)*0.1 - 0.05          # U(-0.05, 0.05)
        scatter_idloglamb = (torch.rand(batch_size, device=device, generator=generator) - 0.5)*2*vary_delta_loglambda
        idloglamb = (1 + scatter_idloglamb) * inv_delta_loglambda
        snr = torch.rand(batch_size, device=device, generator=generator)*(max_snr - min_snr) + min_snr
        radius = int(get_radius(idloglamb).max().item())
        z_noise = torch.randn(flux.shape, device=device, generator=generator)

        #vmap
        new_flux = torch.vmap(qso_corruption_single, (0, 0, 0, 0, 0, 0, 0, None),\
             randomness='different')(flux, delta_z, n_features, amp, idloglamb, snr, z_noise, radius)

        if mean_spectra is not None:
            new_flux = (new_flux - mean_spectra)/std_spectra
        new_flux = new_flux.unsqueeze(1)
        if downsample_factor != 1:
            new_flux = PixelUnShuffle1D(downsample_factor)(new_flux)

        if was_2d:
            new_flux = new_flux.squeeze(0)
            
        if return_latents:
            return new_flux, new_flux
        else:
            return new_flux

    return fwd


class qso_dataloader(torch.nn.Module):
    def __init__(self, wavelength, ds, batch_size=32, downsample=1, mean_spectra=None, std_spectra=None, generator=None):
        super().__init__()
        self.wavelength = wavelength
        self.ds = ds
        self.bs = batch_size
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True, generator=generator)
        self.dl = infinite_dataloader(dl)
        self.downsample = downsample
        self.mean_spectra = mean_spectra
        self.std_spectra = std_spectra

    def transform(self, x):
        x = x.unsqueeze(1)
        x = PixelUnShuffle1D(self.downsample)(x)
        return x
        
    def inv_transform(self, x):
        x = PixelShuffle1D(self.downsample)(x)
        x = x.squeeze(1)
        return x
    
    def norm(self, s):
        return (s - self.mean_spectra)/self.std_spectra
    
    def unnorm(self, s):
        return (s * self.std_spectra + self.mean_spectra)
        
    def forward(self, n_samples):
        assert n_samples %self.bs == 0 
        n = n_samples//self.bs - 1
        x = []
        for i, s in enumerate(self.dl):
            x.append(s)
            if i == n: break
        x = torch.cat(x, dim=0)
        return self.transform(x)
        

def qso_callback(idx, b, deconvolver, dataloader, device, results_folder, losses=None, qdataloader=None, validation_data=None, s=None):

    err, err2 = 0, 0
    ns, bs = 10, 256
    for _ in range(ns):
        data, obs, latents = next(dataloader)
        data, obs = data.to(device), obs.to(device)
        clean = deconvolver.transport(b, obs, None)
        err += ((data - obs)**2).sum().item()
        err2 += ((data - clean)**2).sum().item()
    err, err2 = (err/(ns*bs))**0.5, (err2/(ns*bs))**0.5
    rmse_file = os.path.join(results_folder, f"rmse.npy")
    if os.path.exists(rmse_file): 
        rmse = np.load(rmse_file).tolist()
        rmse.append([err, err2])
    else:
        rmse = [[err, err2]]
    np.save(rmse_file, np.array(rmse))
    print(f"RMSE of corrpted and clean samples is : {err:.4f}, {err2:.4f}")

    wavelength = grab(qdataloader.wavelength)
    try:
        fig, ax = plt.subplots(3, 1, figsize=(9,7), sharex=True)
        for j in range(3):
            i = np.random.randint(data.shape[0])
            ax[j].plot(wavelength, grab((qdataloader.inv_transform(data)[i])), 'k', lw=1.5, label="True")
            ax[j].plot(wavelength, grab((qdataloader.inv_transform(obs)[i])), 'r', alpha=0.7, ls='-', lw=1, label="Corrupted")
            ax[j].plot(wavelength, grab((qdataloader.inv_transform(clean)[i])), 'g', lw=2, alpha=0.5, ls="--", label="Cleaned")
            ax[j].set_ylabel('Flux')
            ax[j].grid(lw=0.3)

        plt.legend()
        plt.xlabel("Wavelength")
        plt.savefig(os.path.join(results_folder, f"sample-{idx}.png"), dpi=300)
        for axis in ax:
            axis.set_xlim(4300, 5500)
        plt.savefig(os.path.join(results_folder, f"sample-zoom-{idx}.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(e)
            
