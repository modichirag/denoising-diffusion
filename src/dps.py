import torch
import numpy as np

def grad_and_value(x_prev, x0, fwd_func, data, latents=None):
    difference = data - fwd_func(x0, latents=latents)
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, create_graph=False)[0]
    return norm_grad, norm


def edm_sampler_dps(net, latents, fwd_func, data, data_latents=None, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, edm_sigma_min=0.002,
    rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    extrap_to_zero_time=True,
    randn_like=torch.randn_like, verbose=False, conditioning_scale=1.0,
    ):
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if extrap_to_zero_time:
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    else:
        t_steps = net.round_sigma(t_steps) #t_N = t_sigma_min

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # print(i)
        x_cur = x_next.detach().requires_grad_(True)

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # DPS update here
        # norm_grad, norm = grad_and_value(x_prev=x_cur, x0=denoised.to(torch.float32), \
        #                                 fwd_func=fwd_func, data=data, latents=data_latents)
        difference = data - fwd_func(denoised.to(torch.float32), latents=data_latents)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_cur)[0]
        x_next -= norm_grad * conditioning_scale
        # torch.cuda.empty_cache()  # Use only for debugging; can slow down training
        del x_cur, denoised, d_cur , norm_grad, norm, difference

    return x_next.to(torch.float32)

