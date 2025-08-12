import torch
import math
from networks import MLPResNet, PositionalEmbedding

class VelocityField(torch.nn.Module):

    def __init__(self, model, use_compile=False):
        super().__init__()
        self.model = model
        self.use_compile = use_compile

        if self.use_compile:
            self.forward = torch.compile(self.forward)

    def forward(self, x, t, latents=None):
        return self.model(x, t, latents=latents)


class MLPVelocityField(torch.nn.Module):
    # a neural network that takes x in R^d and t in [0, 1] and outputs a a value in R^d

    def __init__(self, d,  hidden_dim=512, depth=4, activation=torch.nn.SiLU, t_freq=64, latent_dim=None):
        super(MLPVelocityField, self).__init__()

        self.t_freq = t_freq
        input_dim =  d + self.t_freq
        if latent_dim is not None:
            input_dim += self.t_freq
        output_dim = d
        if latent_dim is not None:
            self.latent_net = MLPResNet(latent_dim[0], hidden_dim, t_freq, 2)
        else:
            self.latent_net = None
        self.net = MLPResNet(input_dim, hidden_dim, output_dim, depth)
        self.time_encoding = PositionalEmbedding(t_freq, max_positions=2)

    def _single_forward(self, x, t, latents=None):
        # t_encoded = fourier_encode(t, self.t_freq)  # [B, 2*n_freqs]
        t_encoded = self.time_encoding(t.unsqueeze(0)).squeeze(0)  # [B, 2*n_freqs]
        x_cond = torch.cat([x, t_encoded])
        if self.latent_net is not None:
            latents = self.latent_net(latents)
            x_cond = torch.cat([x_cond, latents])
        return self.net(x_cond)

    def forward(self, x, t, latents=None):
        if latents is None:
            latents = t*0.
        return torch.vmap(self._single_forward, in_dims=(0,0,0), out_dims=(0))(x, t, latents)


class DeconvolvingInterpolant(torch.nn.Module):

    def __init__(self, push_fwd, use_latents=False, n_steps=80, alpha=1.0, resamples=1, diffusion_coef=0.0, gamma_scale=0.0):
        super().__init__()
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1 / self.n_steps
        self.sqrt_delta_t = self.delta_t**0.5
        self.use_latents = use_latents
        self.alpha = alpha
        self.resamples = resamples
        self.diffusion_coef = diffusion_coef
        self.gamma_scale = gamma_scale
        if use_latents:
            print("Using latents for deonvolving")

    def loss_fn(self, b, x, latent=None, x0=None, b_fixed=None, s=None, s_fixed=None):
        batch_size = x.shape[0]
        loss = 0.
        s_loss = 0.
        # idx = [torch.randperm(batch_size)]

        if x0 is None: # x0 is the cleandata, use if provided
            b_transport = b_fixed if b_fixed is not None else b
            s_transport = s_fixed if s_fixed is not None else s
            x0 = self.transport(b_transport, x, latent=latent, s=s_transport)

        for i in range(self.resamples):
            x1, latent1 = self.push_fwd(x0, return_latents=True)
            latent1 = latent1 if self.use_latents else None
            # x1 = x1[idx]
            # latent1 = latent1[idx] if latent is not None else None
            # pick data with probabability 1-alpha
            raw_mask = torch.bernoulli(torch.full((batch_size,), self.alpha)).to(x.device)
            mask = raw_mask.view(batch_size, *([1] * (x.ndim - 1)))
            x1 = x1 * mask + x * (1 - mask)
            if latent1 is not None:
                mask = raw_mask.view(batch_size, *([1] * (latent1.ndim - 1)))
                latent1 = latent1 * mask + latent * (1 - mask)

            # proceed as before
            t = torch.rand(x.shape[0]).to(x.device)
            new_shape = [-1] + [1] * (x.ndim - 1)
            t = t.reshape(new_shape)
            if self.gamma_scale != 0:
                z = torch.randn(x0.shape).to(x.device)
                It = (1-t)*x0 + t*x1 + self.gamma_scale * t*(1-t) * z
                v_true = x1 - x0 + self.gamma_scale * (1-2*t) * z
                if s is not None:
                    st = s(It, torch.squeeze(t), latent1)
                    s_loss += torch.mean((st - z)**2)
            else:
                It = (1-t)*x0 + t*x1
                v_true = x1 - x0
            vt   = b(It, torch.squeeze(t), latent1)
            loss += torch.mean((vt - v_true)**2)

        if s is not None:
            return loss / self.resamples, s_loss / self.resamples
        else:
            return loss / self.resamples, None  # s_loss is None

    def transport(self, b, x, latent=None, s=None, return_trajectory=False, return_velocity=False):
        traj = [x]
        vel_all = []
        with torch.no_grad():
            Xt_prev = x*1.
            for i in range(1, self.n_steps+1):
                ti_scalar = 1 - (i-1) * self.delta_t
                ti = (torch.ones(x.shape[0]) - (i-1) *self.delta_t).to(x.device)
                v = b(Xt_prev, ti, latent)
                vel_all.append(v)
                Xt_prev -= v * self.delta_t
                if s is not None:
                    Xt_prev -= s(Xt_prev, ti, latent) / (self.gamma_scale * (ti_scalar) * (1-ti_scalar) + 1e-3) * self.diffusion_coef * self.delta_t # score term
                    Xt_prev += math.sqrt(2. * self.diffusion_coef) * self.sqrt_delta_t*torch.randn(x.shape).to(x.device) # diffusion term
                if return_trajectory:
                    traj.append(Xt_prev)
            Xt_final = Xt_prev

        base_state = traj if return_trajectory else Xt_final
        if return_velocity:
            return base_state, vel_all
        else:
            return base_state



class DeconvolvingInterpolantFollmer(torch.nn.Module):

    def __init__(self, push_fwd, use_latents=False, n_steps=80, alpha=1.0, resamples=1, diffusion_coef=0.0, gamma_scale=0.0):
        super().__init__()
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1 / self.n_steps
        self.sqrt_delta_t = self.delta_t**0.5
        self.use_latents = use_latents
        self.alpha = alpha
        self.resamples = resamples
        self.diffusion_coef = diffusion_coef
        self.gamma_scale = gamma_scale
        if use_latents:
            print("Using latents for deonvolving")


    def loss_fn(self, b, x, latent=None):
        x0 = self.transport_follmer(b, x, latent=latent)
        batch_size = x.shape[0]
        loss = 0.
        for i in range(self.resamples):
            x1, latent1 = self.push_fwd(x0, return_latents=True)
            latent1 = latent1 if self.use_latents else None
            # pick data with probabability 1-alpha
            raw_mask = torch.bernoulli(torch.full((batch_size,), self.alpha)).to(x.device)
            mask = raw_mask.view(batch_size, *([1] * (x.ndim - 1)))
            x1 = x1 * mask + x * (1 - mask)
            if latent1 is not None:
                mask = raw_mask.view(batch_size, *([1] * (latent1.ndim - 1)))
                latent1 = latent1 * mask + latent * (1 - mask)
            # proceed as before
            t = torch.rand(x.shape[0]).to(x.device)
            new_shape = [-1] + [1] * (x.ndim - 1)
            t = t.reshape(new_shape)
            # the diffsuion term is different from Eric's notebook due to swtich of ends
            wt = torch.sqrt(1-t)*torch.randn(x.shape).to(x.device)
            It = (1-t)*x0 + t*x1 + self.diffusion_coef * t * wt
            b_true = x1 - x0  + self.diffusion_coef * wt
            bt   = b(It, x1, torch.squeeze(t), latent1)
            loss += torch.mean((bt - b_true)**2)
        return loss / self.resamples

    def loss_fn_cleandata(self, b, x, x0, latent=None):
        # x0 = self.transport_follmer(b, x, latent=latent)
        batch_size = x.shape[0]
        loss = 0.
        for i in range(self.resamples):
            x1, latent1 = self.push_fwd(x0, return_latents=True)
            latent1 = latent1 if self.use_latents else None
            # pick data with probabability 1-alpha
            raw_mask = torch.bernoulli(torch.full((batch_size,), self.alpha)).to(x.device)
            mask = raw_mask.view(batch_size, *([1] * (x.ndim - 1)))
            x1 = x1 * mask + x * (1 - mask)
            if latent1 is not None:
                mask = raw_mask.view(batch_size, *([1] * (latent1.ndim - 1)))
                latent1 = latent1 * mask + latent * (1 - mask)
            # proceed as before
            t = torch.rand(x.shape[0]).to(x.device)
            new_shape = [-1] + [1] * (x.ndim - 1)
            t = t.reshape(new_shape)
            # the diffsuion term is different from Eric's notebook due to swtich of ends
            wt = torch.sqrt(1-t)*torch.randn(x.shape).to(x.device)
            It = (1-t)*x0 + t*x1 + self.diffusion_coef * t * wt
            b_true = x1 - x0  + self.diffusion_coef * wt
            bt   = b(It, x1, torch.squeeze(t), latent1)
            loss += torch.mean((bt - b_true)**2)
        return loss / self.resamples


    def transport_follmer(self, b, x, latent=None, return_trajectory=False):
        traj = [x]
        new_shape = [-1] + [1] * (x.ndim - 1)
        with torch.no_grad():
            X_start = x*1
            Xt_prev = X_start
            for i in range(1, self.n_steps+1):
                ti = (torch.ones(x.shape[0]) - (i-1) *self.delta_t).to(x.device)
                ti_unsqueeze = ti.reshape(new_shape)
                # change t from Eric's notebook into 1-t due to swtich of ends
                Xt_prev = Xt_prev + (-(2-ti_unsqueeze) * b(Xt_prev, X_start, ti, latent) - Xt_prev + X_start) * self.delta_t + self.diffusion_coef * self.sqrt_delta_t*torch.sqrt(1-(1-ti_unsqueeze)**2)*torch.randn(x.shape).to(x.device)
                if return_trajectory:
                    traj.append(Xt_prev)
            Xt_final = Xt_prev

        if return_trajectory:
            return traj
        else:
            return Xt_final




# import copy
# class PeriodicFrozenModel:
#     def __init__(self, model, sync_every):
#         self.model = model
#         self.fixed_model = copy.deepcopy(self._unwrap(model))
#         self.fixed_model.eval()
#         self.sync_every = sync_every
#         self.step = 0

#     def _unwrap(self, model):
#         if hasattr(model, "module"):
#             return model.module
#         return model

#     def maybe_update(self):
#         self.step += 1
#         if self.step % self.sync_every == 0:
#             self.fixed_model.load_state_dict(self._unwrap(self.model).state_dict())

#     def get(self):
#         return self.fixed_model
