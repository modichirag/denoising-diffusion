import torch
import math
from networks import MLPResNet, PositionalEmbedding
import numpy as np

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

    def __init__(self, push_fwd, use_latents=False, n_steps=80, alpha=1.0, resamples=1, diffusion_coeff=0.0, gamma_scale=0.0, sampler='euler', randomize_time=False, n_transports=1):
        super().__init__()
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1 / self.n_steps
        self.sqrt_delta_t = self.delta_t**0.5
        self.randomize_time = randomize_time
        self.use_latents = use_latents
        self.alpha = alpha
        self.resamples = resamples
        self.diffusion_coeff = diffusion_coeff
        self.gamma_scale = gamma_scale
        self.sampler = sampler
        self.n_transports = n_transports
        if sampler == 'heun':
            print("Using heun sampler")
        if self.diffusion_coeff == 'gamma':
            print("Diffusion coeff set to gamma scale at all times")
        elif self.diffusion_coeff > 0.25 * self.gamma_scale:
            print("WARNING: diffusion_coeff is larger than 0.25 * gamma_scale, maximum noise during training.")
        if use_latents:
            print("Using latents for deonvolving")

            
    def loss_fn(self, b, x, latent=None, x0=None, b_fixed=None, s=None, s_fixed=None):
        loss = 0.
        s_loss = 0.

        if x0 is None: # x0 is the cleandata, use if provided
            b_transport = b_fixed if b_fixed is not None else b
            s_transport = s_fixed if s_fixed is not None else s
            # if self.sampler == 'euler':
            #     x0 = self.transport(b_transport, x, latent=latent, s=s_transport)
            # elif self.sampler == 'heun':
            #     x0 = self.transport_heun(b_transport, x, latent=latent, s=s_transport)
            transport = self.transport if self.sampler == 'euler' else self.transport_heun
            x0 = []
            for i in range(self.n_transports):
                x0.append(transport(b_transport, x, latent=latent, s=s_transport))
            x0 = torch.concat(x0, axis=0)
            x = torch.concat([x for _ in range(self.n_transports)], axis=0)
            if latent is not None:
                latent = torch.concat([latent for _ in range(self.n_transports)], axis=0)

        batch_size = x.shape[0]            
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
                if return_velocity:
                    vel_all.append(v)
                Xt_prev -= v * self.delta_t
                if s is not None:
                    if (type(self.diffusion_coeff) == float) or (type(self.diffusion_coeff) == int):
                        Xt_prev -= s(Xt_prev, ti, latent) / (self.gamma_scale * (ti_scalar) * (1-ti_scalar) + 1e-3) * self.diffusion_coeff * self.delta_t # score term
                        Xt_prev += math.sqrt(2. * self.diffusion_coeff) * self.sqrt_delta_t*torch.randn(x.shape).to(x.device) # diffusion term
                    elif self.diffusion_coeff == 'gamma':
                        diffusion_coeff = self.gamma_scale * (ti_scalar) *  (1.0 - ti_scalar)
                        Xt_prev -= s(Xt_prev, ti, latent) *  self.delta_t # score term
                        Xt_prev += math.sqrt(2. * diffusion_coeff) * self.sqrt_delta_t*torch.randn(x.shape).to(x.device) # diffusion term
                if return_trajectory:
                    traj.append(Xt_prev)
            Xt_final = Xt_prev

        base_state = traj if return_trajectory else Xt_final
        if return_velocity:
            return base_state, vel_all
        else:
            return base_state

        
    def transport_heun(self, b, x, latent=None, s=None, return_trajectory=False, return_velocity=False):
        traj = [x]
        vel_all = []

        # helper to build the (deterministic) drift
        def drift(x, ti_scalar, latent):
            ti_tensor = torch.ones(x.shape[0]).to(x.device) * ti_scalar
            v = b(x, ti_tensor, latent)
            score =  s(x, ti_tensor, latent)
            if (type(self.diffusion_coeff) == float) or (type(self.diffusion_coeff) == int):
                score_norm = self.diffusion_coeff / (self.gamma_scale * (ti_scalar) *  (1.0 - ti_scalar) + 1e-3)
            elif self.diffusion_coeff == 'gamma':
                score_norm = 1.
            score_scaled = score * score_norm
            a = -(v + score_scaled)
            return a
        
        with torch.no_grad():
            Xt_prev = x*1.
            for i in range(1, self.n_steps+1):
                ti_scalar = 1 - (i-1) * self.delta_t
                ti = torch.ones(x.shape[0]).to(x.device) * ti_scalar
                if s is None:
                    v = b(Xt_prev, ti, latent)                    
                    Xt_prev -= v * self.delta_t
                else:
                    # first add noise. Then eval two drift. Then add avg drift to noised point.
                    z =  torch.randn(x.shape).to(x.device)
                    if (type(self.diffusion_coeff) == float) or (type(self.diffusion_coeff) == int):
                        diff_norm = math.sqrt(2. * self.diffusion_coeff) * self.sqrt_delta_t
                    elif self.diffusion_coeff == 'gamma':
                        diffusion_coeff = self.gamma_scale * (ti_scalar) *  (1.0 - ti_scalar)
                        diff_norm = math.sqrt(2. * diffusion_coeff) * self.sqrt_delta_t
                    noise_term = z*diff_norm
                    Xt_prev += noise_term
                    
                    a = drift(Xt_prev, ti_scalar, latent) #return is -ve already
                    X_pred = Xt_prev + a * self.delta_t
                    # correction term
                    ti_scalar_next = ti_scalar - self.delta_t
                    if ti_scalar_next > 0:                        
                        a_pred = drift(X_pred, ti_scalar_next, latent)
                        Xt_prev = Xt_prev + 0.5 * (a + a_pred) * self.delta_t  #a is negated already
                    else:
                        Xt_prev = X_pred
                if return_trajectory:
                    traj.append(Xt_prev)
            Xt_final = Xt_prev

        base_state = traj if return_trajectory else Xt_final
        if return_velocity:
            return base_state, vel_all
        else:
            return base_state



class DeconvolvingInterpolantCombined(torch.nn.Module):

    def __init__(self, push_fwd, use_latents=False, n_steps=80, alpha=1.0, resamples=1,  gamma_scale=0.1, sampler='euler', randomize_time=False, n_transports=1):
        super().__init__()
        print("Learning combined drift from drift + score network")
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1 / self.n_steps
        self.sqrt_delta_t = self.delta_t**0.5
        self.use_latents = use_latents
        self.alpha = alpha
        self.resamples = resamples
        self.gamma_scale = gamma_scale
        self.sampler = sampler
        self.randomize_time = randomize_time
        self.n_transports = n_transports
        if self.randomize_time:
            print("Randomize time grid")
        if self.sampler == 'heun':
            print("Using heun sampler")
        if use_latents:
            print("Using latents for deonvolving")
        
    def loss_fn(self, b, x, latent=None, x0=None, b_fixed=None, s=None, s_fixed=None):
        loss = 0.
        s_loss = 0.

        if x0 is None: # x0 is the cleandata, use if provided
            b_transport = b_fixed if b_fixed is not None else b
            #x0 = self.transport(b_transport, x, latent=latent)
            x0 = []
            for i in range(self.n_transports):
                x0.append(self.transport(b_transport, x, latent=latent))
            x0 = torch.concat(x0, axis=0)
            x = torch.concat([x for _ in range(self.n_transports)], axis=0)
            if latent is not None:
                latent = torch.concat([latent for _ in range(self.n_transports)], axis=0)
                
        batch_size = x.shape[0]            
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
            z = torch.randn(x0.shape).to(x.device)
            It = (1-t)*x0 + t*x1 + self.gamma_scale * t*(1-t) * z
            v_true = x1 - x0 + self.gamma_scale * (1-2*t) * z
            vt   = b(It, torch.squeeze(t), latent1)
            loss += torch.mean((vt - v_true - z)**2) #extra z for score
            
        return loss / self.resamples, None  # s_loss is None                                                                                                          

    
    def transport(self, b, x, latent=None, return_trajectory=False, return_velocity=False, s=None):
        traj = [x]
        vel_all = []
        if self.randomize_time:
            times = sorted(list(np.random.uniform(0, 1, self.n_steps-1)), reverse=True)
            times = [1.0] + times + [0.0]
            for i in range(len(times) - 1):
                dt = times[i] - times[i + 1]
                assert dt > 0, f"Non-positive dt at i={i}: {dt}"
        else:
            times = [1 - (i-1) * self.delta_t for i in range(1, self.n_steps+1)]
        with torch.no_grad():
            Xt_prev = x*1.
            for i in range(len(times)-1):
                ti_scalar = times[i]
                ti_scalar_next = times[i+1]
                delta_t = ti_scalar - ti_scalar_next #reversed
                sqrt_delta_t = delta_t ** 0.5
                ti = (torch.ones(x.shape[0]) * ti_scalar).to(x.device)
                z = torch.randn(x.shape).to(x.device)    
                diffusion_coeff = self.gamma_scale * (ti_scalar) * (1-ti_scalar) 
                
                if self.sampler == 'euler':
                    v = b(Xt_prev, ti, latent)
                    Xt_prev -= v * delta_t
                    Xt_prev += math.sqrt(2. * diffusion_coeff) * sqrt_delta_t* z  # diffusion term
                    
                elif self.sampler == 'heun':
                    Xt_prev += math.sqrt(2. * diffusion_coeff) * sqrt_delta_t* z
                    v = b(Xt_prev, ti, latent)
                    X_pred = Xt_prev - v * delta_t
                    # correction term
                    ti_next = (torch.ones(x.shape[0]) * ti_scalar_next).to(x.device)
                    if ti_scalar_next > 0:                        
                        v_pred = b(X_pred, ti_next, latent)
                        Xt_prev = Xt_prev - 0.5 * (v + v_pred) * delta_t
                
            Xt_final = Xt_prev

        return Xt_final




class DeconvolvingInterpolantAWGN(torch.nn.Module):

    def __init__(self, push_fwd, use_latents=False, n_steps=80, alpha=1.0, resamples=1, diffusion_coeff='gamma', gamma_scale=0.0, sampler='euler', noise_scale=0.1):
        super().__init__()
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1 / self.n_steps
        self.sqrt_delta_t = self.delta_t**0.5
        self.use_latents = use_latents
        self.alpha = alpha #not used
        self.resamples = resamples
        self.diffusion_coeff = diffusion_coeff 
        self.gamma_scale = gamma_scale #not used
        self.sampler = sampler
        self.noise_scale = noise_scale
        if sampler == 'heun':
            raise NotImplementedError
        if self.diffusion_coeff == 'gamma':
            print("Diffusion coeff set to gamma scale at all times")
        elif self.diffusion_coeff > 0.25 * self.gamma_scale:
            print("WARNING: diffusion_coeff is larger than 0.25 * gamma_scale, maximum noise during training.")
            
    def loss_fn(self, b, x, latent=None, x0=None, b_fixed=None, s=None, s_fixed=None):
        loss = 0.
        s_loss = 0.

        if x0 is None: # x0 is the cleandata, use if provided
            b_transport = b_fixed if b_fixed is not None else b
            s_transport = s_fixed if s_fixed is not None else s
            transport = self.transport if self.sampler == 'euler' else self.transport_heun
            x0 = transport(b_transport, x, latent=latent, s=s_transport)
            if latent is not None:
                raise NotImplementedError

        batch_size = x.shape[0]
        
        for i in range(self.resamples):
            x1, latent1 = self.push_fwd(x0, return_latents=True) #don't really use it.
            latent1 = None
            # # pick data with probabability 1-alpha --- don't implement for simplicity
            # raw_mask = torch.bernoulli(torch.full((batch_size,), self.alpha)).to(x.device)
            # mask = raw_mask.view(batch_size, *([1] * (x.ndim - 1)))
            # x1 = x1 * mask + x * (1 - mask)

            # proceed as before
            t = torch.rand(x.shape[0]).to(x.device)
            new_shape = [-1] + [1] * (x.ndim - 1)
            t = t.reshape(new_shape)
            z = torch.randn(x0.shape).to(x.device)
            It = x0 + t * self.noise_scale*z #basically interpolant is 1*x0 + 0*x1 + t*z*noise_scale 
            v_true =  self.noise_scale*z
            if s is not None:
                st = s(It, torch.squeeze(t), latent1)
                s_loss += torch.mean((st - z)**2)
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
                if return_velocity:
                    vel_all.append(v)
                Xt_prev -= v * self.delta_t
                if s is not None:
                    if (type(self.diffusion_coeff) == float) or (type(self.diffusion_coeff) == int):
                        raise NotImplementedError
                        # Xt_prev -= s(Xt_prev, ti, latent) / (self.gamma_scale * (ti_scalar) * (1-ti_scalar) + 1e-3) * self.diffusion_coeff * self.delta_t # score term
                        # Xt_prev += math.sqrt(2. * self.diffusion_coeff) * self.sqrt_delta_t*torch.randn(x.shape).to(x.device) # diffusion term
                    elif self.diffusion_coeff == 'gamma':
                        diffusion_coeff = self.noise_scale * (ti_scalar) #change the schedule
                        Xt_prev -= s(Xt_prev, ti, latent) *  self.delta_t # score term
                        Xt_prev += math.sqrt(2. * diffusion_coeff) * self.sqrt_delta_t*torch.randn(x.shape).to(x.device) # diffusion term
                if return_trajectory:
                    traj.append(Xt_prev)
            Xt_final = Xt_prev

        base_state = traj if return_trajectory else Xt_final
        if return_velocity:
            return base_state, vel_all
        else:
            return base_state



    

class DeconvolvingInterpolantFollmer(torch.nn.Module):

    def __init__(self, push_fwd, use_latents=False, n_steps=80, alpha=1.0, resamples=1, diffusion_coeff=0.0, gamma_scale=0.0):
        super().__init__()
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1 / self.n_steps
        self.sqrt_delta_t = self.delta_t**0.5
        self.use_latents = use_latents
        self.alpha = alpha
        self.resamples = resamples
        self.diffusion_coeff = diffusion_coeff
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
            It = (1-t)*x0 + t*x1 + self.diffusion_coeff * t * wt
            b_true = x1 - x0  + self.diffusion_coeff * wt
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
            It = (1-t)*x0 + t*x1 + self.diffusion_coeff * t * wt
            b_true = x1 - x0  + self.diffusion_coeff * wt
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
                Xt_prev = Xt_prev + (-(2-ti_unsqueeze) * b(Xt_prev, X_start, ti, latent) - Xt_prev + X_start) * self.delta_t + self.diffusion_coeff * self.sqrt_delta_t*torch.sqrt(1-(1-ti_unsqueeze)**2)*torch.randn(x.shape).to(x.device)
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
