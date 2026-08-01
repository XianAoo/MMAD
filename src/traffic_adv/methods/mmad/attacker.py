import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import math

class DMADVAttacker:
    def __init__(self, diffusion, surrogate_models, config):
        """
        """
        self.diffusion = diffusion
        self.models = surrogate_models
        self.cfg = config
        self.device = config.DEVICE

    def steep_schedule(self, time, lambda_all, k=0.001):
        """Return a steep time-dependent guidance schedule."""
        T = 900
        t_scalar = time.item() if isinstance(time, torch.Tensor) else time
        if t_scalar >= T:
             return lambda_all * 0.001
        
        normalized_time = t_scalar / T
        factor = np.exp(-k * (1 / (1 - normalized_time + 1e-10) - 1))
        return lambda_all * factor

    def cosine_map(self, x):
        """Map a normalized budget to constraint strength."""
        if not (0 <= x <= 1): return 2 # Fallback
        return 1 * (1 - math.cos(x * math.pi))

    def get_dynamic_gradient(self, x, t, current_weights):
        """
        """
        grads = {}
        escape_rates = {}
        
        x_in = x.detach().requires_grad_(True)
        
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t).to(x.device)
        
        with torch.enable_grad():
            for name, model in self.models.items():
                was_training = model.training
                if name == 'LSTM': model.train()
                else: model.eval()

                logits = model(x_in, t) 

                probs = torch.sigmoid(logits)
                
                log_probs = torch.log(1 - probs + 1e-8)
                
                grad = torch.autograd.grad(log_probs.sum(), x_in)[0]
                grads[name] = grad
                
                with torch.no_grad():
                    escaped = (probs < self.cfg.ESCAPE_THRESHOLD).float().mean().item()
                    escape_rates[name] = escaped
                
                if name == 'LSTM' and not was_training: model.eval()
            
        x_in.requires_grad_(False)

        updated_weights = current_weights.copy()
        
        min_escape_model = min(escape_rates, key=lambda k: escape_rates[k])
        reduce_amount = min(self.cfg.UPDATE_K, updated_weights[min_escape_model] - 0.01)
        updated_weights[min_escape_model] -= reduce_amount
        
        remaining = [n for n in self.models if n != min_escape_model]
        sum_rem_rates = sum([escape_rates[n] for n in remaining])
        
        for n in remaining:
            if sum_rem_rates == 0:
                updated_weights[n] += reduce_amount / len(remaining)
            else:
                ratio = escape_rates[n] / sum_rem_rates
                updated_weights[n] += reduce_amount * ratio
                
        grad_all = torch.zeros_like(x)
        for name in self.models:
            grad_all += updated_weights[name] * grads[name]
            
        return grad_all, updated_weights, escape_rates

    @torch.no_grad()
    def attack(self, maldata, mask=None, desc=None):
        """Time-conditioned classifier base."""
        batch_size = maldata.shape[0]
        device = self.device
        
        total_steps = self.diffusion.num_timesteps
        sampling_steps = self.cfg.SAMPLING_TIMESTEPS
        eta = 0.0 # DDIM Deterministic
        
        times = torch.linspace(-1, total_steps - 1, steps=sampling_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        t_T = torch.full((batch_size,), total_steps - 1, device=device).long()
        eps_ori = torch.randn_like(maldata)
        img = self.diffusion.q_sample(maldata, t_T, noise=eps_ori) # x_T
        
        weights = self.cfg.INIT_WEIGHTS.copy()
        
        iterator = time_pairs
        if bool(getattr(self.cfg, "ATTACK_STEP_PROGRESS", False)):
            iterator = tqdm(
                time_pairs,
                desc=desc or "[MMAD] sampling steps",
                leave=bool(getattr(self.cfg, "ATTACK_STEP_PROGRESS_LEAVE", False)),
            )

        for time, time_next in iterator:
            t_cond = torch.full((batch_size,), time, device=device).long()
            
            preds = self.diffusion.model_predictions(img, t_cond)
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start
            
            if time_next < 0:
                img = x_start
                continue
                
            grad_adv, weights, _ = self.get_dynamic_gradient(img, t_cond, weights)
            
            with torch.enable_grad():
                img.requires_grad_(True)
                loss_mse = F.mse_loss(img, maldata)
                grad_dist = torch.autograd.grad(loss_mse, img)[0]
            if grad_dist.norm() > 0:
                grad_dist = grad_dist / grad_dist.norm()
            img.requires_grad_(False)
            
            alpha = self.diffusion.alphas_cumprod[time]
            alpha_next = self.diffusion.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            lambda_t = self.steep_schedule(time, self.cfg.LAMBDA_ALL)
            lambda_dist = self.steep_schedule(time, self.cosine_map(self.cfg.BUDGET))
            
            adv_noise = pred_noise - lambda_t * (1-alpha).sqrt() * grad_adv
            adv_noise = adv_noise - lambda_dist * (1-alpha).sqrt() * grad_dist
            
            x_start_adv = self.diffusion.predict_start_from_noise(img, t_cond, adv_noise)
            x_start_adv = x_start_adv.clamp(-3., 3.)
            
            noise_random = torch.randn_like(img)
            img = x_start_adv * alpha_next.sqrt() + c * adv_noise + sigma * noise_random
            
        img = self.diffusion.unnormalize(img)
        
        return img
    
