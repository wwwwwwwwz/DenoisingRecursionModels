import torch
import random
from diffusion.correlated_noise import get_correlated_noise
import math

class DiscreteScheduler():
    
    def __init__(self, num_train_timesteps=1000, num_inference_timesteps=20, schedule="linear", noise_type='uniform', confidence_masking=False):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.schedule = schedule
        self.noise_type = noise_type
        self.confidence_masking = confidence_masking

    def get_mask_ratios(self, time_steps: torch.Tensor):

        if self.schedule == 'linear':
            return time_steps/self.num_train_timesteps
        elif self.schedule == 'cosine':
            s=0.008
            x = (time_steps / self.num_train_timesteps + s) / (1+s)
            x = x.clamp(0,1)
            bar_alpha = torch.cos(0.5 * math.pi * x) ** 2
            return 1 - bar_alpha

        elif self.schedule == 'sigmoid':
            # Typical ranges: -3 to 3 (balanced), or -5 to 3 (extends low noise)
            start, end = -5, 3 
            t = time_steps / self.num_train_timesteps
            
            # Map time to the sigmoid range
            v_start = torch.tensor(start, device=time_steps.device)
            v_end = torch.tensor(end, device=time_steps.device)
            vt = v_start + t * (v_end - v_start)
            
            # Sigmoid function: 1 / (1 + exp(-x))
            sig = 1 / (1 + torch.exp(-vt))
            
            # Normalize so it starts strictly at 0 and ends strictly at 1
            sig_start = 1 / (1 + torch.exp(-v_start))
            sig_end = 1 / (1 + torch.exp(-v_end))
            x = (sig - sig_start) / (sig_end - sig_start)
            
            return x  # Returns the noise level (1 - bar_alpha) directly

    # def get_timesteps(self, num_inference_steps=1):
    #     return sorted(random.sample(range(1, self.num_train_timesteps + 1), num_inference_steps), reverse=True)
    
    def get_timesteps(self, skew_exponent=2.0):
        return torch.flip((torch.linspace(0, 1, self.num_inference_timesteps) ** skew_exponent / torch.linspace(0, 1, self.num_inference_timesteps).pow(skew_exponent).max() * (self.num_train_timesteps - 1) + 1).long(), dims=[0]).tolist()

    def get_noise(self, labels: torch.Tensor, device):

        if self.noise_type == 'uniform':
            return torch.rand(labels.shape, device=device) # Uniform over 0,1
        elif self.noise_type == 'spatially_correlated':
            return get_correlated_noise(labels, device).squeeze()