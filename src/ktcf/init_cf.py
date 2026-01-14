import torch
import numpy as np
import random

def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

set_all_seeds(42)

device = "cpu" if not torch.cuda.is_available() else "cuda"


def initialize_response(params, original_r, K=0):
    torch.cuda.manual_seed(42)

    if params['alg'] == 'dice':
        n_repeat = 3
    else:
        n_repeat = 1

    results = []
    for _ in range(n_repeat):
        
        if params['init_scheme'] == 1: # add random noise to original
            cf_r = original_r.clone().float().detach() + params['init_noise_std'] * torch.randn_like(original_r.float())
        
        elif params['init_scheme'] == 2: # completely random cf_r
            cf_r = torch.randint(0, 2, (original_r.size(1),)).float().unsqueeze(0).to(device)
        
        elif params['init_scheme'] == 3: # soft relax -> implemented inside algorithm
            cf_r = torch.randn_like(original_r, requires_grad=True)
        
        elif params['init_scheme'] == 4: # Convex Combination Initialization
            random_r = torch.randint(0, 2, original_r.shape, dtype=original_r.dtype, device=original_r.device).float()
            cf_r = params['init_lambda_convex'] * original_r + (1 - params['init_lambda_convex']) * random_r
        
        elif params['init_scheme'] == 5: # Gumbel-Softmax Relaxation -> implemented inside algorithm
            # cf_r = torch.randn_like(original_r, requires_grad=True)
            cf_r = torch.randn_like(original_r, requires_grad=True)
            
    
        results.append(cf_r)
    
    if n_repeat == 1:
        return results[0].requires_grad_(True)
    else:
        return torch.nn.Parameter(torch.stack(results, dim=0))


