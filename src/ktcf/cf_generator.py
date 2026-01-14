import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from scipy.spatial.distance import hamming

sig = nn.Sigmoid()

target_proba = 1.0


class BaseCFGenerator:
    def generate_counterfactual(self, *args, **kwargs):
        raise NotImplementedError


def get_kc_loss(original_q, original_r, cf_r, target_timestep, node_info, shortest_info):
    cf_r_label = (sig(cf_r.clone().squeeze(0)) > 0.5).float()
    interventions = torch.where((original_r == 0) & (cf_r_label == 1))[1].cpu().numpy()
    
    if len(interventions) == 0:
        return 0.0

    source_kc_ids = original_q[0][interventions].cpu().numpy()
    target_kc_id = original_q.squeeze()[target_timestep].item()
    
    
    id_to_name = node_info.set_index('node_id')['node_name'].to_dict()
    filtered_interventions = []
    filtered_source_kc_ids = []
    node_names = []
    for idx, kc_id in zip(interventions, source_kc_ids):
        if kc_id in id_to_name:
            filtered_interventions.append(idx)
            filtered_source_kc_ids.append(kc_id)
            node_names.append(id_to_name[kc_id])

    source_names = [id_to_name[kc_id] for kc_id in filtered_source_kc_ids]
    target_kc_name = node_info[node_info.node_id == target_kc_id].node_name.item()
    
    target = [target_kc_name] * len(filtered_interventions)
    lengths = [shortest_info[(a, b)] for a, b in zip(source_names, target)]

    kc_mask = torch.zeros(original_r.size(1), dtype=torch.long)
    for i, idx in enumerate(filtered_interventions):
        kc_mask[idx] = lengths[i]
    
    return sum(lengths)/len(interventions)


def sample_gumbel(shape, eps=1e-20, device=None):
    """
    Sample Gumbel noise of a given shape.
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


class KTCF(BaseCFGenerator):
    def __init__(self, model):
        self.model = model
        self.gumbel_noise_1 = sample_gumbel(1).item()
        self.gumbel_noise_2 = sample_gumbel(1).item()

    def generate_cf_dkt(
        self,
        params,
        original_q,
        original_r,
        cf_r,
        cshft,
        target_timestep,
        node_info,
        shortest_info,
        lr=0.1,
        max_iter=100,
        lambda_pred=10.0, 
        lambda_prox=0.1,
        lambda_kc=0.1,
        early_stopping_eta=0.0001,
        verbose=False
    ):
        device = next(self.model.parameters()).device
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        prev_loss = float('inf')
        
        optimizer = optim.Adam([cf_r], lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        mask = (original_r == 0).float().to(device)
            
        start = time.perf_counter() 
        for i in range(max_iter):
            optimizer.zero_grad()
            if params['init_scheme'] == 3:
                y = self.model(original_q, torch.sigmoid(cf_r))
            if params['init_scheme'] == 5:
                y = self.model(original_q, torch.sigmoid((cf_r + self.gumbel_noise_1 - self.gumbel_noise_2) / params['init_temp']))
            else: 
                y = self.model(original_q, cf_r) 
                
            
            y = (y * F.one_hot(cshft.long(), self.model.num_c)).sum(-1)
            pred_at_target = y[0, target_timestep]
            
            # 1. prediction loss
            loss_pred = F.binary_cross_entropy(pred_at_target, torch.tensor(target_proba, device=device))
            
            # 2. sparsity loss - Hamming distance
            if params['init_scheme'] == 3:
                loss_prox = hamming(original_r.cpu().to(torch.int32).squeeze(), (torch.sigmoid(cf_r) > 0.5).cpu().to(torch.int32).squeeze())
            elif params['init_scheme'] == 5:
                loss_prox = hamming(original_r.cpu().to(torch.int32).squeeze(), ((torch.sigmoid((cf_r + self.gumbel_noise_1 - self.gumbel_noise_2) / params['init_temp'])) > 0.5).cpu().to(torch.int32).squeeze())
            else:
                loss_prox = hamming(original_r.cpu().to(torch.int32).squeeze(), (cf_r > 0.5).cpu().to(torch.int32).squeeze())
            
            # 3. KC relation loss
            if params['init_scheme'] == 3:
                loss_kc = get_kc_loss(original_q, original_r, torch.sigmoid(cf_r), target_timestep, node_info, shortest_info)
            elif params['init_scheme'] == 5:
                loss_kc = get_kc_loss(original_q, original_r, torch.sigmoid((cf_r + self.gumbel_noise_1 - self.gumbel_noise_2) / params['init_temp']), target_timestep, node_info, shortest_info)
            else:
                loss_kc = get_kc_loss(original_q, original_r, cf_r, target_timestep, node_info, shortest_info)


            total_loss = lambda_pred * loss_pred + lambda_prox * loss_prox + lambda_kc * loss_kc
            
            if not torch.isfinite(total_loss):
                print(f"Loss is not finite at iteration {i}: {total_loss.item()}")
                break
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                cf_r.data = mask * cf_r.data + (1 - mask) * original_r.data
            
            
            if abs(prev_loss - total_loss) < early_stopping_eta:
                if verbose:
                    print(f"Converged at iteration {i}")
                break

            prev_loss = total_loss
        kt_time = time.perf_counter() - start
        return cf_r.detach(), pred_at_target, kt_time




"""
Implements 'Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. Harv. JL & Tech., 31, 841.'
"""

class WachterCF(BaseCFGenerator):
    def __init__(self, model):
        self.model = model
        self.gumbel_noise = sample_gumbel(1).item()

    def generate_cf_dkt(
        self,
        params,
        original_q,
        original_r,
        cf_r,
        cshft,
        target_timestep,
        lr=0.1,
        max_iter=100,
        lambda_prox=0.1,
        early_stopping_eta=0.0001,
        verbose=False
    ):
        device = next(self.model.parameters()).device
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        prev_loss = float('inf')
        
        optimizer = optim.Adam([cf_r], lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        start = time.perf_counter()
        for i in range(max_iter):
            optimizer.zero_grad()
            
            if params['init_scheme'] == 3:
                y = self.model(original_q, torch.sigmoid(cf_r))
            if params['init_scheme'] == 5:
                y = self.model(original_q, torch.sigmoid((cf_r + self.gumbel_noise) / params['init_temp']))
            else: 
                y = self.model(original_q, cf_r)
            
            y = (y * F.one_hot(cshft.long(), self.model.num_c)).sum(-1)
            pred_at_target = y[0, target_timestep]
            
            loss_pred = F.binary_cross_entropy(pred_at_target, torch.tensor(target_proba, device=device))

            if params['init_scheme'] == 3:
                loss_prox = torch.norm((torch.sigmoid(cf_r) - original_r), p=1)
            elif params['init_scheme'] == 5:
                loss_prox = torch.norm((torch.sigmoid((cf_r + self.gumbel_noise) / params['init_temp']) - original_r), p=1)
            else:
                loss_prox = torch.norm((cf_r - original_r), p=1)
            
            total_loss = loss_pred + lambda_prox * loss_prox
            
            if not torch.isfinite(total_loss):
                print(f"Loss is not finite at iteration {i}: {total_loss.item()}")
                break
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            
            if abs(prev_loss - total_loss) < early_stopping_eta:
                if verbose:
                    print(f"Converged at iteration {i}")
                break
        
        w_time = time.perf_counter() - start
        return cf_r.detach(), pred_at_target, w_time






def compute_proximity(cf_r, original_r, mask):
    loss_prox = torch.norm((cf_r - original_r) * mask, p=1, dim=(1,2))  # [K]
    return loss_prox.mean() 

def compute_diversity(cf_r, mask):
    K = cf_r.shape[0]
    diversity = 0.0
    count = 0
    for k1 in range(K):
        for k2 in range(k1+1, K):
            dist_r = torch.norm((cf_r[k1] - cf_r[k2]) * mask[k1], p=1)
            diversity += dist_r
            count += 1
    if count > 0:
        diversity = diversity / count
    return -diversity 

def hinge_loss(logit, target=1, margin=1.0):
    z = 2 * torch.as_tensor(target, device=logit.device, dtype=logit.dtype) - 1
    return torch.clamp(margin - z * logit, min=0)


"""
Implements 'Mothilal, R. K., Sharma, A., & Tan, C. (2020, January). Explaining machine learning classifiers through diverse counterfactual explanations. 
In Proceedings of the 2020 conference on fairness, accountability, and transparency (pp. 607-617).'
"""

class DiCECF(BaseCFGenerator):
    def __init__(self, model, num_counterfactuals=3, stopping_threshold=0.5):
        self.model = model
        self.num_counterfactuals = num_counterfactuals
        self.stopping_threshold = stopping_threshold
        self.gumbel_noise = sample_gumbel(1).item()

    def generate_cf_dkt(
        self,
        params,
        original_q,
        original_r,
        cf_r,
        cshft,
        target_timestep,
        lr=0.1,
        max_iter=100,
        lambda_prox=0.1,
        lambda_div=0.1,
        margin=1.0,
        early_stopping_eta=0.0001,
        verbose=False
    ):

        device = next(self.model.parameters()).device
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        prev_loss = float('inf')

        B, L = original_q.shape
        K = self.num_counterfactuals
        
        original_q_reshaped = original_q.repeat(K, 1, 1).float()
        
        optimizer = optim.Adam([cf_r], lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        mask = (original_q != 0).float().to(device)
        mask = mask.repeat(K, 1, 1)
        
        start = time.perf_counter()
        for i in range(max_iter):
            optimizer.zero_grad()

            K, B, L = cf_r.shape
            original_q_reshaped = original_q_reshaped.reshape(K*B, L)
            cf_r_reshaped = cf_r.reshape(K*B, L)

            if params['init_scheme'] == 3:
                y = self.model(original_q, torch.sigmoid(cf_r_reshaped))
            elif params['init_scheme'] == 5:
                y = self.model(original_q, torch.sigmoid((cf_r_reshaped + self.gumbel_noise) / params['init_temp']))
            else: 
                y = self.model(original_q_reshaped, cf_r_reshaped)  # [K*B, L, num_c]
            
                    
            y = y.reshape(K, B, L, -1)  # [K, B, L, num_c]
            y = (y * F.one_hot(cshft.long(), self.model.num_c)).sum(-1)  # [K, B, L]
            pred_at_target = y[:, 0, target_timestep]  # [K]
            

            loss_pred = hinge_loss(pred_at_target, target=target_proba, margin=margin).mean()
            
            if params['init_scheme'] == 3:
                loss_prox = compute_proximity(torch.sigmoid(cf_r), original_r, mask)
            elif params['init_scheme'] == 5:
                loss_prox = compute_proximity(torch.sigmoid((cf_r + self.gumbel_noise) / params['init_temp']), original_r, mask)
            else: 
                loss_prox = compute_proximity(cf_r, original_r, mask)
            

            if params['init_scheme'] == 3: 
                loss_div = compute_diversity(cf_r, mask)
            elif params['init_scheme'] == 5:
                loss_div = compute_diversity(cf_r, mask)
            else: 
                loss_div = compute_diversity(cf_r, mask)
                     
            
            total_loss = loss_pred + lambda_prox * loss_prox + lambda_div * loss_div
            
            if not torch.isfinite(total_loss):
                print(f"Loss is not finite at iteration {i}: {total_loss.item()}")
                break
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
        
            if (pred_at_target > self.stopping_threshold).all():
                if verbose:
                    print(f"iteration {i}, early stopping due to all diverse explanation predictions {pred_at_target.data} are over {self.stopping_threshold}.")
                break
        
            if abs(prev_loss - total_loss) < early_stopping_eta:
                if verbose:
                    print(f"Converged at iteration {i}")
                break

        d_time = time.perf_counter() - start
        return cf_r.detach(), pred_at_target, d_time