import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

sig = nn.Sigmoid()

def eval_validity(pred_at_target, alg, threshold=0.5):
        if alg == 'dice':
            return (pred_at_target > 0.5).all().float().item()    
        else:
            return (pred_at_target > 0.5).float().item()



def eval_sparsity(cf_r, original_r):

    cf_r = (sig(cf_r.clone()) > 0.5).float()
    if cf_r.shape == original_r.shape:
        diff_count = (original_r != cf_r).sum().item()
        return diff_count
    elif cf_r.shape[0] == 3 and cf_r.shape[1:] == (1, original_r.shape[1]):
        og_r_expanded = original_r.expand_as(cf_r)
        diff_counts = (og_r_expanded != cf_r).sum(dim=(1,2)).float() 
        # print(f'diff_counts: {diff_counts}')
        sparsity = diff_counts.mean().item()
        return sparsity
    else:
        raise ValueError(f"Unexpected shapes: og_r {original_r.shape}, cf_r {cf_r.shape}")
    


def eval_sparsity_rate(cf_r, original_r, mask=None):
    sparsity = eval_sparsity(cf_r, original_r)
    sparsity_rate = sparsity / original_r.size(1)
    return sparsity_rate



def eval_kc_distance(original_q, original_r, cf_r, target_timestep, node_info, shortest_info):
    if cf_r.shape == original_r.shape:
        cf_r_label = (sig(cf_r.clone().squeeze(0)) > 0.5).float()
        interventions = torch.where((original_r != cf_r_label))[1].cpu().tolist()

        source_kc_ids = original_q[0][interventions].cpu().numpy()
        target_kc_id = original_q.squeeze()[target_timestep].item()
        
        source_names = node_info[node_info['node_id'].isin(source_kc_ids)]['node_name'].tolist()
        target_kc_name = node_info[node_info.node_id == target_kc_id].node_name.item()
        
        target = [target_kc_name] * len(interventions)
        lengths = [shortest_info[(a, b)] for a, b in zip(source_names, target)]

        return sum(lengths)/len(interventions)

    elif cf_r.shape[0] == 3 and cf_r.shape[1:] == (1, original_r.shape[1]):
        cf_r_label = (sig(cf_r.clone().squeeze()) > 0.5).float()
        original_r_squeezed = original_r.squeeze()
        original_q_squeezed = original_q.squeeze()
        results = []
        for i in range(cf_r_label.shape[0]):
            cf_r_i = cf_r_label[i]
            original_r_i = original_r_squeezed[i] if original_r_squeezed.ndim > 1 else original_r_squeezed
            original_q_i = original_q_squeezed[i] if original_q_squeezed.ndim > 1 else original_q_squeezed

            interventions = torch.where((original_r_i != cf_r_i))[0].cpu().tolist()
            source_kc_ids = original_q_i[interventions].cpu().numpy()
            target_kc_id = original_q_i[target_timestep].item()
            source_names = node_info[node_info['node_id'].isin(source_kc_ids)]['node_name'].tolist()
            target_kc_name = node_info[node_info.node_id == target_kc_id].node_name.item()
            target = [target_kc_name] * len(interventions)
            lengths = [shortest_info[(a, b)] for a, b in zip(source_names, target)]
            results.append(sum(lengths)/len(interventions) if interventions else 0)
        return np.mean(results)
    else:
        raise ValueError(f"Unexpected shapes: og_r {original_r.shape}, cf_r {cf_r.shape}")




def eval_actionability(cf_r, original_r):

    if cf_r.dim() == 2:
        cf_r = cf_r.unsqueeze(1)
    if original_r.dim() == 1:
        original_r = original_r.unsqueeze(0).unsqueeze(0)
    elif original_r.dim() == 2:
        original_r = original_r.unsqueeze(0)

    if original_r.shape[0] == 1:
        original_r = original_r.repeat(cf_r.shape[0], 1, 1)


    cf_r_label = (cf_r > 0.5).float()
    action_mask = (original_r == 1) & (cf_r_label == 0)
    nonactionables = action_mask.sum(dim=(1, 2)).float().mean().item()
    return nonactionables