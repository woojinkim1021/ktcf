import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2

import torch
import random
import numpy as np
import pandas as pd
import copy
import json
import pickle
import argparse
import ast
import csv
from datetime import datetime

from ktcf.cf_generator import KTCF, WachterCF, DiCECF
from ktcf.init_cf import initialize_response
from ktcf.eval_metrics import eval_validity, eval_sparsity, eval_sparsity_rate, eval_kc_distance, eval_actionability

from pykt.models import load_model

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

# Set all seeds
set_all_seeds(42)

device = "cpu" if not torch.cuda.is_available() else "cuda"
config_path = "./saved_model/config.json"
ckpt_path = "./saved_model/qid_model.ckpt"
batch_size = 256
save_dir = './saved_model'

with open(config_path) as fin:
    config = json.load(fin)
    model_config = copy.deepcopy(config["model_config"])
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
        if remove_item in model_config:
            del model_config[remove_item]    
    trained_params = config["params"]
    fold = trained_params["fold"]
    model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]

with open("./configs/data_config.json") as fin:
    curconfig = copy.deepcopy(json.load(fin))
    data_config = curconfig[dataset_name]
    data_config["dataset_name"] = dataset_name
    data_config['dpath'] = './data/XES3G5M'


model = load_model(model_name, model_config, data_config, emb_type, save_dir)


network_path = "./data/XES3G5M/metadata/kc_network.pkl"
with open(network_path, 'rb') as f:
    kc_net = pickle.load(f)
kc_net = kc_net.to_undirected()

kc_info = pd.read_csv('./data/XES3G5M/metadata/kc_nodes.csv')

with open("./data/XES3G5M/metadata/shortest_paths.pkl", 'rb') as f:
    shortest_info = pickle.load(f)


test_predictions = pd.read_csv('./data/XES3G5M/XES3G5M_test_predictions_45_zeros.csv').sample(n=200, random_state=42)
test_size = test_predictions.shape[0]


def evaluate(alg, original_q, original_r, cf_r, pred_at_target, target_timestep, node_info, shortest_info):
    validity = eval_validity(pred_at_target, alg, threshold=0.5)
    sparsity = eval_sparsity(cf_r, original_r)
    sparsity_rate = eval_sparsity_rate(cf_r, original_r)
    kc_distance = eval_kc_distance(original_q, original_r, cf_r, target_timestep, node_info, shortest_info)
    actionability = eval_actionability(cf_r, original_r)
    return validity, sparsity, sparsity_rate, kc_distance, actionability


def main(params):
    validity = []
    sparsity = []
    sparsity_rate = []
    kc_distance = []
    actionability = []
    actionability_rate = []
    time = []

    params['verbose'] = True
    params['save_result'] = True

    if params['alg'] == 'ktcf':
        ktcf = KTCF(model)
    elif params['alg'] == 'wachter':
        wachtercf = WachterCF(model)
    elif params['alg'] == 'dice':
        dicecf = DiCECF(model)
    
    idx = 0
    for i, row in test_predictions.iterrows():
        concept, response, concept_shift, target_timestep = row.concept, row.response, row.concept_shift, row.target_timestep
        concept, response, concept_shift = torch.tensor(ast.literal_eval(concept)).unsqueeze(0).to(device), torch.tensor(ast.literal_eval(response)).unsqueeze(0).to(device), torch.tensor(ast.literal_eval(concept_shift)).unsqueeze(0).to(device)

        cf_r = initialize_response(params, response)

        if params['alg'] == 'ktcf':
            cf_r, pred_at_target, runtime = ktcf.generate_cf_dkt(params,
                concept, response, cf_r, concept_shift, target_timestep=target_timestep, node_info=kc_info, shortest_info=shortest_info,
                lr=params['lr'], max_iter=params['max_iter'],
                lambda_pred=params['lambda_pred'], lambda_prox=params['lambda_prox'], lambda_kc=params['lambda_kc'],
                early_stopping_eta=params['early_stopping_eta'], verbose=params['verbose']
            )
            
        elif params['alg'] == 'wachter':
            cf_r, pred_at_target, runtime = wachtercf.generate_cf_dkt(params,
                concept, response, cf_r, concept_shift, target_timestep=target_timestep,
                lr=params['lr'], max_iter=params['max_iter'], 
                lambda_prox=params['lambda_prox'],
                early_stopping_eta=params['early_stopping_eta'], verbose=params['verbose']
            )
        
        else: # DiCE
            cf_r_init_dice = initialize_response(params, response, K=3)
            cf_r, pred_at_target, runtime = dicecf.generate_cf_dkt(params,
                concept, response, cf_r_init_dice, concept_shift, target_timestep=target_timestep, 
                lr=params['lr'], max_iter=params['max_iter'], 
                lambda_prox=params['lambda_prox'], lambda_div=params['lambda_div'],
                early_stopping_eta=params['early_stopping_eta'], verbose=params['verbose']
            )
        
        val, spar, spar_rate, kc_dist, act = evaluate(params['alg'], concept, response, cf_r, pred_at_target, target_timestep, kc_info, shortest_info)
        validity.append(val)
        sparsity.append(spar)
        sparsity_rate.append(spar_rate)
        kc_distance.append(kc_dist)
        actionability.append(act)
        actionability_rate.append((act/spar))
        time.append(runtime)


        if params['verbose']:
            print(f'{params['alg']}, instance {idx}: validity={val}, sparsity={spar}, sparsity %={spar_rate:.6f}, kc_distance={kc_dist:.6f}, actionability={act}, actionability %{(act/spar):.6f}, time={runtime:.6f}')
        idx += 1
        if idx==1:
            break


    if params['save_result']:
        result_save_dir = './results/'
        result_save_path = os.path.join(result_save_dir, params['save_file_name'])
        
        
        header = ['datetime','alg','init_scheme','lr',
                    'init_noise_std','init_lambda_convex','init_temp',
                    'lambda_pred','lambda_prox','lambda_kc','lambda_div',
                    'validity_mean', 'validity_std',
                    'sparsity_mean', 'sparsity_std',
                    'sparsity_rate_mean', 'sparsity_rate_std',
                    'kc_distance_mean', 'kc_distance_std',
                    'actionability_mean', 'actionability_std', 
                    'actionability_rate_mean', 'actionability_rate_std', 
                    'time_mean', 'time_std']
        write_header = not (os.path.exists(result_save_path) and os.path.getsize(result_save_path) > 0)
        
        with open(result_save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
        
            writer.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"), params['alg'], params['init_scheme'], params['lr'], 
                            params['init_noise_std'], params['init_lambda_convex'], params['init_temp'], 
                            params['lambda_pred'], params['lambda_prox'], params['lambda_kc'], params['lambda_div'],
                            np.mean(validity), np.std(validity, ddof=1),
                            np.mean(sparsity), np.std(sparsity, ddof=1),
                            np.mean(sparsity_rate), np.std(sparsity_rate, ddof=1),
                            np.mean(kc_distance), np.std(kc_distance, ddof=1),
                            np.mean(actionability), np.std(actionability, ddof=1),
                            np.mean(actionability_rate), np.std(actionability_rate, ddof=1),
                            np.mean(time), np.std(time, ddof=1)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default='ktcf')
    parser.add_argument("--init_scheme", type=int, default=1) 
    parser.add_argument("--lr", type=float, default=0.01) 
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--init_noise_std", type=float, default=0.1)
    parser.add_argument("--init_lambda_convex", type=float, default=0.5)
    parser.add_argument("--init_temp", type=float, default=0.1)
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_prox", type=float, default=0.1)
    parser.add_argument("--lambda_kc", type=float, default=0.001)
    parser.add_argument("--lambda_div", type=float, default=0.001)
    parser.add_argument("--kc_mask", type=int, default=10)
    parser.add_argument("--early_stopping_eta", type=float, default=0.0001)
    parser.add_argument("--save_file_name", type=str, default='table_1_results.csv')
    

    args = parser.parse_args()
    params = vars(args)
    print(args)
    main(params)