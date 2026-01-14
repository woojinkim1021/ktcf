import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2

import json
import pickle
import torch
import pandas as pd
import argparse
import networkx as nx
import pandas as pd
import ast
import numpy as np
import random
from ktcf.cf_generator import KTCF, WachterCF, DiCECF
from ktcf.init_cf import initialize_response
from pykt.models.init_model import init_model
from itertools import combinations


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


def tspp_with_fixed_target(graph, target_kc_name):
    if target_kc_name not in graph.nodes:
        raise ValueError(f"Target KC '{target_kc_name}' not in graph")
    visited = [target_kc_name]
    current = target_kc_name
    total_cost = 0

    while len(visited) < len(graph.nodes):
        neighbors = [
            (neighbor, data["weight"])
            for neighbor, data in graph[current].items()
            if neighbor not in visited
        ]
        if not neighbors:
            break

        next_node, cost = min(neighbors, key=lambda x: x[1])
        visited.append(next_node)
        total_cost += cost
        current = next_node

    path = list(reversed(visited))
    print(f"[TSP] ! Best path found: {path}")

    return path, total_cost
    


def export_tspp_cf_indices(args, model, test_predictions, kc_info, shortest_info):
    results_dir = os.path.join("./results/")
    os.makedirs(results_dir, exist_ok=True)

    kc_df = pd.read_csv("./data/XES3G5M/metadata/kc_nodes.csv")
    id_to_name = dict(zip(kc_df['node_id'], kc_df['node_name']))
    name_to_id = dict(zip(kc_df['node_name'], kc_df['node_id']))

    with open("./data/XES3G5M/metadata/kc_network.pkl", "rb") as f:
        original_graph = pickle.load(f)
        graph = nx.Graph()
        graph.add_edges_from(original_graph.edges()) 

    device = next(model.parameters()).device
    params = {  'alg': 'ktcf',
                'init_scheme':1,
                'init_noise_std':0.1,
                'lr': 0.01,
            }

    df = pd.DataFrame(columns=["alg", "index", "batch", "instance", "common_wrong_indices", "common_wrong_kc_ids", "common_wrong_kc_names", "unactionables_kc_names",
    "original_path_distance", "original_path_total_distance", "origin_tspp_path_ids", "origin_tspp_path_names", "origin_tspp_distances", 
    "origin_tspp_total_distance", "filtered_kc_ids", "filtered_kc_names", "filtered_path_distances", "filtered_path_total_distance", 
    "filtered_tsp_kc_id", "filtered_tsp_kc_name", "filtered_tspp_path_distances", "filtered_tspp_path_distance"])
    
    for x, row in test_predictions.iterrows():
        concept, response, concept_shift, target_timestep = row.concept, row.response, row.concept_shift, row.target_timestep
        concept, response, concept_shift = torch.tensor(ast.literal_eval(concept)).unsqueeze(0).to(device), torch.tensor(ast.literal_eval(response)).unsqueeze(0).to(device), torch.tensor(ast.literal_eval(concept_shift)).unsqueeze(0).to(device)
        
        if x == args['instance']:
            
            for alg in ['ktcf', 'watcher', 'dice']:
                params['alg'] = alg
        
                if alg == 'ktcf':
                    params['init_scheme'] = 1
                    cf_r = initialize_response(params, response)
                    ktcf = KTCF(model)
                    cf_r, pred_at_target, runtime = ktcf.generate_cf_dkt(params,
                        concept, response, cf_r, concept_shift, target_timestep=target_timestep, node_info=kc_info, shortest_info=shortest_info,
                        lr=0.01, max_iter=200,
                        lambda_pred=1.0, lambda_prox=0.1, lambda_kc=0.001,
                        early_stopping_eta=0.0001, verbose=False
                    )
                elif alg == 'watcher':
                    params['init_scheme'] = 2
                    cf_r = initialize_response(params, response)
                    watchercf = WachterCF(model)
                    cf_r, pred_at_target, runtime = watchercf.generate_cf_dkt(params,
                        concept, response, cf_r, concept_shift, target_timestep=target_timestep,
                        lr=0.01, max_iter=200, 
                        lambda_prox=0.1,
                        early_stopping_eta=0.0001, verbose=False
                    )
                elif alg == 'dice':
                    params['init_scheme'] = 2
                    dicecf = DiCECF(model)
                    cf_r_init_dice = initialize_response(params, response, K=3)
                    cf_r, pred_at_target, runtime = dicecf.generate_cf_dkt(params,
                        concept, response, cf_r_init_dice, concept_shift, target_timestep=target_timestep, 
                        lr=0.01, max_iter=200, 
                        lambda_prox=0.1, lambda_div=0.001,
                        early_stopping_eta=0.0001, verbose=False
                    )
                    cf_r = cf_r[0]


                index = x

                cf_r_label = (torch.sigmoid(cf_r) > 0.5).float()

                unactionables = torch.where(((response == 1) & (cf_r_label == 0 )))[1]
                common_wrong = torch.where((response != cf_r_label ))[1]

                indices = common_wrong.detach().cpu().numpy().tolist()

                common_wrong_kc_ids = concept[0].cpu()[common_wrong.cpu()]
                unactionables_id = concept[0].cpu()[unactionables.cpu()]

                common_wrong_kc_names = [id_to_name[int(kc_id)] for kc_id in common_wrong_kc_ids]
                unactionables_kc_names = [id_to_name[int(kc_id)] for kc_id in unactionables_id]

                if not indices:
                    return
                
                original_path_distance = []
                for i in range(len(common_wrong_kc_names) - 1):
                    try:
                        dist = nx.shortest_path_length(graph, source=common_wrong_kc_names[i], target=common_wrong_kc_names[i+1])
                        original_path_distance.append(dist)
                    except:
                        original_path_distance.append(None) 


                target_kc_id = int(concept_shift[0, target_timestep].item())
                target_kc_name = id_to_name[target_kc_id]


                origin_tspp_graph = nx.Graph()
                origin_all_nodes = common_wrong_kc_names
                if target_kc_name not in origin_all_nodes:
                    origin_all_nodes.append(target_kc_name)
                origin_tspp_graph.add_nodes_from(origin_all_nodes)

            
                with open("././data/XES3G5M/metadata/shortest_paths.pkl", "rb") as f:
                    shortest_paths = pickle.load(f)

                for i in range(len(origin_all_nodes)):
                    for j in range(i+1, len(origin_all_nodes)):
                        u, v = origin_all_nodes[i], origin_all_nodes[j]
                        try:
                            dist = shortest_paths[(u,v)]
                            origin_tspp_graph.add_edge(u, v, weight=dist)
                        except KeyError:
                            continue


                origin_tspp_path, origin_tspp_total_distance = tspp_with_fixed_target(origin_tspp_graph, target_kc_name)
                origin_tspp_path_ids = [name_to_id[name] for name in origin_tspp_path if name in name_to_id]



                origin_tspp_distances = []
                for i in range(len(origin_tspp_path) - 1):
                    try:
                        dist = nx.shortest_path_length(graph, source=origin_tspp_path[i], target=origin_tspp_path[i+1])
                        origin_tspp_distances.append(dist)
                    except:
                        origin_tspp_distances.append(None)

                origin_tspp_total_distance = sum(d for d in origin_tspp_distances if d is not None)

                
                df_new = pd.DataFrame({
                    "alg": alg,
                    "index": index,
                    "batch": [row.batch],
                    "instance": [row.instance],
                    
                    "common_wrong_indices": [indices],
                    "common_wrong_kc_ids": [common_wrong_kc_ids.tolist()],
                    "common_wrong_kc_names": [common_wrong_kc_names],
                    "unactionables_kc_names": [unactionables_kc_names],
                    
                    "original_path_distance": [original_path_distance],
                    "original_path_total_distance": [sum(d for d in original_path_distance if d is not None)],
                    
                    "origin_tspp_path_ids": [origin_tspp_path_ids],
                    "origin_tspp_path_names": [origin_tspp_path],
                    "origin_tspp_distances": [origin_tspp_distances],
                    "origin_tspp_total_distance": [origin_tspp_total_distance],

                })
                df = pd.concat([df, df_new], ignore_index=True)
            save_path = os.path.join(results_dir, "educational_instructions.csv")
            df.to_csv(save_path, index=False, mode='a', header=False, encoding="utf-8-sig")
            print(f"8. Saved TSPP path to: {save_path}")
            break

def main(args):
    with open("saved_model/config.json") as fin:
        config = json.load(fin)
        model_config = config["model_config"]
        for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
            model_config.pop(remove_item, None)
        params = config["params"]
        model_name, dataset_name, emb_type = params["model_name"], params["dataset_name"], params["emb_type"]

    with open("./configs/data_config.json") as fin:
        data_config = json.load(fin)[dataset_name]
        data_config["dataset_name"] = dataset_name

    test_predictions = pd.read_csv('./data/XES3G5M/XES3G5M_test_predictions.csv')
    

    kc_info = pd.read_csv('./data/XES3G5M/metadata/kc_nodes.csv')

    with open("./data/XES3G5M/metadata/shortest_paths.pkl", 'rb') as f:
        shortest_info = pickle.load(f)

    model = init_model(model_name, model_config, data_config, emb_type)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    export_tspp_cf_indices(args, model, test_predictions, kc_info, shortest_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=1452)

    args = parser.parse_args()
    args = vars(args)
    main(args)
