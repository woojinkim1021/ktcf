import os
import sys
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import json
import copy
import ast

from pykt.models.init_model import load_model
from pykt.datasets.init_dataset import init_test_datasets


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


def predict_all_instances(model, test_loader):
    """
    Runs model predictions on all instances in the provided test_loader and collects detailed results.

    For each batch in the test_loader, and for each instance within a batch:
        - Runs the model to obtain predictions.
        - Applies a custom logic for extracting binary labels and identifying a target timestep where both
          the response and prediction are zero.
        - Records relevant information for each valid instance (those with at least one matching timestep).
        - Tracks and reports the number of instances where no valid matching timestep is found.

    Args:
        model (torch.nn.Module): The trained PyTorch model to be used for prediction.
        test_loader (torch.utils.data.DataLoader): DataLoader yielding batches of test data. Each batch must be
            a dictionary containing keys 'cseqs', 'rseqs', and 'shft_cseqs', mapping to tensors for
            concept sequences, response sequences, and shifted concept sequences, respectively.

    Returns:
        tuple of lists:
            - batch_lst (list of int): Batch index for each valid instance.
            - instance_lst (list of int): Instance index within the batch for each valid instance.
            - concept_lst (list of str): String representation of the concept sequence for each instance.
            - response_lst (list of str): String representation of the response sequence for each instance.
            - cshft_lst (list of str): String representation of the shifted concept sequence for each instance.
            - y_lst (list of str): String representation of the model's output predictions for each instance.
            - label_lst (list of str): String representation of the computed binary labels for each instance.
            - target_timestep_lst (list of int): Index of the target timestep (last position where both response
              and prediction are zero) for each valid instance.

    Side Effects:
        Prints the total number of instances that did not meet the selection criteria and were removed.

    Notes:
        - Instances without any timestep matching the condition (response == 0 and prediction == 0) are excluded.
        - All tensor data are converted to NumPy arrays and stringified for convenient CSV or logging use.
        - The function assumes that the model and data are compatible (correct shapes and device).

    """

    num_no_matches = 0
    batch_lst = []
    instance_lst = []
    concept_lst = []
    response_lst = []
    cshft_lst = []
    y_lst = []
    label_lst = []
    target_timestep_lst = []

    with torch.inference_mode():
        for i, data in enumerate(test_loader): # for every batch
            c, r, cshft = data["cseqs"], data["rseqs"], data["shft_cseqs"]
            c, r, cshft = c.to(device), r.to(device), cshft.to(device)
            for j, (concept, response, concept_shift) in enumerate(zip(c, r, cshft)): #for every instance
                model.eval()
                y = model(concept.long(), response.long())
                y = (y * F.one_hot(concept_shift.long(), model.num_c)).sum(-1)
                binary_labels = (y > 0.5).float()
                target_timestep = torch.where((response == 0) & (binary_labels == 0))[0][-1:]

                if len(target_timestep) != 0:
                    batch_lst.append(i)
                    instance_lst.append(j)
                    concept_lst.append(np.array2string(concept.cpu().flatten().numpy(), separator=", "))
                    response_lst.append(np.array2string(response.cpu().flatten().numpy(), separator=", "))
                    cshft_lst.append(np.array2string(concept_shift.cpu().flatten().numpy(), separator=", "))
                    y_lst.append(np.array2string(y.cpu().flatten().numpy(), separator=", "))
                    label_lst.append(np.array2string(binary_labels.cpu().flatten().numpy(), separator=", "))
                    target_timestep_lst.append(target_timestep.cpu().item())
                else:
                    num_no_matches += 1

        print(f'total {num_no_matches} instances are removed.')
        return batch_lst, instance_lst, concept_lst, response_lst, cshft_lst, y_lst, label_lst, target_timestep_lst


def at_least_45_percent_zeros(s):
    """
    Checks whether at least 45% of the elements in a list are zeros.

    The input is expected to be a string representation of a list (e.g., "[1, 0, 0, 1, 0]").
    The function parses the string into a list, counts the number of zeros, and returns True
    if zeros constitute 45% or more of the list, False otherwise.

    Args:
        s (str): A string representation of a list of integers.

    Returns:
        bool: True if zeros make up at least 45% of the list, False otherwise.

    Notes:
        - Uses ast.literal_eval for safe parsing of the string to a list.
    """
    
    lst = ast.literal_eval(s)
    zeros = lst.count(0)
    return zeros / len(lst) >= 0.45


if __name__ == "__main__":
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
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    

    batch_lst, instance_lst, concept_lst, response_lst, cshft_lst, y_lst, label_lst, target_timestep_lst, = predict_all_instances(model, test_loader)
    test_predictions = pd.DataFrame({'batch':batch_lst, 'instance':instance_lst, 
                                    'concept':concept_lst, 'response':response_lst, 'concept_shift':cshft_lst,
                                    'y':y_lst, 'label':label_lst, 'target_timestep':target_timestep_lst})
    test_predictions.to_csv("./data/XES3G5M/XES3G5M_test_predictions.csv", index=False)
    
    filtered_test_predictions = test_predictions[test_predictions['response'].apply(at_least_45_percent_zeros)]
    filtered_test_predictions.to_csv("./data/XES3G5M/XES3G5M_test_predictions_45_zeros.csv", index=False)
