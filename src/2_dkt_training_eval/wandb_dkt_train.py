"""
Script from pyKT package version v1.0.0 (https://github.com/pykt-team/pykt-toolkit).


Reference:
Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). 
pyKT: a python library to benchmark deep learning based knowledge tracing models. 
Advances in Neural Information Processing Systems, 35, 18542-18555.
"""

import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import argparse
from wandb_train import main

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="XES3G5M")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
