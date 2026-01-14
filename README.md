# [AAAI-26] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education

This code appendix contains the source code, scripts, and configuration files used in the experiments for our research paper, "KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education" accepted at **AAAI-26 Special Track on AI for Social Impact** as an **Oral Presentation**.

## 📖 Abstract

> Using Artificial Intelligence to improve teaching and learning benefits greater adaptivity and scalability in education. Knowledge Tracing (KT) is recognized for student modeling task due to its superior performance and application potential in education. To this end, we conceptualize and investigate counterfactual explanation as the connection from XAI for KT to education. Counterfactual explanations offer actionable recourse, are inherently causal and local, and easy for educational stakeholders to understand who are often non-experts. We propose KTCF, a counterfactual explanation generation method for KT that accounts for knowledge concept relationships, and a post-processing scheme that converts a counterfactual explanation into a sequence of educational instructions. We experiment on a large-scale educational dataset and show our KTCF method achieves superior and robust performance over existing methods, with improvements ranging from 5.7% to 34% across metrics. Additionally, we provide a qualitative evaluation of our post-processing scheme, demonstrating that the resulting educational instructions help in reducing large study burden. We show that counterfactuals have the potential to advance the responsible and practical use of AI in education. Future works on XAI for KT may benefit from educationally grounded conceptualization and developing stakeholder-centered methods.

## 👥 Authors

* **Woojin Kim** (Korea University)
* **Changkwon Lee** (Korea University)
* **Hyeoncheol Kim** (Korea University)

---

## Requirements

* Python 3.12.11
* See `requirements.txt` for a full list of dependencies
* Hardware specifications:
   - OS: Ubuntu 20.04.6
   - CPU: Intel(R) Xeon(R) Gold 6430
   - Memory: 256GiB
   - GPU: single NVIDIA GeForce RTX 4090
   - CUDA Version: 12.8


---

## Code Appendix Structure

```
├── configs/                              # configurations of DKT
├── data/
│   └── XES3G5M/                          # directory for XES3G5M dataset
│       └── metadata/                     # directory for XES3G5M's metadata
│           └── kc_routes_map_en.json     # KC names translated to English
├── pykt/                                 # relevant pyKT scripts for training DKT
├── results/                              # directory for experiment results
├── saved_model/                          # directory for DKT model checkpoints
├── src/
│   └── ablation.py                       # script for running ablation study of the paper
│   └── run.py                            # script for running evaluation in Table 1 of the paper
│   └── visualize.py                      # script for visualizing Figure 3 of the paper
│   └── explain.py                        # script for generating the Table 2 of the paper
│   └── ktcf/
│       └── cf_generator.py               # script of KTCF and baseline methods
│       └── eval_metrics.py               # script of evaluation metrics of the paper
│       └── init_cf.py                    # script of initialization strategies of the paper
│   └── 1_data_preprocessing/
│       └── create_kc_network.py          # script for constructing KC relation graph
│       └── preprocess_XES3G5M.py         # script for preprocessing XES3G5M dataset
│   └── 2_dkt_training_eval/
│       └── dkt_inference.py              # script for saving DKT inference results
│       └── wandb_dkt_train.py            # pyKT script of arguments for training DKT model
│       └── wandb_predict.py              # pyKT script for evaluating DKT model
│       └── wandb_train.py                # pyKT script of training DKT model
├── requirements.txt
└── README.md
```
