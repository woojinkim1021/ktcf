"""
Script from pyKT package version v1.0.0 (https://github.com/pykt-team/pykt-toolkit).


Reference:
Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). 
pyKT: a python library to benchmark deep learning based knowledge tracing models. 
Advances in Neural Information Processing Systems, 35, 18542-18555.
"""



que_type_models = ["iekt","qdkt","qikt","lpkt", "rkt", "promptkt"]

qikt_ab_models = ["qikt_ab_a+b+c","qikt_ab_a+b+c+irt","qikt_ab_a+b+irt","qikt_ab_a+c+irt","qikt_ab_a+irt","qikt_ab_b+irt"]

que_type_models += qikt_ab_models