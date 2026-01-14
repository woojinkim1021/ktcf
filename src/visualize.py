import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

pd.set_option('display.float_format', '{:.4f}'.format)

base_args = {
    'init_noise_std': 0.1,
    'init_temp' : 0.1,
    'lambda_kc' : 0.001,
}


def get_lines_to_plot_for_loss(df, alg, init_scheme, hyperparam_name, hyperparam_vals, metric, base_args):
    find_lst = []
    filter_args = {k: v for k, v in base_args.items() if k != hyperparam_name}
    for val in hyperparam_vals:
        mask = (
            (df['alg'] == alg) &
            (df['init_scheme'] == init_scheme) &
            (df[hyperparam_name] == val)
        )
        for k, v in filter_args.items():
            mask = mask & (df[k] == v)
        metrics = df[mask][metric].to_list()
        find_lst.append(metrics)
    results = []
    for i in find_lst:

        results.append(min(i))
    return results

def plot_for_loss(df, lambda_name_lst, lambda_val_lst, filters, scheme_num, legend=False):
    color_lst = ['blue', 'red', 'green', 'purple', 'darkorange']
    metric_names = ['Validity', 'Sparsity', 'KC Distance', 'Time']
    init_names = ['-rn', '-rand', '-sr', '-cc', '-gs']
    param_names = ['lambda_pred', 'lambda_prox', 'lambda_kc']
    tick_shape=['.', '*', ',', 'x', '|']

    fig, axes = plt.subplots(nrows=len(lambda_name_lst), ncols=len(filters), figsize=(2.5 * len(filters), 1.7))
    
    x_labels = [str(val) for val in lambda_val_lst[0]]
    lines = []
    labels = []
    for l in range(len(lambda_name_lst)):
        for i in range(len(filters)):
            for init_s, color in zip([1, 2, 3, 4, 5], color_lst):
                y = get_lines_to_plot_for_loss(df, 'ktcf', init_s, lambda_name_lst[l], lambda_val_lst[l], filters[i], base_args)
                label = 'KTCF'+'-'+init_names[init_s-1]
                line, = axes[i].plot(x_labels, y, tick_shape[init_s-1], linewidth=0.9, label=label, color=color, linestyle='-')
                lines.append(line)
                labels.append(label)
                axes[i].set_xlabel(param_names[scheme_num], fontsize=12)
                axes[i].set_ylabel(metric_names[i], fontsize=10)
                axes[i].tick_params(axis='both', which='major', labelsize=7)
                xticks = axes[i].get_xticks()
                for xtick in xticks:
                    axes[i].axvline(xtick, color=color, linestyle='--', alpha=0.05)

    if legend:
        unique_labels = []
        unique_lines = []
        for line, label in zip(lines, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_lines.append(line)
            if len(unique_labels) == 5:
                break

        fig.legend(unique_lines, unique_labels,
            fontsize=12,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2), 
            ncol=5,                    
            frameon=True
        )
        plt.tight_layout(rect=[0, 0, 1, 1])
    else:
        plt.tight_layout()
    save_path = os.path.join(results_dir, lambda_name_lst[0]+".pdf")
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)


def plot_for_init(df, lambda_name_lst, lambda_val_lst, filters, init_scheme, lamb_num, legend):
    color_lst = ['magenta', 'darkturquoise', 'forestgreen']
    metric_names = ['Validity', 'Sparsity', 'Actionability', 'Actionability Rate', 'Time']
    alg_names = ['KTCF', 'Wachter', 'DiCE']
    init_names = ['-rn', '-rand', '-sr', '-cc', '-gs']
    tick_shape=['.', '*', ',', 'x', '|']
    param_names = ['lambda_noise', 'lambda_convex', 'lambda_temp']


    fig, axes = plt.subplots(nrows=len(lambda_name_lst), ncols=len(filters), figsize=(2.5 * len(filters), 1.7))
    
    x_labels = [str(val) for val in lambda_val_lst[0]]
    lines = []
    labels = []
    for l in range(len(lambda_name_lst)):
        for i in range(len(filters)):
            for alg, c in zip(['ktcf', 'wachter', 'dice'], range(len(color_lst))):
                y = get_lines_to_plot_for_loss(df, alg, init_scheme, lambda_name_lst[l], lambda_val_lst[l], filters[i], base_args)
                label = alg_names[c] + '-' + init_names[init_scheme-1]
                line, = axes[i].plot(x_labels, y, tick_shape[c], label=label, color=color_lst[c], linestyle='-') 
                if i == 0:
                    axes[i].set_ylim(0.62, 1)
                lines.append(line)
                labels.append(label)
                axes[i].set_xlabel(param_names[lamb_num], fontsize=12)
                axes[i].set_ylabel(metric_names[i], fontsize=10)
                axes[i].tick_params(axis='both', which='major', labelsize=7)
                xticks = axes[i].get_xticks()
                for xtick in xticks:
                    axes[i].axvline(xtick, color=color_lst[c], linestyle='--', alpha=0.05)
        
    if legend:
        unique_labels = []
        unique_lines = []
        for line, label in zip(lines, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_lines.append(line)
            if len(unique_labels) == 5:
                break

        fig.legend(unique_lines, unique_labels,
            fontsize=14,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2),  
            ncol=5,                     
            frameon=True
        )
        plt.tight_layout(rect=[0, 0, 1, 1])
    else:
        plt.tight_layout()
    plt.tight_layout()
    save_path = os.path.join(results_dir, lambda_name_lst[0]+".pdf")
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)


if __name__ == "__main__":\

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_csv_name', type=str, default='figure_3_results.csv')
    args = parser.parse_args()
    
    init_noise_lst = [0.01, 0.05, 0.1, 0.5, 1.0]
    init_temp_lst = [0.1, 0.3, 0.5, 1.0, 2.0]
    lambda_kc_lst = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    results_dir = "./results/"
    results_path = os.path.join(results_dir, args.result_csv_name)
    df = pd.read_csv(results_path)

    tartget_metrics_init = ['validity_mean', 'sparsity_mean', 'actionability_mean', 'actionability_rate_mean', 'time_mean']

    plot_for_init(df, ['init_noise_std'], [init_noise_lst], tartget_metrics_init, init_scheme=1, lamb_num=0, legend=True)
    plot_for_init(df, ['init_temp'], [init_temp_lst], tartget_metrics_init, init_scheme=5, lamb_num=2, legend=True)

    tartget_metrics_loss = ['validity_mean', 'sparsity_mean', 'kc_distance_mean', 'time_mean']
    
    plot_for_loss(df, ['lambda_kc'], [lambda_kc_lst], tartget_metrics_loss, 2, legend=True)
