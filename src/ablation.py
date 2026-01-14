import itertools
import subprocess
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = os.path.dirname(os.path.abspath(__file__))
run_file_path = os.path.join(script_dir, 'run.py')


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help="['init', 'loss']")
parser.add_argument('--save_file_name', type=str, default='figure_3_results.csv')
args = parser.parse_args()


init_noise_lst = [0.01, 0.05, 0.1, 0.5, 1.0]
init_temp_lst = [0.1, 0.3, 0.5, 1.0, 2.0]
lambda_kc_lst = [0.00001, 0.0001, 0.001, 0.01, 0.1]


base_args = {
    'init_noise_std': 0.1,
    'init_temp' : 0.1,
    'lambda_kc' : 0.001,
}


success_list = []
fail_list = []

def run_experiment(alg, init_scheme, init_noise_std, init_temp, lambda_kc, save_file_name):
    cmd = [
        sys.executable, run_file_path,
        '--alg', alg,
        '--init_scheme', str(init_scheme),
        '--init_noise_std', str(init_noise_std),
        '--init_temp', str(init_temp),
        '--lambda_kc', str(lambda_kc),
        '--save_file_name', save_file_name,
    ]
    label = f"{alg}, init={init_scheme}, init_noise_std={init_noise_std}, init_temp={init_temp}, lambda_kc={lambda_kc}"
    print(f"[RUNNING] {label}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[! SUCCESS] {label}")
        lines = result.stdout.strip().splitlines()
        
        summary_lines = lines[:]
        print("\n".join(summary_lines))
        """
        summary_lines = [
            line for line in lines
            if "validity:" in line or "sparsity" in line or "distance" in line or "actionability" in line or "time" in line
        ]
        if summary_lines:
            print("\n".join(summary_lines[:]))
        else:
            print("No metric summary found.")
        """
        success_list.append(label)
        
    except subprocess.CalledProcessError as e:
        print(f"[! FAIL] {label}")
        print(e.stderr.strip().splitlines()[-1])
        fail_list.append(label)



if __name__ == "__main__":
    if args.mode == 'loss':
        alg = 'ktcf'
        for init_s in [1, 5]:
            for l_kc in lambda_kc_lst:
                run_experiment(alg, init_s, 
                                base_args['init_noise_std'], 
                                base_args['init_temp'], 
                                l_kc, 
                                args.save_file_name)

    elif args.mode == 'init':
        for alg in ['ktcf', 'wachter', 'dice']:
            for init_scheme in [1, 5]:
                if init_scheme == 1:
                    for noise in init_noise_lst:
                        run_experiment(alg, init_scheme, 
                                        noise, 
                                        base_args['init_temp'], 
                                        base_args['lambda_kc'], 
                                        args.save_file_name)
                elif init_scheme == 5:
                    for temp in init_temp_lst:
                        run_experiment(alg, init_scheme, 
                                        base_args['init_noise_std'], 
                                        temp, 
                                        base_args['lambda_kc'], 
                                        args.save_file_name)

    print("\n==================== SUMMARY ====================")
    print(f"! Success: {len(success_list)} experiments")
    for s in success_list:
        print("  -", s)

    print(f"\n! Failures: {len(fail_list)} experiments")
    for f in fail_list:
        print("  -", f)
