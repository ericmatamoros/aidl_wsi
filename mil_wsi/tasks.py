import os
import yaml
import subprocess
import sys

from mil_wsi import CONFIG_PATH
from loguru import logger

def run_task(command, env_vars=None):
    """Runs a shell command with optional environment variables and prints logs in real-time."""
    env = {**os.environ, **(env_vars or {})}  # Merge with existing environment

    process = subprocess.Popen(
        command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Print logs in real-time
    for line in process.stdout:
        sys.stdout.write(line)  # Print without adding extra newlines
        sys.stdout.flush()  # Ensure real-time output

    for line in process.stderr:
        sys.stderr.write(line)  # Print errors to stderr
        sys.stderr.flush()

    process.wait()  # Wait for process to finish

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

def main():

    with open(f"{CONFIG_PATH}/config.yaml", "r") as file:
        settings = yaml.safe_load(file)
    
    create_patch_settings = settings['create_patches']
    feat_extract_settings = settings['feature_extraction']
    model_conf_settings = settings['model_configuration']

    data_path = settings['data_path']
    results_path = settings['results_path']
    metrics_path = settings['metrics_path']
    models_path = settings['models_path']

    tasks = {
        "1": ("Create Patches", f"conda run -n wsi python -m mil_wsi.scripts.create_patches --source {data_path} --save_dir {results_path} --patch_size {create_patch_settings['patch_size']} --step_size {create_patch_settings['step_size']} --seg --patch --stitch --experiment_name {settings['data_name']}"),
        "2": ("Extract Features", f"conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.extract_features --data_h5_dir {results_path} --data_slide_dir {data_path} --csv_path {results_path} --feat_dir {results_path} --model_name {feat_extract_settings['model_name']} --batch_size {feat_extract_settings['batch_size']} --slide_ext {feat_extract_settings['slide_ext']} --experiment_name {settings['data_name']}"),
        "3": ("MLP Model", f"conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.mlp_model --dir_results {results_path} --dir_data {data_path} --dir_model {models_path} --dir_metrics {metrics_path} --batch_size {model_conf_settings['batch_size']} --hidden_size {model_conf_settings['hidden_size']} --epochs {model_conf_settings['epochs']} --test_size {model_conf_settings['test_size']} --k_folds {model_conf_settings['k_folds']} --learning_rate {model_conf_settings['learning_rate']} --data_name {settings['data_name']} --experiment_name {settings['experiment_name']} --experiment_number {settings['experiment_number']}"),
        "4": ("Attention MIL", f"conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.attention_mil_model --dir_results {results_path} --dir_data {data_path} --dir_model {models_path} --dir_metrics {metrics_path} --batch_size {model_conf_settings['batch_size']} --hidden_size {model_conf_settings['hidden_size']} --epochs {model_conf_settings['epochs']} --test_size {model_conf_settings['test_size']} --k_folds {model_conf_settings['k_folds']} --learning_rate {model_conf_settings['learning_rate']} --attention_class {model_conf_settings['attention_class']} --n_heads {model_conf_settings['n_heads']} --experiment_name {settings['data_name']}"),
        "5": ("Transformer MIL", f"conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.transformer_mil_model --dir_results {results_path} --dir_data {data_path} --dir_model {models_path} --dir_metrics {metrics_path} --batch_size {model_conf_settings['batch_size']} --epochs {model_conf_settings['epochs']} --test_size {model_conf_settings['test_size']} --k_folds {model_conf_settings['k_folds']} --learning_rate {model_conf_settings['learning_rate']} --n_heads {model_conf_settings['n_heads']} --experiment_name {settings['data_name']}"),
        "6": ("Run Full Pipeline", ""),
    }
    
    logger.info("Select a task to run:")
    for key, (name, _) in tasks.items():
        logger.info(f"[{key}] {name}")
    
    choice = input("Enter the number of the task: ")
    
    if choice in tasks:
        name, command = tasks[choice]
        logger.info(f"Running: {name}\n")
        if choice == "6":  # Full pipeline
            for task_key in ["1", "2", "3", "4", "5"]:
                logger.info(f"Executing: {tasks[task_key][0]}")
                run_task(tasks[task_key][1])
        else:
            run_task(command)
    else:
        logger.info("Invalid choice. Please select a valid task number.")

if __name__ == "__main__":
    main()