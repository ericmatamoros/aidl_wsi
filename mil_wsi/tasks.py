import os
import subprocess

def run_task(command, env_vars=None):
    """Runs a shell command with optional environment variables."""
    env = {**os.environ, **(env_vars or {})}  # Merge with existing environment
    subprocess.run(command, shell=True, env=env, check=True)

def main():
    tasks = {
        "1": ("Convert to PNG", "conda run -n wsi python -m mil_wsi.scripts.convert_to_png --source ./mil_wsi/data --save_dir ./mil_wsi/data/images/ --file_extension svs"),
        "2": ("Create Patches", "conda run -n wsi python -m mil_wsi.scripts.create_patches --source ./mil_wsi/data --save_dir ./mil_wsi/results/ --patch_size 2856 --seg --patch --stitch"),
        "3": ("Extract Features", "conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.extract_features --data_h5_dir ./mil_wsi/results/ --data_slide_dir ./mil_wsi/data/ --csv_path ./mil_wsi/results/process_list_autogen.csv --feat_dir ./mil_wsi/results/ --model_name conch_v1 --batch_size 512 --slide_ext .svs"),
        "4": ("MLP Model", "conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.mlp_model --dir_results ./mil_wsi/results/ --dir_data ./mil_wsi/data/ --dir_model ./mil_wsi/models/ --dir_metrics ./mil_wsi/metrics/"),
        "6": ("Weighted Model", "conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.weighted_model --dir_results ./mil_wsi/results/ --dir_data ./mil_wsi/data/ --dir_model ./mil_wsi/models/ --dir_metrics ./mil_wsi/metrics/"),
        "7": ("Attention MIL", "conda run -n wsi CUDA_VISIBLE_DEVICES=0 python -m mil_wsi.scripts.attention_mil_model --dir_results ./mil_wsi/results/ --dir_data ./mil_wsi/data/ --dir_model ./mil_wsi/models/ --dir_metrics ./mil_wsi/metrics/"),
        "8": ("Transformer MIL", "conda run -n wsi python -m mil_wsi.scripts.transformer_mil_model --dir_results ./mil_wsi/results/ --dir_data ./mil_wsi/data/ --dir_model ./mil_wsi/models/ --dir_metrics ./mil_wsi/metrics/"),
        "9": ("Run Full Pipeline", ""),
    }
    
    print("Select a task to run:")
    for key, (name, _) in tasks.items():
        print(f"[{key}] {name}")
    
    choice = input("Enter the number of the task: ")
    
    if choice in tasks:
        name, command = tasks[choice]
        print(f"Running: {name}\n")
        if choice == "9":  # Full pipeline
            for task_key in ["2", "3", "4", "5", "6", "7", "8"]:
                print(f"Executing: {tasks[task_key][0]}")
                run_task(tasks[task_key][1])
        else:
            run_task(command)
    else:
        print("Invalid choice. Please select a valid task number.")

if __name__ == "__main__":
    main()
