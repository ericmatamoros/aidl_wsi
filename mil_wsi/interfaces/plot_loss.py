"""Plot losses of the model"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(loss_list, output_dir, suffix_name, training_type):
    """
    Generates and saves loss plots per epoch:
    
    1. A plot showing the **average loss across all folds** (if using K-Fold).
    2. A plot displaying the **loss for each individual fold**.

    Args:
        loss_list (list of lists or list): 
            - If using K-Fold, it's a list where each element is a list of losses per epoch for a fold.
            - If not using K-Fold, it's a single list of losses per epoch.
        output_dir (str): Path to the directory where the plots will be saved.
        suffix_name (str): Suffix for naming the output files.
        training_type (str): Type of training (e.g., "MIL", "CNN", etc.).

    Saves:
        - `{training_type}_{suffix_name}_loss_plot_mean.png`: Plot of the mean loss per epoch.
        - `{training_type}_{suffix_name}_loss_plot_folds.png`: Plot of loss per epoch for each fold.
    
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert list of lists to a NumPy array and compute the mean if using K-Fold
    if isinstance(loss_list, list) and isinstance(loss_list[0], list):
        loss_array = np.array(loss_list)
        mean_loss = np.mean(loss_array, axis=0)
    else:
        loss_array = np.array([loss_list])  # Convert to a single-dimension array if not using K-Fold
        mean_loss = loss_list

    # 1️. **Plot of the Mean Loss**
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mean_loss) + 1), mean_loss, marker='o', linestyle='-', label="Mean Loss", color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Mean {training_type} Loss per Epoch (Averaged over Folds)")
    plt.legend()
    plt.ylim(np.min(mean_loss)-0.2, np.max(mean_loss)+0.2)
    plt.grid(True)
    output_path_mean = os.path.join(output_dir, f"{training_type}_{suffix_name}_loss_plot_mean.png")
    plt.savefig(output_path_mean)  # Save the plot
    # plt.show()

    # 2️. **Plot of Loss for Each Fold**
    plt.figure(figsize=(8, 6))
    max_loss = 0
    for i, fold_loss in enumerate(loss_array):
        plt.plot(range(1, len(fold_loss) + 1), fold_loss, marker='o', linestyle='-', label=f"Fold {i+1}")
        if np.max(fold_loss) > max_loss:
            max_loss = np.max(fold_loss)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{training_type} Loss per Epoch for Each Fold")
    plt.legend()
    plt.ylim(np.min(loss_array)-0.2, max_loss +0.2)
    plt.grid(True)
    output_path_folds = os.path.join(output_dir, f"{training_type}_{suffix_name}_loss_plot_folds.png")
    plt.savefig(output_path_folds)  # Save the plot
    # plt.show()

    print(f"Loss plots saved to {output_path_mean} and {output_path_folds}")
