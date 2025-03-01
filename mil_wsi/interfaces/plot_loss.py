import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(loss_list, output_dir, suffix_name, training_type):
    """
    Generates and saves two loss plots per epoch:
    1. A plot showing the **average loss across all folds**.
    2. A plot displaying the **loss for each individual fold**.

    Parameters:
    - loss_list: List of losses (a list of lists if using K-Fold).
    - output_dir: Directory where the plot images will be saved.
    - suffix_name: Suffix for naming the output files.
    - training_type: Type of training (e.g., "MIL", "CNN", etc.).
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
    plt.ylim(0, 1)
    plt.grid(True)
    output_path_mean = os.path.join(output_dir, f"{training_type}_{suffix_name}_loss_plot_mean.png")
    plt.savefig(output_path_mean)  # Save the plot
    # plt.show()

    # 2️. **Plot of Loss for Each Fold**
    plt.figure(figsize=(8, 6))
    for i, fold_loss in enumerate(loss_array):
        plt.plot(range(1, len(fold_loss) + 1), fold_loss, marker='o', linestyle='-', label=f"Fold {i+1}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{training_type} Loss per Epoch for Each Fold")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True)
    output_path_folds = os.path.join(output_dir, f"{training_type}_{suffix_name}_loss_plot_folds.png")
    plt.savefig(output_path_folds)  # Save the plot
    # plt.show()

    print(f"Loss plots saved to {output_path_mean} and {output_path_folds}")
