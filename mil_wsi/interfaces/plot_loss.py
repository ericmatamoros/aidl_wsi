import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(loss_list, output_dir, suffix_name, training_type):
    """
    Genera y guarda dos gráficas de la pérdida por época:
    1. Gráfica de la pérdida promedio de todos los folds.
    2. Gráfica de la pérdida de cada fold individualmente.

    Parámetros:
    - loss_list: Lista de pérdidas (una lista de listas si es K-Fold).
    - output_dir: Directorio donde se guardarán las imágenes de los gráficos.
    """
    # Asegurarse de que el directorio existe
    

    # Convertir lista de listas a array numpy y calcular la media si es K-Fold
    if isinstance(loss_list, list) and isinstance(loss_list[0], list):
        loss_array = np.array(loss_list)
        mean_loss = np.mean(loss_array, axis=0)
    else:
        loss_array = np.array([loss_list])  # Convertir a array de una dimensión si es un solo experimento
        mean_loss = loss_list

    # 1️⃣ **Gráfica de la pérdida promedio**
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(mean_loss) + 1), mean_loss, marker='o', linestyle='-', label="Mean Loss", color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Mean {training_type} Loss per Epoch (Averaged over Folds)")
    plt.legend()
    plt.grid(True)
    output_path_mean = os.path.join(output_dir, f"{training_type}_{suffix_name}_loss_plot_mean.png")
    plt.savefig(output_path_mean)  # Guardar la imagen
    #plt.show()

    # 2️⃣ **Gráfica de la pérdida de cada fold**
    plt.figure(figsize=(8,6))
    for i, fold_loss in enumerate(loss_array):
        plt.plot(range(1, len(fold_loss) + 1), fold_loss, marker='o', linestyle='-', label=f"Fold {i+1}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{training_type} Loss per Epoch for Each Fold")
    plt.legend()
    plt.grid(True)
    output_path_folds = os.path.join(output_dir, f"{training_type}_{suffix_name}_loss_plot_folds.png")
    plt.savefig(output_path_folds)  # Guardar la imagen
    #plt.show()

    print(f"Loss plots saved to {output_path_mean} and {output_path_folds}")
