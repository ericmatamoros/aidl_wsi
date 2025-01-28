from mil_wsi.data_loader.dataset_generic import Generic_MIL_Dataset
import torch

# dins de dataset generic s'han modificat 76, 77, 79 i 80

# Configuración inicial
dataset = Generic_MIL_Dataset(
    csv_path='mil_wsi/data/target.csv',
    data_dir='mil_wsi/results',
    shuffle=False, 
    seed=123, 
    print_info=True,
    label_dict={0: 0, 1: 1},  # Etiquetas binarias
    label_col='label'
)

# Dividir los datos en entrenamiento, validación y prueba
train_set, val_set, test_set = torch.utils.data.random_split(dataset,[2,2,2])
# Iterate through the train set and inspect
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

for i in range(len(train_set)):
    features, label = train_set[i]  # Asegúrate de descomprimir los datos correctamente
    print(f"Features shape: {features.shape}, Label: {label}")