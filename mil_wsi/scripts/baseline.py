from mil_wsi.data_loader.dataset_generic import Generic_MIL_Dataset
import torch
from torch.utils.data import random_split


dataset = Generic_MIL_Dataset(
    csv_path='mil_wsi/data/target.csv',
    data_dir='mil_wsi/results',
    print_info=True,
    label_dict={"negative": 0, "positive": 1}, 
    label_col='TRY'  # Posar el nom de la columna on hi ha si la mostra es positiva o negativa"
)

#Split dataset dinamically:
#train_size = int(0.8 * len(dataset))
#val_size = int(0.1 * len(dataset))
#test_size = len(dataset) - train_size - val_size

train_size = 2
test_size = 2
val_size = 2

torch.manual_seed(123)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,test_size,val_size])

print(f"Lenghth of the dataset: {len(dataset)}")

print(f"Lenghth of the train split: {len(train_dataset)}")

print("1rst train image:")
print(train_dataset[0])
print("2nd train image:")
print(train_dataset[1])


