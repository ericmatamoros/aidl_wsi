from loguru import logger
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Capa de entrada a capa oculta
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Capa oculta a capa de salida
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Función de activación relu
        self.relu = nn.ReLU()
        # funcion de activacion sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
            # Propagación hacia adelante
            x = self.relu(self.fc1(x))  # Aplicamos ReLU después de la primera capa
            #x = self.sigmoid(self.fc2(x))  # Salida de la segunda capa
            x = self.fc2(x)  # No cal aplicar Sigmoid. La loss function que fem servir es la BCEWithLogitsLoss, la qual espera logits com a entrada.
            return x
    

def train_mlp(model, train_loader, criterion, optimizer, device: torch.device, epochs: int):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move inputs and labels to the correct device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    return model


def predict_mlp(model, test_loader, device: torch.device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            # Move inputs to the correct device
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).round()  # Sigmoide y redondeo para obtener 0 o 1
            all_preds.append(preds)
    
    return torch.cat(all_preds, dim=0)