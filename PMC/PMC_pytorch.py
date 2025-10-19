import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from funciones_auxiliares import embeber_datos2, cargar_corpus, cargar_modelo, sigmoide_np

# --- Definición del Modelo ---

class PMC_PyTorch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PMC_PyTorch, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 256)
        self.act1 = nn.GELU()
        self.layer2 = nn.Linear(256, 128)
        self.act2 = nn.GELU()
        self.layer3 = nn.Linear(128, output_dim)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        return x

# --- Lógica Principal de Entrenamiento ---

# 1. Carga y Preparación de Datos
corpus, vocab, vocab_size, word_to_idx, idx_to_word = cargar_corpus("mini_corpus.txt")
W1, W2, N, C, eta = cargar_modelo("pesos_cbow_pcshavak-mini_epoca999.npz", "weights")
x_train_np, y_train_np = embeber_datos2(corpus, W1, word_to_idx, 10)

x_train_np = sigmoide_np(x_train_np)
y_train_np = sigmoide_np(y_train_np)

print("x_train shape:", x_train_np.shape)
print("y_train shape:", y_train_np.shape)

input_size = x_train_np.shape[1]
output_size = y_train_np.shape[1]

# 2. Configuración de Hiperparámetros y DataLoader
BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 0.001

x_train = torch.tensor(x_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 3. Instanciación de Modelo, Pérdida y Optimizador
model = PMC_PyTorch(input_size, output_size).to(device)
print(model)

criterion = nn.MSELoss()
criterion_mae = nn.L1Loss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Bucle de Entrenamiento
losses_history = []
mae_history = []

print("Iniciando entrenamiento...")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_mae = 0.0
    
    model.train() 
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mae = criterion_mae(outputs, targets)
            epoch_loss += loss.item() * inputs.size(0)
            epoch_mae += mae.item() * inputs.size(0)

    epoch_loss_avg = epoch_loss / len(train_dataset)
    epoch_mae_avg = epoch_mae / len(train_dataset)
    
    losses_history.append(epoch_loss_avg)
    mae_history.append(epoch_mae_avg)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss (MSE): {epoch_loss_avg:.6f}, MAE: {epoch_mae_avg:.6f}', end='\r')

print("\nEntrenamiento finalizado.")

# 5. Visualización y Guardado
plt.plot(losses_history, label='Entrenamiento (MSE)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (loss)')
plt.title('Evolución del error')
plt.legend()
plt.grid()
plt.show()

torch.save(model, "PMC/modelo.pth")
print("Modelo guardado en PMC/modelo.pth")