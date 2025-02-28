import torch

# Dimensiones pequeñas para un ejemplo simple
N = 3  # Número de embeddings (ej. patches de imagen)
M = 4  # Dimensión del embedding
L = 2  # Dimensión intermedia en la atención

# Embeddings de entrada h_i (cada fila es un embedding)
H = torch.tensor([
    [0.2, 0.4, 0.6, 0.8],
    [0.1, 0.3, 0.5, 0.7],
    [0.3, 0.6, 0.9, 1.2]
])  # Dimensión (N, M)

# Parámetros de la Gated Attention
V = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.4, 0.3, 0.2]
])  # Dimensión (L, M)

U = torch.tensor([
    [0.3, 0.2, 0.1, 0.0],
    [0.4, 0.3, 0.2, 0.1]
])  # Dimensión (L, M)

w = torch.tensor([[0.7, 0.3]])  # Dimensión (1, L)

# Paso 1: Aplicamos las transformaciones
Vh = torch.matmul(V, H.T)  # Dimensión (N, L)
print(Vh)
Uh = torch.matmul(U, H.T)  # Dimensión (N, L)#
print(Uh)

# Paso 2: Aplicamos las funciones de activación
tanh_Vh = torch.tanh(Vh)  # Aplicamos tanh
sigm_Uh = torch.sigmoid(Uh)  # Aplicamos sigmoide

# Paso 3: Multiplicación elemento a elemento (Gated Mechanism)
gate_output = tanh_Vh * sigm_Uh  # Dimensión (N, L)
print(f"\ngate output: \n{gate_output}")

# Paso 4: Multiplicación con el vector de pesos w y aplicación de Softmax
attention_logits = torch.matmul(w,gate_output)  # Dimensión (N, 1)
print(f"\nattention_logits: \n{attention_logits}")
attention_weights = torch.softmax(attention_logits, dim=0)  # Normalizamos con softmax
print(f"\nattention_weights: \n{attention_weights}")

# Paso 5: Obtener la salida ponderada
Z_GA = torch.sum(torch.matmul(attention_weights ,H), dim=0)  # Dimensión (M,)

# Mostramos resultados
print("Attention Weights:", attention_weights.flatten().tolist())
print("Final Output (Z_GA):", Z_GA.tolist())

