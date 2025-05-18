import torch
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
from config import MODEL_CONFIG, EVAL_CONFIG, PATHS

# Carica il dataset
data = torch.load(PATHS['dataset'])
data_in = data['data_in']
data_out = data['data_out']

# Carica le metriche di training
metrics = np.load(PATHS['metrics'])
train_losses = metrics['train_losses']
learning_rates = metrics['learning_rates']

# Carica il modello addestrato
checkpoint = torch.load(PATHS['model'])

# Neural ODE dynamics
class LambOseenODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        # Input layer
        layers.append(torch.nn.Linear(MODEL_CONFIG['input_size'], MODEL_CONFIG['hidden_size']))
        layers.append(torch.nn.Tanh())
        
        # Hidden layers
        for _ in range(MODEL_CONFIG['num_hidden_layers'] - 1):
            layers.append(torch.nn.Linear(MODEL_CONFIG['hidden_size'], MODEL_CONFIG['hidden_size']))
            layers.append(torch.nn.Tanh())
        
        # Output layer
        layers.append(torch.nn.Linear(MODEL_CONFIG['hidden_size'], MODEL_CONFIG['output_size']))
        
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t, xyt):
        x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
        velocities = self.net(xyt)
        return velocities

# Inizializza e carica il modello
ode_func = LambOseenODE()
ode_func.load_state_dict(checkpoint['model_state_dict'])
ode_func.eval()  # Imposta il modello in modalità evaluation

# Visualizzazione della loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva di apprendimento')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig(PATHS['plots']['training_loss'])
plt.close()

# Uso della rete in Neural ODE
x0 = torch.tensor([EVAL_CONFIG['trajectory']['initial_point']], dtype=torch.float32)
t_eval = torch.linspace(EVAL_CONFIG['trajectory']['time_range'][0],
                       EVAL_CONFIG['trajectory']['time_range'][1],
                       EVAL_CONFIG['trajectory']['time_steps'])

def augmented_ode_func(t, xy):
    t_tensor = torch.full((xy.shape[0], 1), t.item(), dtype=xy.dtype, device=xy.device)
    xyt = torch.cat([xy, t_tensor], dim=1)
    return ode_func(t, xyt)

# Calcola la traiettoria
with torch.no_grad():
    trajectory = torchdiffeq.odeint(augmented_ode_func, x0, t_eval)
    trajectory_np = trajectory.detach().numpy()

    # Calcola il campo di velocità predetto dalla rete neurale
    grid_points = data_in
    predicted_velocities = ode_func(0, grid_points).detach().numpy()
    predicted_u_x = predicted_velocities[:, 0].reshape((200, 200))
    predicted_u_y = predicted_velocities[:, 1].reshape((200, 200))

# Visualizzazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot del campo di velocità predetto
X = data_in[:, 0].reshape((200, 200))
Y = data_in[:, 1].reshape((200, 200))
ax1.quiver(X, Y, predicted_u_x, predicted_u_y, scale=50)
ax1.set_title('Campo di velocità predetto')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

# Plot del campo di velocità reale
u_x = data_out[:, 0].reshape((200, 200))
u_y = data_out[:, 1].reshape((200, 200))
ax2.quiver(X, Y, u_x, u_y, scale=50)
ax2.set_title('Campo di velocità reale (3 vortici Lamb-Oseen)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.tight_layout()
plt.savefig(PATHS['plots']['velocity_fields'])
plt.close()

# Calcola e mostra l'errore medio
mse = np.mean((predicted_u_x - u_x)**2 + (predicted_u_y - u_y)**2)
print(f"Errore quadratico medio: {mse:.6f}")

# Visualizzazione della traiettoria
plt.figure(figsize=(8, 8))
plt.plot(trajectory_np[:, 0, 0], trajectory_np[:, 0, 1], 'b-', label='Traiettoria')
plt.plot(trajectory_np[0, 0, 0], trajectory_np[0, 0, 1], 'go', label='Punto iniziale')
plt.plot(trajectory_np[-1, 0, 0], trajectory_np[-1, 0, 1], 'ro', label='Punto finale')
plt.title('Traiettoria del punto nel campo di velocità')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig(PATHS['plots']['trajectory'])
plt.close()
