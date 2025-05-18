import torch
import numpy as np
from config import TRAINING_CONFIG, MODEL_CONFIG, PATHS

# Carica il dataset salvato
data = torch.load(PATHS['dataset'])
data_in = data['data_in']
data_out = data['data_out']

# Funzione esatta del Lamb-Oseen per un singolo vortice
def lamb_oseen_velocity(x, y, gamma, t, nu, x0=0, y0=0):
    # Traslazione delle coordinate
    x = x - x0
    y = y - y0
    r2 = x**2 + y**2
    factor = gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / (4 * nu * t)))
    u_x = -y * factor
    u_y = x * factor
    return u_x, u_y

# Dataset sintetico con tre vortici
gamma1, gamma2, gamma3 = 1.0, -0.8, 0.6  # Intensità dei vortici
nu = 0.01
t = 1.0
grid_resolution = 200
grid = np.linspace(-2, 2, grid_resolution)  # Estendo il dominio
X, Y = np.meshgrid(grid, grid)

# Posizioni dei tre vortici
x1, y1 = -0.5, 0.0
x2, y2 = 0.5, 0.5
x3, y3 = 0.0, -0.5

# Calcolo del campo di velocità totale
u_x1, u_y1 = lamb_oseen_velocity(X, Y, gamma1, t, nu, x1, y1)
u_x2, u_y2 = lamb_oseen_velocity(X, Y, gamma2, t, nu, x2, y2)
u_x3, u_y3 = lamb_oseen_velocity(X, Y, gamma3, t, nu, x3, y3)

# Somma dei campi di velocità
u_x = u_x1 + u_x2 + u_x3
u_y = u_y1 + u_y2 + u_y3

# Neural ODE dynamics
class LambOseenODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),  # Aumentato il numero di neuroni
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, t, xyt):
        x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
        velocities = self.net(xyt)
        return velocities

# Training loop
ode_func = LambOseenODE()
optimizer = torch.optim.Adam(ode_func.parameters(), lr=TRAINING_CONFIG['learning_rate'])
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    'min', 
    patience=TRAINING_CONFIG['scheduler_patience'], 
    factor=TRAINING_CONFIG['scheduler_factor']
)

# Liste per memorizzare le metriche
train_losses = []
learning_rates = []

for epoch in range(TRAINING_CONFIG['num_epochs']):
    pred = ode_func(0, data_in)
    loss = loss_fn(pred, data_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
    # Salva le metriche
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    if epoch % TRAINING_CONFIG['print_every'] == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

# Salva le metriche
np.savez(PATHS['metrics'], 
         train_losses=np.array(train_losses),
         learning_rates=np.array(learning_rates))

# Salva il modello addestrato
torch.save({
    'model_state_dict': ode_func.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'final_loss': loss.item()
}, PATHS['model'])

print("Modello e metriche salvati")
