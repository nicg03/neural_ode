import torch
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt

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

data_in = torch.tensor(np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1), dtype=torch.float32)
data_out = torch.tensor(np.stack([u_x.flatten(), u_y.flatten()], axis=1), dtype=torch.float32)

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
optimizer = torch.optim.Adam(ode_func.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

# Liste per memorizzare le metriche
train_losses = []
learning_rates = []

for epoch in range(16000):
    pred = ode_func(0, data_in)
    loss = loss_fn(pred, data_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
    # Salva le metriche
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

# Salva le metriche
np.savez('training_metrics.npz', 
         train_losses=np.array(train_losses),
         learning_rates=np.array(learning_rates))

# Salva il modello addestrato
torch.save({
    'model_state_dict': ode_func.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'final_loss': loss.item()
}, 'lamb_oseen_multivortex_model.pth')

print("Modello e metriche salvati")

# Visualizzazione della loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Curva di apprendimento')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig('training_loss.png')
plt.close()

# Uso della rete in Neural ODE
x0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # Punto iniziale al centro
t_eval = torch.linspace(0, 2, 100)

def augmented_ode_func(t, xy):
    t_tensor = torch.full((xy.shape[0], 1), t.item(), dtype=xy.dtype, device=xy.device)
    xyt = torch.cat([xy, t_tensor], dim=1)
    return ode_func(t, xyt)

trajectory = torchdiffeq.odeint(augmented_ode_func, x0, t_eval)
trajectory_np = trajectory.detach().numpy()

# Calcola il campo di velocità predetto dalla rete neurale
grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1), dtype=torch.float32)
predicted_velocities = ode_func(0, grid_points).detach().numpy()
predicted_u_x = predicted_velocities[:, 0].reshape(X.shape)
predicted_u_y = predicted_velocities[:, 1].reshape(X.shape)

# Visualizzazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot del campo di velocità predetto
ax1.quiver(X, Y, predicted_u_x, predicted_u_y, scale=50)
ax1.set_title('Campo di velocità predetto')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

# Plot del campo di velocità reale
ax2.quiver(X, Y, u_x, u_y, scale=50)
ax2.set_title('Campo di velocità reale (3 vortici Lamb-Oseen)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Calcola e mostra l'errore medio
mse = np.mean((predicted_u_x - u_x)**2 + (predicted_u_y - u_y)**2)
print(f"Errore quadratico medio: {mse:.6f}")
