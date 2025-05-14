import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from matplotlib.animation import FuncAnimation

# ---------------------- Parametri fisici ------------------------
GAMMA1 = 1.0   # Intensità primo vortice
GAMMA2 = -0.8  # Intensità secondo vortice
GAMMA3 = 0.6   # Intensità terzo vortice
NU = 0.01     # Viscosità
TIME = 1.0    # Tempo fisso per dataset

# ---------------------- Funzione Lamb-Oseen ---------------------
def lamb_oseen_velocity(x, y, gamma, t, nu, x0=0, y0=0):
    # Traslazione delle coordinate
    x = x - x0
    y = y - y0
    r2 = x**2 + y**2 + 1e-8  # aggiungo epsilon per evitare divisione per 0
    factor = gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / (4 * nu * t)))
    u_x = -y * factor
    u_y = x * factor
    return u_x, u_y

# ---------------------- Dataset sintetico -----------------------
def generate_dataset(grid_resolution=200):
    grid = np.linspace(-2, 2, grid_resolution)
    X, Y = np.meshgrid(grid, grid)
    
    # Posizioni dei tre vortici
    x1, y1 = -0.5, 0.0
    x2, y2 = 0.5, 0.5
    x3, y3 = 0.0, -0.5
    
    # Calcolo del campo di velocità totale
    u_x1, u_y1 = lamb_oseen_velocity(X, Y, GAMMA1, TIME, NU, x1, y1)
    u_x2, u_y2 = lamb_oseen_velocity(X, Y, GAMMA2, TIME, NU, x2, y2)
    u_x3, u_y3 = lamb_oseen_velocity(X, Y, GAMMA3, TIME, NU, x3, y3)
    
    # Somma dei campi di velocità
    u_x = u_x1 + u_x2 + u_x3
    u_y = u_y1 + u_y2 + u_y3
    
    data_in = torch.tensor(np.stack([X.flatten(), Y.flatten(), np.full(X.size, TIME)], axis=1), dtype=torch.float32)
    data_out = torch.tensor(np.stack([u_x.flatten(), u_y.flatten()], axis=1), dtype=torch.float32)
    return data_in, data_out, X, Y, u_x, u_y

# ---------------------- ANODE Model -----------------------------
class AugmentedLambOseenODE(nn.Module):
    def __init__(self, augment_dim=8):
        super().__init__()
        self.augment_dim = augment_dim
        self.net = nn.Sequential(
            nn.Linear(3 + augment_dim, 128),  # x, y, t, augmented dims
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2 + augment_dim)  # dx/dt, dy/dt, da/dt
        )

    def forward(self, t, state):
        xy = state[:, :2]
        aug = state[:, 2:]
        t_vec = torch.ones_like(xy[:, :1]) * t
        input_vec = torch.cat([xy, t_vec, aug], dim=1)
        deriv = self.net(input_vec)
        return deriv

# ---------------------- Training --------------------------------
def train_anode(model, data_in, data_out, epochs=1000, lr=1e-3, optimizer=None, scheduler=None):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    
    loss_fn = nn.MSELoss()
    
    # Liste per memorizzare le metriche
    train_losses = []
    learning_rates = []
    
    for epoch in range(epochs):
        aug = torch.zeros((data_in.shape[0], model.augment_dim))
        state_in = torch.cat([data_in[:, :2], aug], dim=1)
        t_in = data_in[:, 2]

        # In questo caso non integriamo, ma apprendiamo la derivata diretta
        pred = model.forward(TIME, state_in)[:, :2]
        loss = loss_fn(pred, data_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        
        # Salva le metriche
        train_losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    return train_losses, learning_rates

# ---------------------- Simulazione Traiettoria -----------------
def simulate_trajectory(model, x0, y0, t_eval):
    aug_init = torch.zeros((1, model.augment_dim))
    state0 = torch.cat([torch.tensor([[x0, y0]], dtype=torch.float32), aug_init], dim=1)
    trajectory = odeint(model, state0, t_eval)
    return trajectory[:, 0, :2].detach().numpy()

# ---------------------- Animazione Particelle -------------------
def animate_particles(model, n_particles=100, t_max=5.0, fps=30):
    # Genera posizioni iniziali casuali per le particelle
    x0 = np.random.uniform(-1.0, 1.0, n_particles)
    y0 = np.random.uniform(-1.0, 1.0, n_particles)
    
    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Animazione Particelle nel Vortice')
    
    # Inizializza i punti
    points = ax.scatter(x0, y0, c='blue', alpha=0.6)
    center = ax.scatter(0, 0, c='red', label='Centro Vortice')
    ax.legend()
    
    # Pre-calcola tutte le traiettorie
    trajectories = []
    t_eval = torch.linspace(0, t_max, int(t_max * fps))
    for i in range(n_particles):
        trajectory = simulate_trajectory(model, x0[i], y0[i], t_eval)
        trajectories.append(trajectory)
    trajectories = np.array(trajectories)
    
    # Funzione di aggiornamento per l'animazione
    def update(frame):
        points.set_offsets(trajectories[:, frame])
        return points,
    
    # Crea l'animazione
    anim = FuncAnimation(fig, update, frames=len(t_eval),
                        interval=1000/fps, blit=True)
    return anim

# ---------------------- Main ------------------------------------
if __name__ == "__main__":
    # Genera dataset
    data_in, data_out, X, Y, u_x, u_y = generate_dataset(grid_resolution=200)

    # Crea e allena ANODE
    augment_dim = 6
    model = AugmentedLambOseenODE(augment_dim=augment_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    train_losses, learning_rates = train_anode(model, data_in, data_out, epochs=20000, lr=1e-3, optimizer=optimizer, scheduler=scheduler)

    # Salva le metriche
    np.savez('anode_training_metrics.npz', 
             train_losses=np.array(train_losses),
             learning_rates=np.array(learning_rates))

    # Salva il modello addestrato
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_loss': train_losses[-1]
    }, 'anode_multivortex_model.pth')

    print("Modello e metriche ANODE salvati")

    # Visualizzazione della loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curva di apprendimento ANODE')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('anode_training_loss.png')
    plt.close()

    # Calcola il campo di velocità predetto
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten(), np.full(X.size, TIME)], axis=1), dtype=torch.float32)
    aug = torch.zeros((grid_points.shape[0], model.augment_dim))
    state_in = torch.cat([grid_points[:, :2], aug], dim=1)
    predicted_velocities = model.forward(TIME, state_in)[:, :2].detach().numpy()
    predicted_u_x = predicted_velocities[:, 0].reshape(X.shape)
    predicted_u_y = predicted_velocities[:, 1].reshape(X.shape)

    # Visualizzazione
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot del campo di velocità predetto
    ax1.quiver(X, Y, predicted_u_x, predicted_u_y, scale=50)
    ax1.set_title('Campo di velocità predetto (ANODE)')
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
    plt.savefig('anode_velocity_field_comparison.png')
    plt.show()

    # Calcola e mostra l'errore medio
    mse = np.mean((predicted_u_x - u_x)**2 + (predicted_u_y - u_y)**2)
    print(f"Errore quadratico medio ANODE: {mse:.6f}")

    # Crea e mostra l'animazione
    anim = animate_particles(model, n_particles=100, t_max=5.0)
    plt.show()
