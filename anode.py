import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from config import TRAINING_CONFIG, MODEL_CONFIG, PATHS

# Carica il dataset salvato
data = torch.load(PATHS['dataset'])
data_in = data['data_in']
data_out = data['data_out']

# ANODE Model
class AugmentedLambOseenODE(nn.Module):
    def __init__(self, augment_dim=6):
        super().__init__()
        self.augment_dim = augment_dim
        self.net = nn.Sequential(
            nn.Linear(3 + augment_dim, 64),  # x, y, t, augmented dims
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2 + augment_dim)  # dx/dt, dy/dt, da/dt
        )

    def forward(self, t, state):
        xy = state[:, :2]
        aug = state[:, 2:]
        t_vec = torch.ones_like(xy[:, :1]) * t
        input_vec = torch.cat([xy, t_vec, aug], dim=1)
        deriv = self.net(input_vec)
        return deriv

# Training loop
model = AugmentedLambOseenODE(augment_dim=6)
optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
loss_fn = nn.MSELoss()
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
    aug = torch.zeros((data_in.shape[0], model.augment_dim))
    state_in = torch.cat([data_in[:, :2], aug], dim=1)
    t_in = data_in[:, 2]

    # Forward pass
    pred = model.forward(1.0, state_in)[:, :2]
    loss = loss_fn(pred, data_out)

    # Backward pass
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
np.savez(PATHS['metrics'].replace('.npz', '_anode.npz'), 
         train_losses=np.array(train_losses),
         learning_rates=np.array(learning_rates))

# Salva il modello addestrato
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'final_loss': loss.item()
}, PATHS['model'].replace('.pth', '_anode.pth'))

print("Modello e metriche salvati")
