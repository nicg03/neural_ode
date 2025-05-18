import torch
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
from config import MODEL_CONFIG, EVAL_CONFIG, PATHS
from neural_ode import LambOseenODE
from anode import AugmentedLambOseenODE

def evaluate_model(model, data_path, metrics_path, model_path, plot_paths):
    """
    Evaluate a trained model and generate visualizations.
    
    Args:
        model: The trained model to evaluate
        data_path: Path to the dataset
        metrics_path: Path to the training metrics
        model_path: Path to the model checkpoint
        plot_paths: Dictionary containing paths for saving plots
    """
    # Carica il dataset
    data = torch.load(data_path)
    data_in = data['data_in']
    data_out = data['data_out']

    # Carica le metriche di training
    metrics = np.load(metrics_path)
    train_losses = metrics['train_losses']
    learning_rates = metrics['learning_rates']

    # Carica il modello addestrato
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Imposta il modello in modalità evaluation

    # Visualizzazione della loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curva di apprendimento')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_paths['training_loss'])
    plt.close()

    # Uso della rete in Neural ODE
    x0 = torch.tensor([EVAL_CONFIG['trajectory']['initial_point']], dtype=torch.float32)
    t_eval = torch.linspace(EVAL_CONFIG['trajectory']['time_range'][0],
                           EVAL_CONFIG['trajectory']['time_range'][1],
                           EVAL_CONFIG['trajectory']['time_steps'])

    def augmented_ode_func(t, xy):
        if isinstance(model, AugmentedLambOseenODE):
            # Per ANODE, aggiungi le dimensioni augmentate
            aug = torch.zeros((xy.shape[0], model.augment_dim))
            state = torch.cat([xy, aug], dim=1)
            return model(t, state)[:, :2]  # Prendi solo le prime due dimensioni (dx/dt, dy/dt)
        else:
            # Per il modello standard
            t_tensor = torch.full((xy.shape[0], 1), t.item(), dtype=xy.dtype, device=xy.device)
            xyt = torch.cat([xy, t_tensor], dim=1)
            return model(t, xyt)

    # Calcola la traiettoria
    with torch.no_grad():
        trajectory = torchdiffeq.odeint(augmented_ode_func, x0, t_eval)
        trajectory_np = trajectory.detach().numpy()

        # Calcola il campo di velocità predetto dalla rete neurale
        grid_points = data_in
        if isinstance(model, AugmentedLambOseenODE):
            # Per ANODE, aggiungi le dimensioni augmentate
            aug = torch.zeros((grid_points.shape[0], model.augment_dim))
            state = torch.cat([grid_points[:, :2], aug], dim=1)
            predicted_velocities = model(0, state)[:, :2].detach().numpy()
        else:
            # Per il modello standard
            predicted_velocities = model(0, grid_points).detach().numpy()
            
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
    plt.savefig(plot_paths['velocity_fields'])
    plt.close()

    # Calcola e mostra l'errore medio
    # Converti i tensori in numpy array se necessario
    if isinstance(u_x, torch.Tensor):
        u_x = u_x.detach().numpy()
    if isinstance(u_y, torch.Tensor):
        u_y = u_y.detach().numpy()
    
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
    plt.savefig(plot_paths['trajectory'])
    plt.close()

    return mse

if __name__ == "__main__":
    # Inizializza il modello
    model = AugmentedLambOseenODE()
    
    # Esegui la valutazione
    evaluate_model(
        model=model,
        data_path=PATHS['dataset'],
        metrics_path=PATHS['metrics'].replace('.npz', '_anode.npz'),
        model_path=PATHS['model'].replace('.pth', '_anode.pth'),
        plot_paths=PATHS['plots']
    )
