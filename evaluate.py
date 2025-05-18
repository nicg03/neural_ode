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

    # Visualizzazione delle loss e learning rates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot training losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Curva di apprendimento')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()

    # Plot learning rates
    ax2.plot(learning_rates, label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_paths['training_loss'])
    plt.show()
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
    plt.show()
    plt.close()

    # Calcola e mostra l'errore medio
    # Converti i tensori in numpy array se necessario
    if isinstance(u_x, torch.Tensor):
        u_x = u_x.detach().numpy()
    if isinstance(u_y, torch.Tensor):
        u_y = u_y.detach().numpy()
    
    mse = np.mean((predicted_u_x - u_x)**2 + (predicted_u_y - u_y)**2)
    print(f"Errore quadratico medio: {mse:.6f}")

    # Visualizzazione della traiettoria con campo vettoriale
    plt.figure(figsize=(10, 10))
    
    # Plot del campo vettoriale di sfondo
    skip = 15  # Densità più bassa per il campo vettoriale di sfondo
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               u_x[::skip, ::skip], u_y[::skip, ::skip], 
               scale=50, color='gray', alpha=0.3, label='Campo di velocità')
    
    # Plot della traiettoria
    plt.plot(trajectory_np[:, 0, 0], trajectory_np[:, 0, 1], 'b-', 
             linewidth=2, label='Traiettoria')
    plt.plot(trajectory_np[0, 0, 0], trajectory_np[0, 0, 1], 'go', 
             markersize=10, label='Punto iniziale')
    plt.plot(trajectory_np[-1, 0, 0], trajectory_np[-1, 0, 1], 'ro', 
             markersize=10, label='Punto finale')
    
    plt.title('Traiettoria della particella nel campo di velocità')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.savefig(plot_paths['trajectory'])
    plt.show()
    plt.close()

    return mse

if __name__ == "__main__":
    # Scegli il modello da valutare
    model_type = "standard"  # oppure "augmented"
    
    if model_type == "standard":
        model = LambOseenODE()
        metrics_path = PATHS['metrics']
        model_path = PATHS['model']
    else:  # augmented
        model = AugmentedLambOseenODE()
        metrics_path = PATHS['metrics'].replace('.npz', '_anode.npz')
        model_path = PATHS['model'].replace('.pth', '_anode.pth')
    
    # Esegui la valutazione
    evaluate_model(
        model=model,
        data_path=PATHS['dataset'],
        metrics_path=metrics_path,
        model_path=model_path,
        plot_paths=PATHS['plots']
    )
