import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from anode import AugmentedLambOseenODE, lamb_oseen_velocity, NU, TIME

N_VORTICES = 300

class LambOseenODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, t, xyt):
        return self.net(xyt)

def generate_vortices(n_vortices=N_VORTICES):
    """Generate random positions and intensities for vortices"""
    # Generate random positions in the domain [-2, 2] x [-2, 2]
    x_positions = np.random.uniform(-2, 2, n_vortices)
    y_positions = np.random.uniform(-2, 2, n_vortices)
    
    # Generate random intensities between -1 and 1
    intensities = np.random.uniform(-1, 1, n_vortices)
    
    return x_positions, y_positions, intensities

def generate_points(n_points=1000000):
    """Generate random points in the domain [-2, 2] x [-2, 2]"""
    x = np.random.uniform(-2, 2, n_points)
    y = np.random.uniform(-2, 2, n_points)
    return x, y

def compute_analytical_solution(x, y, n_vortices=N_VORTICES):
    """Compute the analytical solution using the Lamb-Oseen velocity field"""
    # Generate vortices
    x_positions, y_positions, intensities = generate_vortices(n_vortices)
    
    # Initialize velocity fields
    u_x = np.zeros_like(x)
    u_y = np.zeros_like(y)
    
    # Compute velocity field for each vortex
    for i in range(len(x_positions)):
        u_x_i, u_y_i = lamb_oseen_velocity(x, y, intensities[i], TIME, NU, x_positions[i], y_positions[i])
        u_x += u_x_i
        u_y += u_y_i
    
    return u_x, u_y

def plot_velocity_fields(x, y, u_x, u_y, anode_pred, lamb_pred, title="Velocity Fields Comparison"):
    """Plot the velocity fields for comparison"""
    # Reshape the predictions to match the grid
    grid_size = int(np.sqrt(len(x)))
    X = x.reshape(grid_size, grid_size)
    Y = y.reshape(grid_size, grid_size)
    
    u_x_reshaped = u_x.reshape(grid_size, grid_size)
    u_y_reshaped = u_y.reshape(grid_size, grid_size)
    anode_u_x = anode_pred[:, 0].reshape(grid_size, grid_size)
    anode_u_y = anode_pred[:, 1].reshape(grid_size, grid_size)
    lamb_u_x = lamb_pred[:, 0].reshape(grid_size, grid_size)
    lamb_u_y = lamb_pred[:, 1].reshape(grid_size, grid_size)
    
    plt.figure(figsize=(20, 6))
    
    # Plot analytical solution
    plt.subplot(131)
    plt.quiver(X, Y, u_x_reshaped, u_y_reshaped, scale=50)
    plt.title('Soluzione Analitica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    # Plot ANODE prediction
    plt.subplot(132)
    plt.quiver(X, Y, anode_u_x, anode_u_y, scale=50)
    plt.title('Predizione ANODE')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    # Plot Lamb-Oseen prediction
    plt.subplot(133)
    plt.quiver(X, Y, lamb_u_x, lamb_u_y, scale=50)
    plt.title('Predizione Neural ODE')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('velocity_fields_comparison.png')
    plt.show()  # This will display the plot

def plot_speedup(anode_time, lamb_time, analytical_time):
    """Plot bar chart of speedup factors"""
    methods = ['Soluzione Analitica', 'Augmented Neural ODE', 'Neural ODE']
    speedups = [1.0, analytical_time/anode_time, analytical_time/lamb_time]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, speedups, color=['#e74c3c', '#2ecc71', '#3498db'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom')
    
    plt.title('Fattore di Speedup Comparison')
    plt.ylabel('Fattore di Speedup (volte più veloce)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=1 to show the baseline
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    
    plt.savefig('speedup_comparison.png')
    plt.show()

def plot_speedup_vs_vortices():
    """Plot speedup factors as a function of number of vortices"""
    # Generate range of vortex numbers to test
    n_vortices_range = np.linspace(1, 2000, 20, dtype=int)  # 20 points for smooth curve
    n_points = 1000000  # Same number of points as in main() for fair comparison
    
    # Initialize arrays to store results
    anode_speedups = []
    lamb_speedups = []
    analytical_times = []
    
    print("\nComputing speedups for different numbers of vortices...")
    
    # Load models once
    anode_checkpoint = torch.load('anode_multivortex_model.pth')
    anode_model = AugmentedLambOseenODE(augment_dim=6)
    anode_model.load_state_dict(anode_checkpoint['model_state_dict'])
    anode_model.eval()
    
    lamb_checkpoint = torch.load('lamb_oseen_multivortex_model.pth')
    lamb_model = LambOseenODE()
    lamb_model.load_state_dict(lamb_checkpoint['model_state_dict'])
    lamb_model.eval()
    
    # Generate fixed set of points
    x, y = generate_points(n_points)
    data_in = torch.tensor(np.stack([x, y, np.full_like(x, TIME)], axis=1), dtype=torch.float32)
    
    # Prepare ANODE input
    aug = torch.zeros((data_in.shape[0], anode_model.augment_dim))
    state_in = torch.cat([data_in[:, :2], aug], dim=1)
    
    # Time the model predictions
    print("Timing model predictions...")
    start_time = time.time()
    with torch.no_grad():
        anode_predicted_velocities = anode_model.forward(TIME, state_in)[:, :2]
    anode_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        lamb_predicted_velocities = lamb_model.forward(0, data_in)
    lamb_time = time.time() - start_time
    
    print(f"ANODE model time: {anode_time:.4f} seconds")
    print(f"Lamb-Oseen model time: {lamb_time:.4f} seconds")
    
    # Test different numbers of vortices
    for n_vortices in n_vortices_range:
        print(f"Testing {n_vortices} vortices...")
        
        # Time analytical solution
        start_time = time.time()
        u_x, u_y = compute_analytical_solution(x, y, n_vortices)
        analytical_time = time.time() - start_time
        analytical_times.append(analytical_time)
        
        # Calculate speedups
        anode_speedups.append(analytical_time / anode_time)
        lamb_speedups.append(analytical_time / lamb_time)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(n_vortices_range, [1.0] * len(n_vortices_range), 'r--', label='Soluzione Analitica', alpha=0.5)
    plt.plot(n_vortices_range, anode_speedups, 'g-', label='Augmented Neural ODE', linewidth=2)
    plt.plot(n_vortices_range, lamb_speedups, 'b-', label='Neural ODE', linewidth=2)
    
    plt.title('Fattore di Speedup vs Numero di Vortici')
    plt.xlabel('Numero di Vortici')
    plt.ylabel('Fattore di Speedup (volte più veloce)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig('speedup_vs_vortices.png')
    plt.show()

def main():
    # Load both models
    print("Loading the models...")
    
    # Load ANODE model
    anode_checkpoint = torch.load('anode_multivortex_model.pth')
    anode_model = AugmentedLambOseenODE(augment_dim=6)
    anode_model.load_state_dict(anode_checkpoint['model_state_dict'])
    anode_model.eval()
    
    # Load Lamb-Oseen model
    lamb_checkpoint = torch.load('lamb_oseen_multivortex_model.pth')
    lamb_model = LambOseenODE()
    lamb_model.load_state_dict(lamb_checkpoint['model_state_dict'])
    lamb_model.eval()
    
    # Generate points
    print("Generating 1,000,000 random points...")
    x, y = generate_points(1000000)
    
    # Time the ANODE prediction
    print("\nComputing ANODE prediction...")
    start_time = time.time()
    
    # Prepare input for ANODE
    data_in = torch.tensor(np.stack([x, y, np.full_like(x, TIME)], axis=1), dtype=torch.float32)
    aug = torch.zeros((data_in.shape[0], anode_model.augment_dim))
    state_in = torch.cat([data_in[:, :2], aug], dim=1)
    
    # Get ANODE prediction
    with torch.no_grad():
        anode_predicted_velocities = anode_model.forward(TIME, state_in)[:, :2]
    
    anode_time = time.time() - start_time
    print(f"ANODE computation time: {anode_time:.4f} seconds")
    
    # Time the Lamb-Oseen prediction
    print("\nComputing Lamb-Oseen prediction...")
    start_time = time.time()
    
    # Prepare input for Lamb-Oseen
    lamb_input = torch.tensor(np.stack([x, y, np.full_like(x, TIME)], axis=1), dtype=torch.float32)
    
    # Get Lamb-Oseen prediction
    with torch.no_grad():
        lamb_predicted_velocities = lamb_model.forward(0, lamb_input)
    
    lamb_time = time.time() - start_time
    print(f"Lamb-Oseen computation time: {lamb_time:.4f} seconds")
    
    # Time the analytical solution
    print("\nComputing analytical solution...")
    start_time = time.time()
    u_x, u_y = compute_analytical_solution(x, y)
    analytical_time = time.time() - start_time
    print(f"Analytical solution computation time: {analytical_time:.4f} seconds")
    
    # Calculate speedups
    anode_speedup = analytical_time / anode_time
    lamb_speedup = analytical_time / lamb_time
    print(f"\nAugmented Neural ODE is {anode_speedup:.2f}x faster than the analytical solution")
    print(f"Neural ODE is {lamb_speedup:.2f}x faster than the analytical solution")
    
    # Calculate total squared errors
    anode_error = np.sum((anode_predicted_velocities[:, 0].numpy() - u_x)**2 + 
                        (anode_predicted_velocities[:, 1].numpy() - u_y)**2)
    lamb_error = np.sum((lamb_predicted_velocities[:, 0].numpy() - u_x)**2 + 
                       (lamb_predicted_velocities[:, 1].numpy() - u_y)**2)
    
    print(f"\nTotal squared errors:")
    print(f"Augmented Neural ODE total error: {anode_error:.6f}")
    print(f"Neural ODE total error: {lamb_error:.6f}")
    
    # Plot speedup comparison
    plot_speedup(anode_time, lamb_time, analytical_time)
    
    # Plot speedup vs vortices
    plot_speedup_vs_vortices()
    
    # Plot velocity fields for a subset of points (for visualization)
    subset_size = 10000  # Use a smaller subset for visualization
    indices = np.random.choice(len(x), subset_size, replace=False)
    plot_velocity_fields(x[indices], y[indices], 
                        u_x[indices], u_y[indices],
                        anode_predicted_velocities[indices].numpy(),
                        lamb_predicted_velocities[indices].numpy(),
                        "Velocity Fields Comparison (10,000 points)")

if __name__ == "__main__":
    main() 