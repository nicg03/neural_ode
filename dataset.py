import torch
import numpy as np

def lamb_oseen_velocity(x, y, gamma, t, nu, x0=0, y0=0):
    """
    Calcola il campo di velocità per un singolo vortice Lamb-Oseen.
    
    Args:
        x, y: Coordinate spaziali
        gamma: Intensità del vortice
        t: Tempo
        nu: Viscosità cinematica
        x0, y0: Posizione del centro del vortice
    
    Returns:
        u_x, u_y: Componenti della velocità
    """
    x = x - x0
    y = y - y0
    r2 = x**2 + y**2
    factor = gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / (4 * nu * t)))
    u_x = -y * factor
    u_y = x * factor
    return u_x, u_y

def create_multi_vortex_dataset(grid_resolution=200, domain_size=2.0, t=1.0, nu=0.01):
    """
    Crea un dataset sintetico con tre vortici Lamb-Oseen.
    
    Args:
        grid_resolution: Risoluzione della griglia
        domain_size: Dimensione del dominio [-domain_size, domain_size]
        t: Tempo di simulazione
        nu: Viscosità cinematica
    
    Returns:
        data_in: Tensor di input (x, y, t)
        data_out: Tensor di output (u_x, u_y)
        X, Y: Griglie per la visualizzazione
        u_x, u_y: Campi di velocità per la visualizzazione
    """
    # Parametri dei vortici
    gamma1, gamma2, gamma3 = 1.0, -0.8, 0.6
    x1, y1 = -0.5, 0.0
    x2, y2 = 0.5, 0.5
    x3, y3 = 0.0, -0.5

    # Creazione della griglia
    grid = np.linspace(-domain_size, domain_size, grid_resolution)
    X, Y = np.meshgrid(grid, grid)

    # Calcolo dei campi di velocità
    u_x1, u_y1 = lamb_oseen_velocity(X, Y, gamma1, t, nu, x1, y1)
    u_x2, u_y2 = lamb_oseen_velocity(X, Y, gamma2, t, nu, x2, y2)
    u_x3, u_y3 = lamb_oseen_velocity(X, Y, gamma3, t, nu, x3, y3)

    # Somma dei campi di velocità
    u_x = u_x1 + u_x2 + u_x3
    u_y = u_y1 + u_y2 + u_y3

    # Creazione dei tensori per il training
    data_in = torch.tensor(
        np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1),
        dtype=torch.float32
    )
    data_out = torch.tensor(
        np.stack([u_x.flatten(), u_y.flatten()], axis=1),
        dtype=torch.float32
    )

    return data_in, data_out, X, Y, u_x, u_y

if __name__ == "__main__":
    # Esempio di utilizzo
    data_in, data_out, X, Y, u_x, u_y = create_multi_vortex_dataset()
    print(f"Shape del dataset di input: {data_in.shape}")
    print(f"Shape del dataset di output: {data_out.shape}") 