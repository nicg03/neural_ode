import torch
import numpy as np
import os
from config import DATASET_CONFIG, VORTEX_CONFIG, PATHS

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

def create_multi_vortex_dataset():
    """
    Crea un dataset sintetico con tre vortici Lamb-Oseen.
    
    Returns:
        data_in: Tensor di input (x, y, t)
        data_out: Tensor di output (u_x, u_y)
        X, Y: Griglie per la visualizzazione
        u_x, u_y: Campi di velocità per la visualizzazione
    """
    # Estrai i parametri dalla configurazione
    grid_resolution = DATASET_CONFIG['grid_resolution']
    domain_size = DATASET_CONFIG['domain_size']
    t = DATASET_CONFIG['time']
    nu = DATASET_CONFIG['viscosity']

    # Creazione della griglia
    grid = np.linspace(-domain_size, domain_size, grid_resolution)
    X, Y = np.meshgrid(grid, grid)

    # Calcolo dei campi di velocità per ogni vortice
    u_x1, u_y1 = lamb_oseen_velocity(X, Y, VORTEX_CONFIG['vortex1']['gamma'], t, nu,
                                   VORTEX_CONFIG['vortex1']['x'], VORTEX_CONFIG['vortex1']['y'])
    u_x2, u_y2 = lamb_oseen_velocity(X, Y, VORTEX_CONFIG['vortex2']['gamma'], t, nu,
                                   VORTEX_CONFIG['vortex2']['x'], VORTEX_CONFIG['vortex2']['y'])
    u_x3, u_y3 = lamb_oseen_velocity(X, Y, VORTEX_CONFIG['vortex3']['gamma'], t, nu,
                                   VORTEX_CONFIG['vortex3']['x'], VORTEX_CONFIG['vortex3']['y'])

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

def save_dataset(data_in, data_out, X, Y, u_x, u_y):
    """
    Salva il dataset e i dati per la visualizzazione.
    
    Args:
        data_in: Tensor di input (x, y, t)
        data_out: Tensor di output (u_x, u_y)
        X, Y: Griglie per la visualizzazione
        u_x, u_y: Campi di velocità per la visualizzazione
    """
    # Crea la directory se non esiste
    os.makedirs(os.path.dirname(PATHS['dataset']), exist_ok=True)
    
    # Salva i tensori per il training
    torch.save({
        'data_in': data_in,
        'data_out': data_out
    }, PATHS['dataset'])
    
    print(f"Dataset salvato in {PATHS['dataset']}")

if __name__ == "__main__":
    # Esempio di utilizzo
    data_in, data_out, X, Y, u_x, u_y = create_multi_vortex_dataset()
    print(f"Shape del dataset di input: {data_in.shape}")
    print(f"Shape del dataset di output: {data_out.shape}")
    
    # Salva il dataset
    save_dataset(data_in, data_out, X, Y, u_x, u_y) 