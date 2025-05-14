import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def load_model(model_path, model_class):
    """
    Carica un modello PyTorch salvato
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_metrics(metrics, save_path=None):
    """
    Visualizza le metriche di validazione
    """
    plt.figure(figsize=(12, 6))
    
    for metric_name, values in metrics.items():
        if isinstance(values, np.ndarray):
            plt.plot(values, label=metric_name)
    
    plt.title('Metriche di Validazione')
    plt.xlabel('Epoca')
    plt.ylabel('Valore')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Configurazione
    model_path = "/Users/niccologatti/Desktop/neural_ode/lamb_oseen_multivortex_model.pth"
    metrics_path = "/Users/niccologatti/Desktop/neural_ode/training_metrics.npz"  # Modifica con il percorso del file .npz delle metriche
    
    # Carica il modello
    # model = load_model(model_path, YourModelClass)  # Sostituisci YourModelClass con la tua classe modello
    
    # Carica le metriche dal file .npz
    metrics = dict(np.load(metrics_path))
    
    # Visualizza le metriche
    plot_metrics(metrics, save_path="validation_metrics.png")

if __name__ == "__main__":
    main()
