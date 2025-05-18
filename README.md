# Neural ODE per la Simulazione della Dinamica dei Vortici

Questo repository contiene l'implementazione delle Equazioni Differenziali Ordinarie Neurali (Neural ODEs) per la simulazione della dinamica dei vortici, con particolare attenzione al modello di vortice di Lamb-Oseen. Il progetto fa parte di un lavoro di tesi sull'applicazione di tecniche di deep learning ai problemi di fluidodinamica.

## Panoramica del Progetto

Il progetto implementa due approcci diversi:
1. Neural ODE Standard
2. Augmented Neural Ode (ANODE)

Entrambi i modelli sono addestrati per apprendere il campo di velocità generato da vortici multipli di Lamb-Oseen, che è un modello fondamentale nella fluidodinamica per lo studio del comportamento dei vortici.

## Struttura del Repository

```
.
├── config.py           # Parametri di configurazione
├── dataset.py         # Generazione e gestione del dataset
├── neural_ode.py      # Implementazione Neural ODE standard
├── anode.py          # Implementazione Neural ODE aumentata
├── evaluate.py       # Valutazione e visualizzazione del modello
└── data/             # Directory per i dataset

```

## Requisiti

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- torchdiffeq

## Installazione

1. Clona il repository:
```bash
git clone [url-repository]
cd neural_ode
```

2. Installa i pacchetti necessari:
```bash
pip install -r requirements.txt
```

## Utilizzo

1. Genera il dataset:
```bash
python dataset.py
```

2. Addestra i modelli:
```bash
# Per Neural ODE standard
python neural_ode.py

# Per Neural ODE aumentata
python anode.py
```

3. Valuta i modelli:
```bash
python evaluate.py
```

## Configurazione

Il file `config.py` contiene tutti i parametri di configurazione per:
- Impostazioni di training
- Generazione del dataset
- Architettura del modello
- Parametri di valutazione
- Percorsi dei file

## Risultati

Lo script di valutazione genera tre tipi di visualizzazioni:
1. Curva della loss di training
2. Confronto dei campi di velocità (predetti vs. reali)
3. Traiettoria della particella nel campo di velocità
