# DataDrift-Project

Questo progetto analizza e misura il "data drift" tra dataset di immagini (principalmente ImageNet-1k, 
ImageNet-C, ImageNet-R) e valuta l'impatto del drift sulla robustezza dei modelli.
Il repository contiene script per preprocessamento dei dataset, analisi esplorativa dei dati (EDA),
metriche di drift e testing di modelli su dataset degradati.


Se servono GPU/torch con supporto CUDA, installare la versione di `torch` compatibile con la tua GPU/driver come indicato sulla pagina ufficiale di PyTorch.

## Struttura della repository

Principali cartelle e file:

- `data_preprocess/` - script per preprocessare i dataset.
- `data_analysis/` - notebook e script per analisi dei dataset, metriche di drift e utilità.
- `model_testing/` - script per testare modelli su dataset degradati (`test_model_ImageNetC.py`, `test_model_ImageNetR.py`).
- `eda_outputs/` - immagini, tabelle e grafici prodotti dall'EDA.
- `results/` - risultati sperimentali salvati (per dataset e configurazioni).

## Dataset

Il repository non include i dataset originali per ragioni di dimensione.
Per scaricare i dataset, esegui lo script di preprocessing.

## Analisi dei dati (EDA) e metriche di drift

I notebook principali si trovano in `data_analysis/`:

- `dataset_analysis.ipynb` - analisi esplorativa della distribuzione delle classi e analisi del drift su feature visive.
- `metrics_analysis.ipynb` - calcolo e confronto di metriche di drift MMD sulle feature estratte dal modello.
