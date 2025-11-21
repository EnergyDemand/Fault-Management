## Overview 

This repository contains the official implementation of the paper:

“Explainable fault classification mechanism using solar data based on ensemble paradigm for efficient hydrogen energy production.”

Green hydrogen production relies heavily on the continuous and efficient operation of solar panels. However, traditional machine-learning-based fault classification methods struggle with imbalanced datasets, poor generalization, and limited explainability, which can lead to incorrect fault detection and reduced hydrogen production efficiency.

To address these challenges, this project introduces the Explainable Ensemble Fault Classification (EEFC) framework—an advanced voting-based ensemble model integrating multiple high-performing classifiers. EEFC is designed to deliver robust, balanced fault classification, especially in scenarios where rare fault types are critical.

## Features

- Data ingestion and preprocessing (see the `data/` folder)  
- Experimentation workflows (`experiment_ensemble.py`) for modelling fault scenarios  
- Results generation and reporting (see `result/` folder)  
- Modular code structure under `src/` for maintainability and extension  
- Helps folder for utilities, scripts, and documentation under `helps/`
