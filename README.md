# TIER: Thermodynamic-Informational Entropic Relationships in Neural Networks

This repository contains computational models demonstrating how thermodynamic costs of information processing predict patterns of neurodegeneration in hierarchical neural networks.

## Overview

The code implements neural network simulations that track mechanical work accumulation across hierarchical processing levels, testing the hypothesis that brain regions performing the most intensive information processing accumulate damage most rapidly. These models support the theoretical framework that neurodegeneration emerges from fundamental physical constraints on biological information processing.

## Models Included

### 1. S2C2-OuterLoop.py
**Pyramidal Architecture Model**
- Implements a 3-level hierarchical neural network with convergent information flow
- Level 1: 4 primary sensory nodes (A, B, D, E)
- Level 2: 2 unimodal integration nodes (C, F) 
- Level 3: 1 heteromodal integration node (G)
- Tracks thermodynamic entropy, informational entropy, and Lyapunov stability

### 2. S2C2-OuterLoop-ColumnSparse.py
**Columnar Architecture Model**
- Implements a 3-level hierarchical network with distributed processing
- Level 1: 4 primary nodes (A, B, D, E)
- Level 2: 4 unimodal nodes (C0, C1, F0, F1)
- Level 3: 4 heteromodal nodes (G0-G3)
- Includes both layer-level and individual unit analyses

### 3. Tier-siphon-5-26-25.py
**Siphon Effect Model**
- Simulates coupled cortical-support cell populations
- Demonstrates how support systems fail before primary networks
- Models compensatory dynamics and cascading failure
- Tracks work intensity per cell and population dynamics

## Key Metrics Tracked

- **Thermodynamic Entropy**: Cumulative synaptic weight modifications (mechanical work)
- **Informational Entropy**: Uncertainty in activation patterns (Shannon entropy)
- **Lyapunov Stability**: Sensitivity to input perturbations
- **Population Dynamics**: Cell survival and failure rates over time
- **Work Intensity**: Work per surviving cell

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib
- SciPy

### Setup
```bash
# Clone the repository
git clone https://github.com/pspressman/TIER.git
cd TIER

# Install dependencies
pip install numpy matplotlib scipy
```

## Usage

### Running the Pyramidal Model
```bash
python S2C2-OuterLoop.py
```
Generates plots showing entropy dynamics across hierarchical levels over 2000 simulation runs.

### Running the Columnar Model
```bash
python S2C2-OuterLoop-ColumnSparse.py
```
Produces comparative analysis of distributed vs convergent architectures.

### Running the Siphon Effect Model
```bash
python Tier-siphon-5-26-25.py
```
Simulates support system failure dynamics and generates comprehensive visualizations.

## Output

Each model generates:
- Multi-panel figures showing temporal dynamics
- Statistical analyses with 95% confidence intervals
- Console output with summary metrics
- Starting/ending values and area under curve (AUC) calculations

## Key Parameters

### Network Models (S2C2)
- Learning rate (η): 0.1
- Iterations per run: 20
- Total runs: 2000
- Input range: [0.4, 0.6] ± 0.05
- Activation function: tanh

### Siphon Model
- Cortical cells: 1000
- Support cells: 300
- Cognitive work demand: 3000 units
- FFE (cortex): 6.0
- FFE (support): 2.0
- Time steps: 2000
- dt: 0.01

## Theoretical Background

These models test the hypothesis that:
1. Hierarchical information integration requires increasing mechanical work
2. Work accumulation drives structural failure through fatigue
3. Support systems fail before primary networks through compensatory exhaustion
4. Network architecture influences vulnerability patterns

## Citation

If you use this code in your research, please cite appropriately. 

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Corresponding Author

- Peter S. Pressman - Oregon Health & Science University

## Acknowledgments

Models and early development assisted by Anthropic's Claude and OpenAI's ChatGPT.
