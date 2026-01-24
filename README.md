# MLX
# Usable Machine Learning Models (NumPy-based)

## Overview
This repository is dedicated to building a package of **Machine Learning models** that are actually usable.  
While they may not match industrial standards at the very start, they are designed with unique features that ensure clarity, consistency, and accessibility.

## Key Features
- **Standardized APIs across all models**:
  - `.train(training_examples, labels)` → trains the model (for supervised learning).
  - `.tell(data)` → generates predictions or outputs for given input data.
  - `.plot(X_data, Y_data)` → visualizes predictions vs actual outputs, along with the model’s cost convergence.

- **Consistent instantiation pattern**:
  ```python
  model = ClassName(*parameters)

## Usage:
    Clone the repository and install dependencies:
    **git clone https://github.com/Sanhik-2/ML_Journey.git**
    cd ML_Journey
    pip install -r requirements.txt
    Run the example script:
    python src/main.py


