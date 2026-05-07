# HUBO/QAOA Feature Selection

Small Python implementation of HUBO-style feature selection with QAOA.

The repository intentionally contains no dataset-specific code. The main
function accepts your own feature matrix `X` and target vector `y`.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from quantum_hubo_feature_selection import hubo_qaoa_feature_selection

result = hubo_qaoa_feature_selection(
    X,
    y,
    top_k=6,
    device="cpu",
    p_layers=1,
    trials=8,
    maxiter=80,
    shots=4096,
    rho=0.2,
    seed=42,
)

print(result.selected_features)
print(result.ranking)
```

`X` can be a pandas `DataFrame` or a NumPy array. If `X` is a DataFrame with
categorical columns, they are one-hot encoded by default.

## What It Does

1. Estimates feature relevance with mutual information `MI(X_i, y)`.
2. Estimates pair redundancy with `MI(X_i, X_j)`.
3. Estimates triple redundancy with cyclic mutual information.
4. Builds a binary HUBO objective where `x_i = 1` means feature `i` is selected.
5. Converts the HUBO objective to an Ising Hamiltonian with `x_i = (1 - Z_i) / 2`.
6. Runs QAOA on a statevector simulator.
7. Ranks features by frequency among low-energy samples.

## CPU/GPU

CPU mode uses Qiskit `Statevector`:

```python
result = hubo_qaoa_feature_selection(X, y, device="cpu")
```

GPU mode tries Qiskit Aer GPU and falls back to CPU if unavailable:

```python
result = hubo_qaoa_feature_selection(
    X,
    y,
    device="gpu",
    aer_precision="single",
)
```

For small feature counts, CPU is usually faster because GPU overhead dominates.

## Main API

```python
hubo_qaoa_feature_selection(
    X,
    y,
    feature_names=None,
    top_k=6,
    config=None,
    one_hot=True,
    device="cpu",
    aer_precision="single",
    p_layers=1,
    trials=8,
    maxiter=80,
    shots=4096,
    rho=0.2,
    seed=42,
    exact=True,
    max_exact_qubits=22,
    verbose=True,
)
```

The returned `FeatureSelectionResult` contains:

- `ranking`: pandas DataFrame with feature scores;
- `selected_features`: top-k feature names;
- `model`: the Ising/HUBO model;
- `qaoa_params`: optimized QAOA parameters;
- `counts`: sampled bitstrings;
- `best_sample_features`: best low-energy sampled feature subset;
- optional exact optimum fields when `exact=True`.

## Scaling Note

The default implementation uses exact enumeration of all `2^n` feature states
to evaluate the QAOA objective. This is convenient and transparent for small
feature counts, but it does not scale to large `n`.

For larger feature sets, first preselect features with a classical filter or
replace the objective evaluation with an estimator/sampling-based routine.
