"""HUBO/QAOA feature selection.

The public entry point is ``hubo_qaoa_feature_selection``. It accepts any
tabular feature matrix ``X`` and target vector ``y``, builds a HUBO-style
feature-selection objective from mutual information, converts it to an Ising
Hamiltonian, and estimates feature importance from QAOA low-energy samples.

The qubits correspond to features, not dataset rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mutual_info_score

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass(frozen=True)
class HuboConfig:
    """Hyperparameters for the HUBO feature-selection objective."""

    n_bins: int = 8
    w1: float = 1.0
    w2: float = 0.5
    w3: float = 0.15
    low_relevance_penalty: float = 0.35
    low_relevance_threshold: float = 0.25
    penalty_power: int = 3


@dataclass
class IsingModel:
    """Ising/HUBO model in Z variables."""

    feature_names: list[str]
    const: float
    h: np.ndarray
    j: dict[tuple[int, int], float]
    k: dict[tuple[int, int, int], float]
    relevance_mi_norm: np.ndarray
    binary_linear_coeff: np.ndarray
    hamiltonian: SparsePauliOp


@dataclass
class FeatureSelectionResult:
    """Result returned by ``hubo_qaoa_feature_selection``."""

    ranking: pd.DataFrame
    selected_features: list[str]
    model: IsingModel
    qaoa_params: np.ndarray
    qaoa_energy: float
    counts: Counter[str]
    best_sample_bitstring: str
    best_sample_energy: float
    best_sample_features: list[str]
    exact_best_bitstring: str | None = None
    exact_best_energy: float | None = None
    exact_best_features: list[str] | None = None


def hubo_qaoa_feature_selection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list[float],
    *,
    feature_names: list[str] | None = None,
    top_k: int = 6,
    config: HuboConfig | None = None,
    one_hot: bool = True,
    device: str = "cpu",
    aer_precision: str = "single",
    p_layers: int = 1,
    trials: int = 8,
    maxiter: int = 80,
    shots: int = 4096,
    rho: float = 0.2,
    seed: int = 42,
    exact: bool = True,
    max_exact_qubits: int = 22,
    verbose: bool = True,
) -> FeatureSelectionResult:
    """Select features with a HUBO objective and QAOA sampling.

    Parameters
    ----------
    X:
        Feature matrix. A pandas DataFrame preserves column names. If it
        contains categorical columns and ``one_hot=True``, they are one-hot
        encoded before the HUBO model is built.
    y:
        Target vector.
    feature_names:
        Optional names for ndarray columns. Ignored when ``X`` is a DataFrame.
    top_k:
        Number of top-ranked features to return in ``selected_features``.
    config:
        HUBO objective hyperparameters.
    one_hot:
        One-hot encode categorical DataFrame columns before feature selection.
    device:
        ``"cpu"`` uses Qiskit ``Statevector``. ``"gpu"`` tries Qiskit Aer GPU
        and falls back to CPU if unavailable.
    exact:
        If true, enumerate all ``2^n`` states and report the exact best state.
        QAOA optimization currently uses this exact energy vector, so keep this
        true unless you replace the objective for larger feature counts.

    Returns
    -------
    FeatureSelectionResult
        Contains the ranking table, selected features, Ising model, QAOA
        parameters, sampled counts, and optional exact optimum.
    """

    if config is None:
        config = HuboConfig()

    x_df, y_series = prepare_tabular_data(X, y, feature_names, one_hot)

    if not 0 < rho <= 1:
        raise ValueError("rho must be in the interval (0, 1].")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if verbose:
        print(f"rows = {len(x_df)}")
        print(f"n features = {x_df.shape[1]}")
        print("features =", x_df.columns.tolist())

    x_binned, y_binned = discretize_dataset(x_df, y_series, config.n_bins)
    model = build_hubo_ising_model(
        x_binned=x_binned,
        y_binned=y_binned,
        feature_names=x_df.columns.tolist(),
        config=config,
    )

    all_bitstrings = None
    all_energies = None
    exact_best_bitstring = None
    exact_best_energy = None
    exact_best_features = None

    if exact:
        all_bitstrings, all_energies = enumerate_state_energies(
            model,
            max_exact_qubits=max_exact_qubits,
        )
        exact_best_idx = int(np.argmin(all_energies))
        exact_best_bitstring = str(all_bitstrings[exact_best_idx])
        exact_best_energy = float(all_energies[exact_best_idx])
        exact_best_features = selected_features(model, exact_best_bitstring)

        if verbose:
            print("\nExact best:")
            print("energy =", exact_best_energy)
            print("bitstring =", exact_best_bitstring)
            print("features =", exact_best_features)

    if all_energies is None:
        raise NotImplementedError(
            "QAOA objective currently expects exact state energies. "
            "Use exact=True or provide a custom large-scale objective."
        )

    aer_backend = init_aer_backend(device, aer_precision, verbose=verbose)
    qaoa_result = optimize_qaoa(
        model=model,
        all_energies=all_energies,
        p_layers=p_layers,
        n_trials=trials,
        maxiter=maxiter,
        seed=seed,
        aer_backend=aer_backend,
        verbose=verbose,
    )

    if verbose:
        print("\nQAOA optimized energy:", float(qaoa_result.fun))
        print("params:", qaoa_result.x)

    qaoa_circuit = build_qaoa_circuit(model, qaoa_result.x, p_layers)
    probabilities = get_probabilities(qaoa_circuit, aer_backend)
    counts = sample_qaoa_distribution(
        probabilities=probabilities,
        n_qubits=len(model.feature_names),
        shots=shots,
        seed=seed,
    )

    best_sample = min(counts, key=lambda bitstring: bitstring_energy(model, bitstring))
    best_sample_energy = bitstring_energy(model, best_sample)
    best_sample_features = selected_features(model, best_sample)

    if verbose:
        print("\nBest sampled:")
        print("energy =", best_sample_energy)
        print("bitstring =", best_sample)
        print("features =", best_sample_features)

    ranking = build_feature_ranking(model, counts, rho)
    selected = ranking.head(top_k)["feature"].tolist()

    if verbose:
        print("\nFeature ranking:")
        print(ranking.to_string(index=False))
        print(f"\nTop-{top_k} QAOA/HUBO features:", selected)

    return FeatureSelectionResult(
        ranking=ranking,
        selected_features=selected,
        model=model,
        qaoa_params=qaoa_result.x,
        qaoa_energy=float(qaoa_result.fun),
        counts=counts,
        best_sample_bitstring=best_sample,
        best_sample_energy=best_sample_energy,
        best_sample_features=best_sample_features,
        exact_best_bitstring=exact_best_bitstring,
        exact_best_energy=exact_best_energy,
        exact_best_features=exact_best_features,
    )


def prepare_tabular_data(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list[float],
    feature_names: list[str] | None,
    one_hot: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert user data into a numeric DataFrame and aligned target Series."""

    if isinstance(X, pd.DataFrame):
        x_df = X.copy()
    else:
        x_array = np.asarray(X)

        if x_array.ndim != 2:
            raise ValueError("X must be a 2D array or pandas DataFrame.")

        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(x_array.shape[1])]

        if len(feature_names) != x_array.shape[1]:
            raise ValueError("feature_names length must match the number of X columns.")

        x_df = pd.DataFrame(x_array, columns=feature_names)

    y_series = pd.Series(y, name="target")

    if len(x_df) != len(y_series):
        raise ValueError("X and y must have the same number of rows.")

    if one_hot:
        categorical_cols = x_df.select_dtypes(include=["object", "category"]).columns
        x_df = pd.get_dummies(x_df, columns=list(categorical_cols), dummy_na=False)

    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x_df = x_df.fillna(0.0)
    x_df = x_df.astype(float)

    y_series = pd.to_numeric(y_series, errors="coerce")
    data = pd.concat([x_df, y_series], axis=1).dropna(subset=["target"])

    x_clean = data[x_df.columns].copy()
    y_clean = data["target"].copy()

    if x_clean.shape[1] == 0:
        raise ValueError("X must contain at least one feature after preprocessing.")

    return x_clean, y_clean


def quantile_bin(values: Iterable[float], n_bins: int) -> np.ndarray:
    """Discretize values for mutual information estimation."""

    series = pd.Series(values)

    if series.isna().all():
        return np.zeros(len(series), dtype=int)

    if series.nunique(dropna=True) <= 2:
        series = series.fillna(0)
        return pd.factorize(series, sort=True)[0].astype(int)

    series = series.fillna(series.median())

    if series.nunique() <= 1:
        return np.zeros(len(series), dtype=int)

    ranks = series.rank(method="first")
    return (
        pd.qcut(
            ranks,
            q=min(n_bins, series.nunique()),
            labels=False,
            duplicates="drop",
        )
        .astype(int)
        .to_numpy()
    )


def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    """Min-max normalize values to [0, 1]."""

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)

    return (values - values.min()) / span


def discretize_dataset(
    x_df: pd.DataFrame,
    y: pd.Series,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize all features and the target for mutual information."""

    x = x_df.to_numpy()
    x_binned = np.column_stack([quantile_bin(x[:, i], n_bins) for i in range(x.shape[1])])
    y_binned = quantile_bin(y.to_numpy(), n_bins)
    return x_binned, y_binned


def build_hubo_ising_model(
    x_binned: np.ndarray,
    y_binned: np.ndarray,
    feature_names: list[str],
    config: HuboConfig,
) -> IsingModel:
    """Build a HUBO objective and convert it to an Ising Hamiltonian.

    The binary objective is:

        E(x) = sum_i a_i x_i
             + sum_ij b_ij x_i x_j
             + sum_ijk c_ijk x_i x_j x_k

    where x_i = 1 means selected. Negative a_i rewards relevant features,
    while positive b_ij and c_ijk penalize redundant pairs/triples.

    The conversion uses x_i = (1 - Z_i) / 2.
    """

    n_features = len(feature_names)

    relevance_mi = np.array(
        [mutual_info_score(x_binned[:, i], y_binned) for i in range(n_features)]
    )
    relevance_mi_norm = minmax_normalize(relevance_mi)

    low_relevance_penalty = np.where(
        relevance_mi_norm < config.low_relevance_threshold,
        config.low_relevance_penalty
        * (
            (config.low_relevance_threshold - relevance_mi_norm)
            / config.low_relevance_threshold
        )
        ** config.penalty_power,
        0.0,
    )

    binary_linear_coeff = -config.w1 * relevance_mi_norm + low_relevance_penalty

    pairs = list(combinations(range(n_features), 2))
    pair_mi = np.array(
        [mutual_info_score(x_binned[:, i], x_binned[:, j]) for i, j in pairs]
    )
    pair_penalty = dict(zip(pairs, config.w2 * minmax_normalize(pair_mi)))

    def joint_code(i: int, j: int) -> np.ndarray:
        return x_binned[:, i] * config.n_bins + x_binned[:, j]

    triples = list(combinations(range(n_features), 3))
    triple_mi = []

    for i, j, k in triples:
        mij_k = mutual_info_score(joint_code(i, j), x_binned[:, k])
        mik_j = mutual_info_score(joint_code(i, k), x_binned[:, j])
        mjk_i = mutual_info_score(joint_code(j, k), x_binned[:, i])
        triple_mi.append((mij_k + mik_j + mjk_i) / 3)

    triple_penalty = dict(
        zip(triples, config.w3 * minmax_normalize(np.asarray(triple_mi)))
    )

    const = 0.0
    h = np.zeros(n_features)
    j_terms: dict[tuple[int, int], float] = defaultdict(float)
    k_terms: dict[tuple[int, int, int], float] = defaultdict(float)

    for i, coeff in enumerate(binary_linear_coeff):
        const += coeff / 2
        h[i] += -coeff / 2

    for (i, j), coeff in pair_penalty.items():
        const += coeff / 4
        h[i] += -coeff / 4
        h[j] += -coeff / 4
        j_terms[(i, j)] += coeff / 4

    for (i, j, k), coeff in triple_penalty.items():
        const += coeff / 8
        h[i] += -coeff / 8
        h[j] += -coeff / 8
        h[k] += -coeff / 8

        j_terms[(i, j)] += coeff / 8
        j_terms[(i, k)] += coeff / 8
        j_terms[(j, k)] += coeff / 8

        k_terms[(i, j, k)] += -coeff / 8

    j_terms = dict(j_terms)
    k_terms = dict(k_terms)
    hamiltonian = build_sparse_pauli_op(n_features, h, j_terms, k_terms)

    return IsingModel(
        feature_names=feature_names,
        const=float(const),
        h=h,
        j=j_terms,
        k=k_terms,
        relevance_mi_norm=relevance_mi_norm,
        binary_linear_coeff=binary_linear_coeff,
        hamiltonian=hamiltonian,
    )


def build_sparse_pauli_op(
    n_qubits: int,
    h: np.ndarray,
    j_terms: dict[tuple[int, int], float],
    k_terms: dict[tuple[int, int, int], float],
) -> SparsePauliOp:
    """Create a Qiskit SparsePauliOp for the Ising Hamiltonian."""

    terms: list[tuple[str, float]] = []

    for i, coeff in enumerate(h):
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - i] = "Z"
        terms.append(("".join(pauli), float(coeff)))

    for (i, j), coeff in j_terms.items():
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - i] = "Z"
        pauli[n_qubits - 1 - j] = "Z"
        terms.append(("".join(pauli), float(coeff)))

    for (i, j, k), coeff in k_terms.items():
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - i] = "Z"
        pauli[n_qubits - 1 - j] = "Z"
        pauli[n_qubits - 1 - k] = "Z"
        terms.append(("".join(pauli), float(coeff)))

    return SparsePauliOp.from_list(terms)


def bitstring_energy(model: IsingModel, bitstring: str) -> float:
    """Evaluate Ising energy for a Qiskit-order bitstring."""

    qbits = bitstring[::-1]
    z = np.array([1 if bit == "0" else -1 for bit in qbits])

    energy = model.const + float(np.dot(model.h, z))
    energy += sum(coeff * z[i] * z[j] for (i, j), coeff in model.j.items())
    energy += sum(coeff * z[i] * z[j] * z[k] for (i, j, k), coeff in model.k.items())
    return float(energy)


def selected_features(model: IsingModel, bitstring: str) -> list[str]:
    """Return selected feature names for a Qiskit-order bitstring."""

    qbits = bitstring[::-1]
    return [model.feature_names[i] for i, bit in enumerate(qbits) if bit == "1"]


def enumerate_state_energies(
    model: IsingModel,
    max_exact_qubits: int = 22,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all bitstrings and energies."""

    n_qubits = len(model.feature_names)

    if n_qubits > max_exact_qubits:
        raise ValueError(
            f"Exact enumeration needs 2^{n_qubits} states. "
            f"Increase max_exact_qubits if you really want this."
        )

    state_indices = np.arange(2**n_qubits, dtype=np.uint64)
    shifts = np.arange(n_qubits, dtype=np.uint64)
    x = ((state_indices[:, None] >> shifts) & 1).astype(np.int8)
    z = 1 - 2 * x

    energies = model.const + z @ model.h

    for (i, j), coeff in model.j.items():
        energies += coeff * z[:, i] * z[:, j]

    for (i, j, k), coeff in model.k.items():
        energies += coeff * z[:, i] * z[:, j] * z[:, k]

    bitstrings = np.array([format(int(i), f"0{n_qubits}b") for i in state_indices])
    return bitstrings, energies.astype(float)


def add_z_phase(qc: QuantumCircuit, qubits: list[int], theta: float) -> None:
    """Apply exp(-i theta Z...Z) using CNOT parity accumulation."""

    if len(qubits) == 1:
        qc.rz(2 * theta, qubits[0])
        return

    target = qubits[-1]

    for qubit in qubits[:-1]:
        qc.cx(qubit, target)

    qc.rz(2 * theta, target)

    for qubit in reversed(qubits[:-1]):
        qc.cx(qubit, target)


def build_qaoa_circuit(
    model: IsingModel,
    params: np.ndarray,
    p_layers: int,
) -> QuantumCircuit:
    """Build a p-layer QAOA circuit for the HUBO/Ising model."""

    n_qubits = len(model.feature_names)
    gammas = params[:p_layers]
    betas = params[p_layers:]

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for layer in range(p_layers):
        gamma = gammas[layer]

        for i, coeff in enumerate(model.h):
            add_z_phase(qc, [i], gamma * coeff)

        for (i, j), coeff in model.j.items():
            add_z_phase(qc, [i, j], gamma * coeff)

        for (i, j, k), coeff in model.k.items():
            add_z_phase(qc, [i, j, k], gamma * coeff)

        for qubit in range(n_qubits):
            qc.rx(2 * betas[layer], qubit)

    return qc


def init_aer_backend(device: str, precision: str, verbose: bool):
    """Initialize Qiskit Aer GPU backend, or return None for CPU Statevector."""

    if device.lower() != "gpu":
        if verbose:
            print("Simulator: CPU via qiskit.quantum_info.Statevector")
        return None

    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(
            method="statevector",
            device="GPU",
            precision=precision,
        )

        test_circuit = QuantumCircuit(1)
        test_circuit.h(0)
        test_circuit.save_statevector()
        backend.run(test_circuit, shots=1).result()

        if verbose:
            print("Simulator: GPU via qiskit-aer")

        return backend

    except Exception as exc:  # noqa: BLE001 - fallback should catch backend issues.
        if verbose:
            print("GPU simulator unavailable; falling back to CPU Statevector.")
            print(f"Reason: {exc!r}")
        return None


def get_probabilities(qc: QuantumCircuit, aer_backend=None) -> np.ndarray:
    """Return state probabilities from CPU Statevector or Aer GPU backend."""

    if aer_backend is None:
        return Statevector.from_instruction(qc).probabilities()

    qc_saved = qc.copy()
    qc_saved.save_statevector()

    result = aer_backend.run(qc_saved, shots=1).result()

    try:
        state = result.get_statevector(qc_saved)
    except Exception:  # noqa: BLE001 - Qiskit versions differ on this accessor.
        state = result.get_statevector(0)

    state = np.asarray(state, dtype=complex)
    probabilities = np.abs(state) ** 2
    probabilities = np.maximum(probabilities, 0)
    return probabilities / probabilities.sum()


def optimize_qaoa(
    model: IsingModel,
    all_energies: np.ndarray,
    p_layers: int,
    n_trials: int,
    maxiter: int,
    seed: int,
    aer_backend=None,
    verbose: bool = True,
):
    """Optimize QAOA parameters with COBYLA and random restarts."""

    rng = np.random.default_rng(seed)
    best_result = None

    def objective(params: np.ndarray) -> float:
        qc = build_qaoa_circuit(model, params, p_layers)
        probabilities = get_probabilities(qc, aer_backend)
        return float(probabilities @ all_energies)

    for trial in range(n_trials):
        x0 = np.r_[
            rng.uniform(0, 2 * np.pi, p_layers),
            rng.uniform(0, np.pi, p_layers),
        ]

        result = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": maxiter},
        )

        if verbose:
            print(f"trial {trial + 1:02d}/{n_trials}: energy = {result.fun:.6f}")

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result


def sample_qaoa_distribution(
    probabilities: np.ndarray,
    n_qubits: int,
    shots: int,
    seed: int,
) -> Counter[str]:
    """Sample bitstrings from a probability distribution."""

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(probabilities), size=shots, p=probabilities)
    return Counter(format(int(index), f"0{n_qubits}b") for index in sample_indices)


def feature_importance_from_low_energy_samples(
    model: IsingModel,
    counts: Counter[str],
    rho: float,
) -> np.ndarray:
    """Estimate feature importance from the lowest-energy sample fraction."""

    n_features = len(model.feature_names)
    n_top_samples = max(1, int(rho * sum(counts.values())))
    importance = np.zeros(n_features)
    used = 0

    sorted_counts = sorted(
        counts.items(),
        key=lambda item: bitstring_energy(model, item[0]),
    )

    for bitstring, count in sorted_counts:
        take = min(count, n_top_samples - used)
        if take <= 0:
            break

        qbits = bitstring[::-1]
        selected = np.array([1 if bit == "1" else 0 for bit in qbits])
        importance += take * selected
        used += take

    return importance / used


def build_feature_ranking(
    model: IsingModel,
    counts: Counter[str],
    rho: float,
) -> pd.DataFrame:
    """Create the final feature-ranking table."""

    importance = feature_importance_from_low_energy_samples(model, counts, rho)
    return (
        pd.DataFrame(
            {
                "feature": model.feature_names,
                "relevance_mi_norm": model.relevance_mi_norm,
                "binary_linear_coeff": model.binary_linear_coeff,
                "ising_h": model.h,
                "qaoa_importance": importance,
            }
        )
        .sort_values("qaoa_importance", ascending=False)
        .reset_index(drop=True)
    )
