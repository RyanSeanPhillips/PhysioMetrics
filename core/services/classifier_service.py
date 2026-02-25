"""
ClassifierService — pure Python ML classifier operations.

No Qt dependencies. Handles model loading, prediction, and label management
for all three model tiers:
  Model 1: breath vs noise
  Model 2: sigh detection
  Model 3: eupnea vs sniffing

Also wraps GMM clustering as an algorithm-agnostic classifier.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Model loading ────────────────────────────────────────────────


def load_models(models_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all ML models from a directory.

    Returns:
        Dict mapping model_key (e.g. 'model1_xgboost_95') to
        {'model': sklearn_model, 'metadata': dict, 'path': str}
    """
    import core.ml_prediction as ml_prediction

    loaded = {}
    for model_file in sorted(models_dir.glob("model*.pkl")):
        try:
            model, metadata = ml_prediction.load_model(model_file)
            loaded[model_file.stem] = {
                "model": model,
                "metadata": metadata,
                "path": str(model_file),
            }
        except Exception as e:
            print(f"[ClassifierService] Warning: Failed to load {model_file.name}: {e}")

    return loaded


def available_algorithms(loaded_models: Dict[str, Any]) -> Dict[str, List[str]]:
    """Report which algorithms are available for each model tier.

    Returns:
        {'model1': ['threshold', 'xgboost', ...], 'model2': [...], 'model3': [...]}
    """
    result = {
        "model1": ["threshold"],  # always available
        "model2": ["manual"],     # always available
        "model3": ["gmm", "all_eupnea", "none"],  # always available
    }
    for key in loaded_models:
        for tier in ("model1", "model2", "model3"):
            if key.startswith(f"{tier}_"):
                algo = key[len(tier) + 1:].split("_")[0]  # e.g. 'xgboost' from 'model1_xgboost_95'
                if algo not in result[tier]:
                    result[tier].append(algo)
    return result


# ── Model 1: Breath vs Noise ────────────────────────────────────


def predict_breath_vs_noise(
    all_peaks_by_sweep: Dict[int, Dict],
    loaded_models: Dict[str, Any],
    algorithm: str,
    get_peak_metrics_fn=None,
) -> int:
    """Run Model 1 predictions for all sweeps and store in all_peaks_data.

    Args:
        all_peaks_by_sweep: Sweep→peaks data dict (modified in-place)
        loaded_models: Loaded model dict
        algorithm: 'xgboost', 'rf', or 'mlp'
        get_peak_metrics_fn: Optional callable(sweep_idx, all_peaks_data) -> peak_metrics
            Used to lazily compute peak metrics if missing.

    Returns:
        Number of sweeps successfully processed.
    """
    import core.ml_prediction as ml_prediction

    model_key_prefix = f"model1_{algorithm}"
    matching = [k for k in loaded_models if k.startswith(model_key_prefix)]
    if not matching:
        return 0

    labels_key = f"labels_{algorithm}_ro"
    computed = 0

    for s, all_peaks_data in all_peaks_by_sweep.items():
        # Skip if already computed
        if labels_key in all_peaks_data and all_peaks_data[labels_key] is not None:
            continue

        # Ensure peak metrics exist
        peak_metrics = all_peaks_data.get("peak_metrics")
        if peak_metrics is None and get_peak_metrics_fn:
            peak_metrics = get_peak_metrics_fn(s, all_peaks_data)
            all_peaks_data["peak_metrics"] = peak_metrics
        if peak_metrics is None:
            continue

        try:
            predictions = ml_prediction.predict_with_cascade(
                peak_metrics=peak_metrics,
                models=loaded_models,
                algorithm=algorithm,
                debug=(s == 0),
            )
            all_peaks_data[labels_key] = predictions["final_labels"]

            # Store eupnea/sniff and sigh predictions if available
            if "eupnea_sniff_class" in predictions:
                all_peaks_data[f"eupnea_sniff_{algorithm}_ro"] = predictions["eupnea_sniff_class"]
            if "sigh_class" in predictions:
                all_peaks_data[f"sigh_{algorithm}_ro"] = predictions["sigh_class"]

            computed += 1
        except Exception as e:
            print(f"[ClassifierService] Model1 {algorithm} sweep {s} failed: {e}")
            all_peaks_data[labels_key] = None

    return computed


def apply_classifier_labels(
    all_peaks_by_sweep: Dict[int, Dict],
    algorithm: str,
) -> bool:
    """Copy the selected classifier's read-only labels to the active 'labels' array.

    Returns True if any fallback was used.
    """
    fallback_used = False
    labels_key = f"labels_{algorithm}_ro"

    for s, data in all_peaks_by_sweep.items():
        if labels_key in data and data[labels_key] is not None:
            data["labels"] = data[labels_key].copy()
            data["label_source"] = np.array(["auto"] * len(data["labels"]))
        else:
            fallback_used = True
            if "labels_threshold_ro" in data and data["labels_threshold_ro"] is not None:
                data["labels"] = data["labels_threshold_ro"].copy()
                data["label_source"] = np.array(["auto"] * len(data["labels"]))

    return fallback_used


# ── Model 3: Eupnea vs Sniffing ─────────────────────────────────


def predict_eupnea_sniff(
    all_peaks_by_sweep: Dict[int, Dict],
    loaded_models: Dict[str, Any],
    algorithm: str,
    active_classifier: str = "threshold",
    get_peak_metrics_fn=None,
) -> int:
    """Run Model 3 predictions for all sweeps.

    Returns number of sweeps processed.
    """
    import core.ml_prediction as ml_prediction

    model_key_prefix = f"model3_{algorithm}"
    matching = [k for k in loaded_models if k.startswith(model_key_prefix)]
    if not matching:
        return 0

    model_key = matching[0]
    model = loaded_models[model_key]["model"]
    metadata = loaded_models[model_key]["metadata"]
    feature_names = metadata.get("feature_names", [])

    eupnea_sniff_key = f"eupnea_sniff_{algorithm}_ro"
    computed = 0

    for s, data in all_peaks_by_sweep.items():
        if eupnea_sniff_key in data and data[eupnea_sniff_key] is not None:
            continue

        labels = data.get("labels")
        if labels is None:
            labels = data.get(f"labels_{active_classifier}_ro")
        if labels is None:
            continue

        breath_mask = labels == 1
        breath_indices = np.where(breath_mask)[0]

        if len(breath_indices) == 0:
            data[eupnea_sniff_key] = np.full(len(labels), -1, dtype=np.int8)
            continue

        peak_metrics = data.get("peak_metrics")
        if peak_metrics is None and get_peak_metrics_fn:
            peak_metrics = get_peak_metrics_fn(s, data)
            data["peak_metrics"] = peak_metrics
        if peak_metrics is None:
            continue

        breath_metrics = [peak_metrics[i] for i in breath_indices]

        try:
            X = ml_prediction.extract_features_for_prediction(
                breath_metrics, feature_names, debug=(s == 0)
            )
            if len(X) > 0:
                preds = model.predict(X)
                eupnea_sniff_class = np.full(len(labels), -1, dtype=np.int8)
                for i, idx in enumerate(breath_indices):
                    eupnea_sniff_class[idx] = preds[i]
                data[eupnea_sniff_key] = eupnea_sniff_class
                computed += 1
        except Exception as e:
            print(f"[ClassifierService] Model3 {algorithm} sweep {s} failed: {e}")
            data[eupnea_sniff_key] = None

    return computed


def apply_eupnea_sniff_labels(
    all_peaks_by_sweep: Dict[int, Dict],
    classifier: str,
):
    """Copy selected eupnea/sniff classifier results to active breath_type_class."""
    source_key = "gmm_class_ro" if classifier == "gmm" else f"eupnea_sniff_{classifier}_ro"

    for s, data in all_peaks_by_sweep.items():
        if source_key in data and data[source_key] is not None:
            data["breath_type_class"] = data[source_key].copy()
            n = len(data.get("indices", []))
            data["eupnea_sniff_source"] = np.array([classifier] * n)


def set_all_eupnea_sniff(all_peaks_by_sweep: Dict[int, Dict], class_value: int, source: str):
    """Set all breaths to a specific class (0=eupnea, 1=sniffing)."""
    for data in all_peaks_by_sweep.values():
        indices = data.get("indices")
        if indices is not None:
            n = len(indices)
            data["breath_type_class"] = np.full(n, class_value, dtype=int)
            data["eupnea_sniff_source"] = np.array([source] * n)


def clear_eupnea_sniff(all_peaks_by_sweep: Dict[int, Dict]):
    """Clear all eupnea/sniff labels."""
    for data in all_peaks_by_sweep.values():
        data.pop("breath_type_class", None)
        data.pop("eupnea_sniff_source", None)


# ── Model 2: Sigh Detection ─────────────────────────────────────


def predict_sighs(
    all_peaks_by_sweep: Dict[int, Dict],
    loaded_models: Dict[str, Any],
    algorithm: str,
    active_classifier: str = "threshold",
    get_peak_metrics_fn=None,
) -> int:
    """Run Model 2 sigh predictions for all sweeps. Returns sweeps processed."""
    import core.ml_prediction as ml_prediction

    model_key_prefix = f"model2_{algorithm}"
    matching = [k for k in loaded_models if k.startswith(model_key_prefix)]
    if not matching:
        return 0

    model_key = matching[0]
    model = loaded_models[model_key]["model"]
    metadata = loaded_models[model_key]["metadata"]
    feature_names = metadata.get("feature_names", [])

    sigh_key = f"sigh_{algorithm}_ro"
    computed = 0

    for s, data in all_peaks_by_sweep.items():
        if sigh_key in data and data[sigh_key] is not None:
            continue

        labels = data.get("labels")
        if labels is None:
            labels = data.get(f"labels_{active_classifier}_ro")
        if labels is None:
            continue

        breath_indices = np.where(labels == 1)[0]
        if len(breath_indices) == 0:
            data[sigh_key] = np.full(len(labels), -1, dtype=np.int8)
            continue

        peak_metrics = data.get("peak_metrics")
        if peak_metrics is None and get_peak_metrics_fn:
            peak_metrics = get_peak_metrics_fn(s, data)
            data["peak_metrics"] = peak_metrics
        if peak_metrics is None:
            continue

        breath_metrics = [peak_metrics[i] for i in breath_indices]

        try:
            X = ml_prediction.extract_features_for_prediction(
                breath_metrics, feature_names, debug=(s == 0)
            )
            if len(X) > 0:
                preds = model.predict(X)
                sigh_class = np.full(len(labels), -1, dtype=np.int8)
                for i, idx in enumerate(breath_indices):
                    sigh_class[idx] = preds[i]
                data[sigh_key] = sigh_class
                computed += 1
        except Exception as e:
            print(f"[ClassifierService] Model2 {algorithm} sweep {s} failed: {e}")
            data[sigh_key] = None

    return computed


def apply_sigh_labels(
    all_peaks_by_sweep: Dict[int, Dict],
    sigh_by_sweep: Dict[int, Any],
    classifier: str,
):
    """Copy selected sigh classifier results to active sigh_class."""
    if classifier == "manual":
        source_key = "sigh_manual_ro"
    else:
        source_key = f"sigh_{classifier}_ro"

    for s, data in all_peaks_by_sweep.items():
        if source_key in data and data[source_key] is not None:
            data["sigh_class"] = data[source_key].copy()
            n = len(data.get("indices", []))
            data["sigh_source"] = np.array([classifier] * n)
            sigh_mask = data["sigh_class"] == 1
            sigh_by_sweep[s] = data["indices"][sigh_mask].tolist()


def clear_sighs(all_peaks_by_sweep: Dict[int, Dict], sigh_by_sweep: Dict[int, Any]):
    """Clear all sigh labels."""
    for data in all_peaks_by_sweep.values():
        data.pop("sigh_class", None)
        data.pop("sigh_source", None)
    sigh_by_sweep.clear()


# ── GMM clustering ───────────────────────────────────────────────


def run_gmm(
    feature_matrix: np.ndarray,
    feature_keys: List[str],
    n_clusters: int = 2,
) -> Optional[Dict[str, Any]]:
    """Run GMM clustering on breath features.

    Args:
        feature_matrix: (n_breaths, n_features) array
        feature_keys: Feature names (must include 'if' or 'ti' for cluster ID)
        n_clusters: Number of clusters

    Returns:
        Dict with cluster_labels, probabilities, sniffing_cluster_id,
        silhouette_score, or None if clustering fails.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    if len(feature_matrix) < n_clusters:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full")
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    sil = silhouette_score(X, labels) if n_clusters > 1 else -1

    # Identify sniffing cluster (highest IF, lowest Ti)
    sniff_id = _identify_sniffing_cluster(feature_matrix, labels, feature_keys)

    return {
        "cluster_labels": labels,
        "probabilities": probs,
        "sniffing_cluster_id": sniff_id,
        "silhouette_score": sil,
        "scaler": scaler,
        "gmm_model": gmm,
    }


def _identify_sniffing_cluster(
    features: np.ndarray,
    labels: np.ndarray,
    feature_keys: List[str],
) -> Optional[int]:
    """Identify which cluster is sniffing based on IF/Ti."""
    if_idx = feature_keys.index("if") if "if" in feature_keys else None
    ti_idx = feature_keys.index("ti") if "ti" in feature_keys else None

    if if_idx is None and ti_idx is None:
        return None

    unique = np.unique(labels)
    n = len(unique)
    scores = {}

    for cid in unique:
        mask = labels == cid
        score = 0
        if if_idx is not None:
            means = [np.mean(features[labels == c, if_idx]) for c in unique]
            rank = sorted(means).index(np.mean(features[mask, if_idx]))
            score += rank / (n - 1) if n > 1 else 0
        if ti_idx is not None:
            means = [np.mean(features[labels == c, ti_idx]) for c in unique]
            rank = sorted(means, reverse=True).index(np.mean(features[mask, ti_idx]))
            score += rank / (n - 1) if n > 1 else 0
        scores[cid] = score

    return max(scores, key=scores.get)
