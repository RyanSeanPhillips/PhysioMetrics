"""
ML Prediction Module

Handles loading ML models and running predictions on detected peaks.
Integrates with the 3-model cascade architecture:
- Model 1: Breath vs Noise detection
- Model 2: Sigh vs Normal breath classification
- Model 3: Eupnea vs Sniffing classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
import warnings

# Suppress sklearn feature name warnings (benign - predictions are still correct)
warnings.filterwarnings('ignore', message='X has feature names, but.*was fitted without feature names')


def load_model(filepath: Path) -> Tuple[any, Dict]:
    """
    Load trained model from disk.

    Args:
        filepath: Path to .pkl model file

    Returns:
        (model_object, metadata_dict)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Handle old format (just model object)
    if not isinstance(data, dict):
        return data, {}

    # Handle new format (model + metadata)
    return data.get('model'), data.get('metadata', {})


def extract_features_for_prediction(peak_metrics: List[Dict], feature_names: List[str], debug: bool = False) -> pd.DataFrame:
    """
    Extract features from peak metrics for ML prediction.

    Args:
        peak_metrics: List of metric dictionaries (one per peak)
        feature_names: List of feature names expected by the model
        debug: If True, print debug information about feature extraction

    Returns:
        DataFrame with features in correct order
    """
    if not peak_metrics:
        if debug:
            print(f"[ML Feature Extract] WARNING: Empty peak_metrics list")
        return pd.DataFrame()

    if not feature_names:
        if debug:
            print(f"[ML Feature Extract] WARNING: Empty feature_names list")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(peak_metrics)

    if debug:
        print(f"[ML Feature Extract] Got {len(peak_metrics)} peaks, {len(df.columns)} columns in metrics")
        print(f"[ML Feature Extract] Expected {len(feature_names)} features: {feature_names[:5]}...")

    # Track missing features
    missing_features = []

    # Ensure all required features exist, fill missing with 0
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0.0
            missing_features.append(feat)

    if missing_features and debug:
        print(f"[ML Feature Extract] WARNING: {len(missing_features)} missing features filled with 0: {missing_features[:5]}...")

    # Return features in the order expected by model
    X = df[feature_names].copy()  # .copy() to avoid SettingWithCopyWarning

    # Handle NaN values (critical for MLP which doesn't handle them natively)
    # Replace NaN with column median, or 0 if entire column is NaN
    # Suppress numpy warnings during median computation
    nan_cols = []
    all_nan_cols = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # Suppress "Mean of empty slice"
        warnings.simplefilter("ignore", category=FutureWarning)   # Suppress fillna downcasting warning

        if X.isna().any().any():
            for col in X.columns:
                if X[col].isna().any():
                    nan_cols.append(col)
                    # Count non-NaN values to detect all-NaN columns
                    non_nan_count = X[col].notna().sum()
                    if non_nan_count == 0:
                        all_nan_cols.append(col)
                        fill_val = 0.0
                    else:
                        median_val = X[col].median()
                        fill_val = median_val if not pd.isna(median_val) else 0.0
                    X[col] = X[col].fillna(fill_val)

    if debug and nan_cols:
        print(f"[ML Feature Extract] Filled NaN in {len(nan_cols)} columns: {nan_cols[:5]}...")
        if all_nan_cols:
            print(f"[ML Feature Extract] WARNING: {len(all_nan_cols)} columns were ALL NaN (filled with 0): {all_nan_cols}")

    return X


def predict_with_cascade(
    peak_metrics: List[Dict],
    models: Dict[str, Dict],
    algorithm: str = 'xgboost',
    debug: bool = False
) -> Dict[str, np.ndarray]:
    """
    Run 3-model cascade prediction on peaks.

    Args:
        peak_metrics: List of metric dictionaries (one per peak)
        models: Dict of loaded models {'model1_xgboost': {'model': ..., 'metadata': ...}, ...}
        algorithm: Which algorithm to use ('xgboost', 'rf', or 'mlp')
        debug: If True, print debug information about predictions

    Returns:
        Dictionary with prediction results:
            'breath_mask': Boolean array (True = breath, False = noise)
            'sigh_mask': Boolean array (True = sigh, False = normal) for breaths only
            'eupnea_mask': Boolean array (True = eupnea, False = sniffing) for non-sigh breaths
            'final_labels': Integer array (1 = breath, 0 = noise) for display
    """
    n_peaks = len(peak_metrics)

    if debug:
        print(f"[ML Cascade] predict_with_cascade called: {n_peaks} peaks, algorithm={algorithm}")
        print(f"[ML Cascade] Available models: {list(models.keys())}")

    # Initialize all as noise
    breath_mask = np.zeros(n_peaks, dtype=bool)
    sigh_mask = np.zeros(n_peaks, dtype=bool)
    eupnea_mask = np.zeros(n_peaks, dtype=bool)

    # Initialize eupnea/sniff class array (-1=unclassified, 0=eupnea, 1=sniffing)
    eupnea_sniff_class = np.full(n_peaks, -1, dtype=np.int8)

    # Initialize sigh class array (-1=unclassified/noise, 0=normal breath, 1=sigh)
    sigh_class = np.full(n_peaks, -1, dtype=np.int8)

    if n_peaks == 0:
        return {
            'breath_mask': breath_mask,
            'sigh_mask': sigh_mask,
            'sigh_class': sigh_class,
            'eupnea_mask': eupnea_mask,
            'eupnea_sniff_class': eupnea_sniff_class,
            'final_labels': breath_mask.astype(int)
        }

    # Model 1: Breath vs Noise
    # Find model key (may have accuracy suffix like "model1_xgboost_100%")
    model1_key_prefix = f'model1_{algorithm}'
    model1_keys = [k for k in models.keys() if k.startswith(model1_key_prefix)]

    if debug:
        print(f"[ML Cascade] Looking for Model 1 with prefix '{model1_key_prefix}'")
        print(f"[ML Cascade] Found model1_keys: {model1_keys}")

    if model1_keys:
        model1_key = model1_keys[0]  # Use first match
        model1 = models[model1_key]['model']
        model1_metadata = models[model1_key]['metadata']

        if debug:
            feature_names = model1_metadata.get('feature_names', [])
            print(f"[ML Cascade] Model 1 expects {len(feature_names)} features")

        # Extract features
        X1 = extract_features_for_prediction(peak_metrics, model1_metadata.get('feature_names', []), debug=debug)

        # Predict
        if len(X1) > 0:
            breath_predictions = model1.predict(X1)
            breath_mask = (breath_predictions == 1)
            if debug:
                print(f"[ML Cascade] Model 1 predicted: {np.sum(breath_mask)} breaths, {np.sum(~breath_mask)} noise")
        elif debug:
            print(f"[ML Cascade] WARNING: Model 1 got empty feature matrix (len(X1)=0)")
    elif debug:
        print(f"[ML Cascade] WARNING: No Model 1 found for algorithm '{algorithm}'")

    # Get indices of peaks classified as breaths
    breath_indices = np.where(breath_mask)[0]

    if len(breath_indices) == 0:
        return {
            'breath_mask': breath_mask,
            'sigh_mask': sigh_mask,
            'sigh_class': sigh_class,
            'eupnea_mask': eupnea_mask,
            'eupnea_sniff_class': eupnea_sniff_class,
            'final_labels': breath_mask.astype(int)
        }

    # Model 2: Sigh vs Normal (only on breaths)
    # Find model key (may have accuracy suffix)
    model2_key_prefix = f'model2_{algorithm}'
    model2_keys = [k for k in models.keys() if k.startswith(model2_key_prefix)]

    if model2_keys:
        model2_key = model2_keys[0]  # Use first match
        model2 = models[model2_key]['model']
        model2_metadata = models[model2_key]['metadata']

        # Extract features for breath peaks only
        breath_metrics = [peak_metrics[i] for i in breath_indices]
        X2 = extract_features_for_prediction(breath_metrics, model2_metadata.get('feature_names', []))

        # Predict
        if len(X2) > 0:
            sigh_predictions = model2.predict(X2)
            # Map back to full array
            for i, breath_idx in enumerate(breath_indices):
                sigh_mask[breath_idx] = (sigh_predictions[i] == 1)
                sigh_class[breath_idx] = sigh_predictions[i]  # 0 or 1

    # Get indices of non-sigh breaths for Model 3
    normal_breath_indices = breath_indices[~sigh_mask[breath_indices]]

    if len(normal_breath_indices) == 0:
        return {
            'breath_mask': breath_mask,
            'sigh_mask': sigh_mask,
            'sigh_class': sigh_class,
            'eupnea_mask': eupnea_mask,
            'eupnea_sniff_class': eupnea_sniff_class,
            'final_labels': breath_mask.astype(int)
        }

    # Model 3: Eupnea vs Sniffing (only on normal breaths)
    # Find model key (may have accuracy suffix)
    model3_key_prefix = f'model3_{algorithm}'
    model3_keys = [k for k in models.keys() if k.startswith(model3_key_prefix)]

    if model3_keys:
        model3_key = model3_keys[0]  # Use first match
        model3 = models[model3_key]['model']
        model3_metadata = models[model3_key]['metadata']

        # Extract features for normal breath peaks only
        normal_metrics = [peak_metrics[i] for i in normal_breath_indices]
        X3 = extract_features_for_prediction(normal_metrics, model3_metadata.get('feature_names', []))

        # Predict
        if len(X3) > 0:
            eupnea_predictions = model3.predict(X3)
            # Map back to full array (0 = eupnea, 1 = sniffing)
            for i, normal_idx in enumerate(normal_breath_indices):
                eupnea_mask[normal_idx] = (eupnea_predictions[i] == 0)
                eupnea_sniff_class[normal_idx] = eupnea_predictions[i]  # 0 or 1

    return {
        'breath_mask': breath_mask,
        'sigh_mask': sigh_mask,  # Keep for backward compatibility
        'sigh_class': sigh_class,  # NEW: -1/0/1 array
        'eupnea_mask': eupnea_mask,  # Keep for backward compatibility
        'eupnea_sniff_class': eupnea_sniff_class,  # NEW: -1/0/1 array
        'final_labels': breath_mask.astype(int)
    }


def get_model_summary(models: Dict[str, Dict]) -> str:
    """
    Generate a summary string of loaded models.

    Args:
        models: Dict of loaded models

    Returns:
        Formatted string with model information
    """
    if not models:
        return "No models loaded"

    lines = []
    for model_key, model_data in sorted(models.items()):
        metadata = model_data.get('metadata', {})
        model_num = metadata.get('model_number', '?')
        model_type = metadata.get('model_type', '?').upper()
        accuracy = metadata.get('test_accuracy', 0)
        cv_mean = metadata.get('cv_mean', 0)

        lines.append(f"Model {model_num} ({model_type}): {accuracy:.1%} accuracy (CV: {cv_mean:.1%})")

    return "\n".join(lines)
