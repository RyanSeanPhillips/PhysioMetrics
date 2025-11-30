"""
ML Model Training Module

Provides functions to train breath classification models:
- Model 1: Breath vs Noise (binary classification)
- Model 2: Breath Type (multiclass: eupnea/sniffing/sigh)

Supported algorithms:
- Random Forest
- XGBoost
- MLP (Multi-Layer Perceptron)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# For plot generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TrainingResult:
    """Results from training a model."""
    model: Any  # Trained model object
    model_type: str  # 'rf', 'xgboost', or 'mlp'
    model_name: str  # Display name (e.g., "Model 1: Breath vs Noise")

    # Accuracy metrics
    train_accuracy: float
    test_accuracy: float
    cv_mean: float
    cv_std: float

    # Per-class metrics
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]

    # Confusion matrix
    confusion_matrix: np.ndarray
    class_labels: List[str]

    # Feature importance
    feature_importance: pd.DataFrame  # Columns: 'feature', 'importance'

    # Training info
    n_train: int
    n_test: int
    n_features: int
    feature_names: List[str]

    # Plots (as PNG bytes)
    feature_importance_plot: Optional[bytes] = None
    confusion_matrix_plot: Optional[bytes] = None
    learning_curve_plot: Optional[bytes] = None

    # Optional: comparison to baseline
    baseline_accuracy: Optional[float] = None
    accuracy_improvement: Optional[float] = None
    error_reduction_pct: Optional[float] = None
    baseline_recall: Optional[Dict[str, float]] = None  # Per-class recall for baseline

    # Class distribution
    class_distribution: Optional[Dict[str, int]] = None  # {class_name: count}

    # Training time
    training_time_seconds: Optional[float] = None

    # Learning curve analysis
    is_converged: Optional[bool] = None  # Whether learning curve has plateaued
    needs_more_data: Optional[bool] = None  # Whether more data would likely help


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'rf',
    model_name: str = "Model",
    test_size: float = 0.2,
    random_state: int = 42,
    baseline_accuracy: Optional[float] = None,
    baseline_recall: Optional[Dict[str, float]] = None,
    generate_plots: bool = True,
    **model_kwargs
) -> TrainingResult:
    """
    Train a classification model and return structured results.

    Args:
        X: Feature DataFrame
        y: Target labels (Series or array)
        model_type: 'rf' (Random Forest), 'xgboost', or 'mlp'
        model_name: Display name for the model
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        baseline_accuracy: Optional baseline overall accuracy to compare against
        baseline_recall: Optional per-class baseline recall to compare against
        generate_plots: Whether to generate visualization plots
        **model_kwargs: Additional arguments passed to model constructor

    Returns:
        TrainingResult with model, metrics, and optional plots
    """
    import time

    # Calculate class distribution
    class_dist = y.value_counts().to_dict()
    class_dist = {str(k): int(v) for k, v in class_dist.items()}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model based on type (with timing)
    start_time = time.time()
    if model_type == 'rf':
        model = _train_random_forest(X_train, y_train, **model_kwargs)
    elif model_type == 'xgboost':
        model = _train_xgboost(X_train, y_train, **model_kwargs)
    elif model_type == 'mlp':
        model = _train_mlp(X_train, y_train, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    training_time = time.time() - start_time

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Cross-validation on full dataset
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, zero_division=0
    )

    class_labels = sorted(y.unique())
    precision_dict = {str(label): p for label, p in zip(class_labels, precision)}
    recall_dict = {str(label): r for label, r in zip(class_labels, recall)}
    f1_dict = {str(label): f for label, f in zip(class_labels, f1)}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=class_labels)

    # Feature importance
    feature_importance_df = _get_feature_importance(model, X.columns, model_type)

    # Baseline comparison
    accuracy_improvement = None
    error_reduction_pct = None
    if baseline_accuracy is not None:
        accuracy_improvement = test_accuracy - baseline_accuracy
        if baseline_accuracy < 1.0:
            error_reduction_pct = (accuracy_improvement / (1 - baseline_accuracy)) * 100

    # Generate plots and analyze learning curve
    feature_importance_plot_bytes = None
    confusion_matrix_plot_bytes = None
    learning_curve_plot_bytes = None
    is_converged = None
    needs_more_data = None

    if generate_plots:
        feature_importance_plot_bytes = _plot_feature_importance(
            feature_importance_df, model_name, model_type
        )
        confusion_matrix_plot_bytes = _plot_confusion_matrix(
            cm, class_labels, model_name, model_type
        )
        learning_curve_plot_bytes, is_converged, needs_more_data = _plot_learning_curve(
            model, X, y, model_name, model_type
        )

    # Create result
    result = TrainingResult(
        model=model,
        model_type=model_type,
        model_name=model_name,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        cv_mean=cv_mean,
        cv_std=cv_std,
        precision=precision_dict,
        recall=recall_dict,
        f1_score=f1_dict,
        confusion_matrix=cm,
        class_labels=[str(l) for l in class_labels],
        feature_importance=feature_importance_df,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=X.shape[1],
        feature_names=list(X.columns),
        feature_importance_plot=feature_importance_plot_bytes,
        confusion_matrix_plot=confusion_matrix_plot_bytes,
        learning_curve_plot=learning_curve_plot_bytes,
        baseline_accuracy=baseline_accuracy,
        accuracy_improvement=accuracy_improvement,
        error_reduction_pct=error_reduction_pct,
        baseline_recall=baseline_recall,
        class_distribution=class_dist,
        training_time_seconds=training_time,
        is_converged=is_converged,
        needs_more_data=needs_more_data
    )

    return result


def _train_random_forest(X_train, y_train, **kwargs):
    """Train Random Forest with default hyperparameters."""
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(kwargs)

    rf = RandomForestClassifier(**default_params)
    rf.fit(X_train, y_train)
    return rf


def _train_xgboost(X_train, y_train, **kwargs):
    """Train XGBoost with default hyperparameters."""
    # Encode labels if string
    if y_train.dtype == 'object':
        label_map = {label: idx for idx, label in enumerate(sorted(y_train.unique()))}
        y_train_encoded = y_train.map(label_map)
        # Store label map for later use
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss' if len(label_map) > 2 else 'logloss'
        }
    else:
        y_train_encoded = y_train
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }

    default_params.update(kwargs)

    xgb_model = xgb.XGBClassifier(**default_params)
    xgb_model.fit(X_train, y_train_encoded)

    # Store original labels for prediction
    xgb_model._label_map = label_map if y_train.dtype == 'object' else None

    return xgb_model


def _train_mlp(X_train, y_train, **kwargs):
    """Train MLP with default hyperparameters."""
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    default_params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1
    }
    default_params.update(kwargs)

    mlp = MLPClassifier(**default_params)
    mlp.fit(X_train_scaled, y_train)

    # Store scaler for prediction
    mlp._scaler = scaler

    return mlp


def _get_feature_importance(model, feature_names, model_type):
    """Extract feature importance from model."""
    if model_type == 'rf':
        importance = model.feature_importances_
    elif model_type == 'xgboost':
        importance = model.feature_importances_
    elif model_type == 'mlp':
        # MLP doesn't have direct feature importance, use absolute mean weights from first layer
        importance = np.abs(model.coefs_[0]).mean(axis=1)
        # Normalize to sum to 1
        importance = importance / importance.sum()
    else:
        importance = np.zeros(len(feature_names))

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return df


def _plot_feature_importance(feature_importance_df, model_name, model_type):
    """Generate feature importance plot and return as PNG bytes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dark theme colors
    bg_color = '#1e1e1e'
    text_color = '#d4d4d4'
    grid_color = '#3e3e42'

    # Set dark theme
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Plot top 15 features
    top_features = feature_importance_df.head(15)

    sns.barplot(
        data=top_features,
        y='feature',
        x='importance',
        hue='feature',
        ax=ax,
        palette='viridis',
        legend=False
    )

    ax.set_xlabel('Importance', fontsize=12, color=text_color)
    ax.set_ylabel('Feature', fontsize=12, color=text_color)
    ax.set_title(f'Top 15 Feature Importance - {model_name} ({model_type.upper()})',
                 fontsize=14, fontweight='bold', color=text_color)
    ax.grid(axis='x', alpha=0.3, color=grid_color)

    # Set tick colors
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=bg_color)
    buf.seek(0)
    plot_bytes = buf.read()
    buf.close()
    plt.close(fig)

    return plot_bytes


def _plot_confusion_matrix(cm, class_labels, model_name, model_type):
    """Generate confusion matrix heatmap and return as PNG bytes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Dark theme colors
    bg_color = '#1e1e1e'
    text_color = '#d4d4d4'

    # Set dark theme
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Create heatmap with dark-friendly colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar_kws={'label': 'Count'},
        annot_kws={'color': 'white'}  # Make annotations white for better visibility
    )

    ax.set_xlabel('Predicted Label', fontsize=12, color=text_color)
    ax.set_ylabel('True Label', fontsize=12, color=text_color)
    ax.set_title(f'Confusion Matrix - {model_name} ({model_type.upper()})',
                 fontsize=14, fontweight='bold', color=text_color)

    # Set tick colors
    ax.tick_params(colors=text_color)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
    cbar.set_label('Count', color=text_color)

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=bg_color)
    buf.seek(0)
    plot_bytes = buf.read()
    buf.close()
    plt.close(fig)

    return plot_bytes


def _plot_learning_curve(model, X, y, model_name, model_type):
    """Generate learning curve plot with per-class recall and return as PNG bytes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dark theme colors
    bg_color = '#1e1e1e'
    text_color = '#d4d4d4'
    grid_color = '#3e3e42'

    # Set dark theme
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Compute learning curve
    # Use 5 training sizes from 10% to 95% of data (can't use 100% due to sklearn constraint)
    train_sizes = np.linspace(0.1, 0.95, 5)

    # Initialize convergence analysis variables
    is_converged = None
    needs_more_data = None

    # Define class labels and colors based on model name
    if "Breath vs Noise" in model_name or "Model 1" in model_name:
        class_labels = {0: "Noise", 1: "Breath"}
        class_colors = {0: '#ce9178', 1: '#4ec9b0'}  # orange for noise, teal for breath
    elif "Sigh" in model_name or "Model 2" in model_name:
        class_labels = {0: "Normal", 1: "Sigh"}
        class_colors = {0: '#4ec9b0', 1: '#c586c0'}  # teal for normal, purple for sigh
    elif "Eupnea" in model_name or "Model 3" in model_name:
        class_labels = {0: "Eupnea", 1: "Sniffing"}
        class_colors = {0: '#569cd6', 1: '#dcdcaa'}  # blue for eupnea, yellow for sniffing
    else:
        # Fallback to generic labels
        unique_classes = np.unique(y)
        class_labels = {cls: f"Class {cls}" for cls in unique_classes}
        colors = ['#4ec9b0', '#ce9178', '#c586c0', '#569cd6', '#dcdcaa']
        class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    try:
        # Get unique classes from data
        unique_classes = sorted(np.unique(y))

        # Calculate learning curves for each class using recall scoring
        # We'll compute recall for each class separately
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import make_scorer, recall_score

        # Store results for each class
        class_recall_results = {cls: {'train': [], 'val': [], 'train_std': [], 'val_std': []}
                               for cls in unique_classes}
        train_sizes_abs = []

        # Compute learning curve at each training size
        for train_size in train_sizes:
            n_samples = int(len(X) * train_size)
            if n_samples < 20:  # Skip if too few samples
                continue

            # Check if this training size gives enough samples per class for CV
            # Need at least 10 samples per class (2 per fold with 5-fold CV)
            min_samples_needed = 10
            skip_this_size = False

            for cls in unique_classes:
                n_class_samples = int((y == cls).sum() * train_size)
                if n_class_samples < min_samples_needed:
                    print(f"Skipping train_size={train_size:.1%} (n={n_samples}): "
                          f"Class {cls} would have only {n_class_samples} samples (need {min_samples_needed})")
                    skip_this_size = True
                    break

            if skip_this_size:
                continue

            train_sizes_abs.append(n_samples)

            # Subsample data
            from sklearn.model_selection import train_test_split
            try:
                X_sub, _, y_sub, _ = train_test_split(
                    X, y, train_size=train_size, random_state=42, stratify=y
                )
            except ValueError as e:
                # Stratification might fail if class too small
                print(f"Skipping train_size={train_size:.1%}: Stratification failed - {e}")
                train_sizes_abs.pop()  # Remove the size we just added
                continue

            # For each class, compute recall using cross-validation
            for cls in unique_classes:
                # Create binary scorer for this class (one-vs-rest)
                scorer = make_scorer(recall_score, pos_label=cls, average=None, zero_division=0)

                # Perform cross-validation
                from sklearn.base import clone
                train_recalls = []
                val_recalls = []

                from sklearn.model_selection import KFold
                # Determine safe number of folds based on smallest class
                min_class_count = min((y_sub == c).sum() for c in unique_classes)
                n_splits = min(5, n_samples // 10, min_class_count // 2)
                n_splits = max(2, n_splits)  # At least 2 folds

                try:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                except ValueError:
                    # If still too few samples, skip this class for this size
                    print(f"Skipping class {cls} at train_size={train_size:.1%}: too few samples for CV")
                    continue

                for train_idx, val_idx in kf.split(X_sub):
                    X_train_fold = X_sub.iloc[train_idx] if hasattr(X_sub, 'iloc') else X_sub[train_idx]
                    X_val_fold = X_sub.iloc[val_idx] if hasattr(X_sub, 'iloc') else X_sub[val_idx]
                    y_train_fold = y_sub.iloc[train_idx] if hasattr(y_sub, 'iloc') else y_sub[train_idx]
                    y_val_fold = y_sub.iloc[val_idx] if hasattr(y_sub, 'iloc') else y_sub[val_idx]

                    # Clone and train model
                    model_clone = clone(model)
                    model_clone.fit(X_train_fold, y_train_fold)

                    # Predict
                    y_train_pred = model_clone.predict(X_train_fold)
                    y_val_pred = model_clone.predict(X_val_fold)

                    # Calculate recall for this class
                    train_recall = recall_score(y_train_fold, y_train_pred, pos_label=cls, average='binary', zero_division=0)
                    val_recall = recall_score(y_val_fold, y_val_pred, pos_label=cls, average='binary', zero_division=0)

                    train_recalls.append(train_recall)
                    val_recalls.append(val_recall)

                # Store mean and std for this class at this training size
                class_recall_results[cls]['train'].append(np.mean(train_recalls))
                class_recall_results[cls]['val'].append(np.mean(val_recalls))
                class_recall_results[cls]['train_std'].append(np.std(train_recalls))
                class_recall_results[cls]['val_std'].append(np.std(val_recalls))

        train_sizes_abs = np.array(train_sizes_abs)

        # Plot per-class recall curves
        for cls in unique_classes:
            label = class_labels.get(cls, f"Class {cls}")
            color = class_colors.get(cls, '#4ec9b0')

            train_recall = np.array(class_recall_results[cls]['train'])
            val_recall = np.array(class_recall_results[cls]['val'])
            train_std = np.array(class_recall_results[cls]['train_std'])
            val_std = np.array(class_recall_results[cls]['val_std'])

            # Plot validation recall (solid line - most important)
            ax.plot(train_sizes_abs, val_recall, 'o-', color=color,
                    label=f'{label} (Val)', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs,
                            val_recall - val_std,
                            val_recall + val_std,
                            alpha=0.2, color=color)

            # Plot training recall (dashed line - for reference)
            ax.plot(train_sizes_abs, train_recall, 'o--', color=color,
                    label=f'{label} (Train)', linewidth=1.5, markersize=4, alpha=0.6)

        ax.set_xlabel('Training Set Size', fontsize=12, color=text_color)
        ax.set_ylabel('Recall Score', fontsize=12, color=text_color)
        ax.set_title(f'Per-Class Recall Learning Curves - {model_name} ({model_type.upper()})',
                     fontsize=14, fontweight='bold', color=text_color)
        ax.legend(loc='best', facecolor=bg_color, edgecolor=grid_color,
                 labelcolor=text_color, fontsize=9)
        ax.grid(alpha=0.3, color=grid_color)
        ax.set_ylim([0, 1.05])  # Recall is between 0 and 1

        # Set tick colors
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)

        plt.tight_layout()

        # Analyze convergence based on validation recalls
        # Check the worst-performing class (minority class typically)
        try:
            min_val_recalls = []
            for cls in unique_classes:
                val_recall = np.array(class_recall_results[cls]['val'])
                if len(val_recall) >= 2:
                    min_val_recalls.append(val_recall[-1])

            if len(min_val_recalls) >= 2:
                # Use the minimum (worst) class recall for convergence analysis
                worst_recall = min(min_val_recalls)

                # Check if worst class has converged
                worst_class = unique_classes[np.argmin([class_recall_results[cls]['val'][-1]
                                                       for cls in unique_classes])]
                val_recall_worst = np.array(class_recall_results[worst_class]['val'])

                if len(val_recall_worst) >= 2:
                    last_two_diff = abs(val_recall_worst[-1] - val_recall_worst[-2])
                    is_converged = last_two_diff < 0.01

                    # Check if more data would help
                    if len(val_recall_worst) >= 3:
                        is_improving = val_recall_worst[-1] > val_recall_worst[-3]
                        train_recall_worst = np.array(class_recall_results[worst_class]['train'])
                        train_val_gap = train_recall_worst[-1] - val_recall_worst[-1]
                        needs_more_data = is_improving and train_val_gap > 0.05
        except:
            pass

    except Exception as e:
        # If learning curve fails, show error message
        import traceback
        error_msg = f'Learning curve failed:\n{str(e)}'
        full_traceback = traceback.format_exc()

        # Print full error to console for debugging
        print(f"\n{'='*60}\nLearning Curve Error Details:\n{'='*60}")
        print(f"Model: {model_name} ({model_type})")
        print(f"Error: {str(e)}")
        print(f"\nFull Traceback:\n{full_traceback}")
        print('='*60 + '\n')

        # Display simplified error on plot with larger font
        ax.text(0.5, 0.5, error_msg,
                ha='center', va='center', color='#ce9178',
                transform=ax.transAxes, fontsize=12, wrap=True,
                bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=grid_color, alpha=0.8))
        ax.set_xlabel('Training Set Size', fontsize=12, color=text_color)
        ax.set_ylabel('Recall Score', fontsize=12, color=text_color)
        ax.set_title(f'Per-Class Recall Learning Curves - {model_name} ({model_type.upper()})',
                     fontsize=14, fontweight='bold', color=text_color)

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=bg_color)
    buf.seek(0)
    plot_bytes = buf.read()
    buf.close()
    plt.close(fig)

    return plot_bytes, is_converged, needs_more_data


def load_training_data_from_directory(
    data_dir: Path,
    deduplicate: bool = False,
    model_number: int = 1,
    file_paths: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, str, Optional[float], Optional[Dict[str, float]]]:
    """
    Load and combine ML training data from directory of .npz files.

    Args:
        data_dir: Path to directory containing .npz training files
        deduplicate: If True, remove duplicate files from same source (keep most recent)
        model_number: 1 for breath vs noise, 2 for sigh vs normal, 3 for eupnea vs sniffing
        file_paths: Optional list of specific file paths to use (for filtering).
                   If provided, only these files will be loaded (data_dir is ignored).

    Returns:
        (X, y, dataset_type, baseline_accuracy, baseline_recall): Features, labels, dataset type,
            optional baseline threshold accuracy, and optional per-class baseline recall
    """
    # Use provided file paths if given, otherwise glob the directory
    if file_paths:
        npz_files = [Path(p) for p in file_paths]
        print(f"Using {len(npz_files)} filtered training files")
    else:
        npz_files = list(Path(data_dir).glob("*.npz"))

    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")

    print(f"Found {len(npz_files)} training files")

    # Deduplicate if requested
    if deduplicate:
        # Group files by source_file metadata
        from collections import defaultdict
        import os

        source_to_files = defaultdict(list)
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                source = str(data.get('source_file', ''))
                if source:
                    # Try to parse timestamp from filename first (more reliable than OS mtime)
                    # Expected format: *_YYYYMMDD_HHMMSS_ml_training.npz
                    import re
                    timestamp_match = re.search(r'_(\d{8}_\d{6})_ml_training\.npz$', npz_file.name)

                    if timestamp_match:
                        # Parse timestamp from filename
                        timestamp_str = timestamp_match.group(1)
                        from datetime import datetime
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').timestamp()
                        except ValueError:
                            # Fall back to file modification time if parsing fails
                            timestamp = os.path.getmtime(npz_file)
                    else:
                        # No timestamp in filename, use file modification time
                        timestamp = os.path.getmtime(npz_file)

                    source_to_files[source].append((npz_file, timestamp))
            except Exception as e:
                print(f"Warning: Could not read {npz_file.name} for deduplication: {e}")
                continue

        # Keep only the most recent file for each source
        deduplicated_files = []
        for source, file_list in source_to_files.items():
            # Sort by timestamp (most recent first)
            file_list.sort(key=lambda x: x[1], reverse=True)
            most_recent = file_list[0][0]
            deduplicated_files.append(most_recent)

            if len(file_list) > 1:
                print(f"  Deduplication: Found {len(file_list)} files from source '{source}', keeping most recent: {most_recent.name}")

        npz_files = deduplicated_files
        print(f"After deduplication: {len(npz_files)} files")

    # Load first file to get feature names and dataset type
    first_data = np.load(npz_files[0], allow_pickle=True)

    # Determine dataset type based on model number
    if model_number == 1:
        # Model 1: Breath vs Noise (binary classification)
        # Use all_peaks dataset
        if 'all_peaks_columns' not in first_data:
            raise ValueError("NPZ file missing 'all_peaks_columns' - required for Model 1 training")
        dataset_type = 'all_peaks'
        col_key = 'all_peaks_columns'
        label_key = 'all_peaks_is_breath'
    elif model_number == 2:
        # Model 2: Sigh vs Normal Breath (binary classification)
        # Use breaths dataset (only labeled breaths)
        if 'breaths_columns' not in first_data:
            raise ValueError("NPZ file missing 'breaths_columns' - required for Model 2 training")
        dataset_type = 'breaths'
        col_key = 'breaths_columns'
        label_key = None  # Computed from is_sigh
    elif model_number == 3:
        # Model 3: Eupnea vs Sniffing (binary classification)
        # Use breaths dataset, excluding sighs
        if 'breaths_columns' not in first_data:
            raise ValueError("NPZ file missing 'breaths_columns' - required for Model 3 training")
        dataset_type = 'breaths'
        col_key = 'breaths_columns'
        label_key = None  # Computed from is_eupnea and is_sniffing
    else:
        raise ValueError(f"Invalid model_number: {model_number}. Must be 1, 2, or 3.")

    # Collect data from all files
    all_dfs = []
    all_labels = []
    all_label_sources = []  # Track label sources for baseline calculation
    all_labels_for_baseline = []  # Track actual labels alongside sources

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)

        # Get column names
        columns = data[col_key]

        # Build DataFrame
        # Keys in NPZ are like "all_peaks_column_name" or "breaths_column_name"
        df_data = {col: data[f"{dataset_type}_{col}"]
                   for col in columns}
        df = pd.DataFrame(df_data)

        # Remove metadata columns (not features for ML)
        metadata_cols = ['sweep_idx', 'peak_idx', 'is_breath', 'label_source',
                        'is_sigh', 'is_eupnea', 'is_sniffing', 'was_merged_away']
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        X_subset = df[feature_cols]

        # Store label source for baseline calculation (only for Model 1)
        label_source_subset = None
        if model_number == 1 and 'label_source' in df.columns:
            label_source_subset = df['label_source'].values

        # Get labels based on model number
        if model_number == 1:
            # Model 1: Binary classification (breath vs noise)
            labels = data[label_key]

        elif model_number == 2:
            # Model 2: Sigh vs Normal Breath (binary classification)
            is_sigh = df['is_sigh'].values

            # Create binary labels: 0=normal breath, 1=sigh
            labels = is_sigh.astype(int)

            # No filtering needed - all breaths are either sigh or normal
            print(f"  File {npz_file.name}: {len(X_subset)} breaths "
                  f"(normal: {(labels==0).sum()}, sigh: {(labels==1).sum()})")

        elif model_number == 3:
            # Model 3: Eupnea vs Sniffing (binary classification)
            # Exclude sighs from this dataset
            is_sigh = df['is_sigh'].values
            is_eupnea = df['is_eupnea'].values
            is_sniffing = df['is_sniffing'].values

            # Filter out sighs
            non_sigh_mask = (is_sigh == 0)
            X_subset = X_subset[non_sigh_mask]

            # Get eupnea/sniffing labels for non-sigh breaths
            is_eupnea_filtered = is_eupnea[non_sigh_mask]
            is_sniffing_filtered = is_sniffing[non_sigh_mask]

            # Create binary labels: 0=eupnea, 1=sniffing
            # Only include breaths that are clearly labeled as one or the other
            labels = np.full(len(X_subset), -1, dtype=int)
            labels[is_eupnea_filtered == 1] = 0  # Eupnea
            labels[is_sniffing_filtered == 1] = 1  # Sniffing

            # Filter out unlabeled breaths (neither eupnea nor sniffing)
            labeled_mask = (labels != -1)
            X_subset = X_subset[labeled_mask]
            labels = labels[labeled_mask]

            print(f"  File {npz_file.name}: {len(X_subset)} labeled normal breaths "
                  f"(eupnea: {(labels==0).sum()}, sniffing: {(labels==1).sum()})")

        all_dfs.append(X_subset)
        all_labels.append(labels)
        if label_source_subset is not None:
            all_label_sources.append(label_source_subset)
            all_labels_for_baseline.append(labels)

    # Combine all data
    X = pd.concat(all_dfs, ignore_index=True)
    y = pd.Series(np.concatenate(all_labels))

    print(f"Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")

    # Calculate baseline threshold accuracy and per-class recall (only for Model 1)
    baseline_accuracy = None
    baseline_recall_dict = None
    if model_number == 1 and len(all_label_sources) > 0:
        label_sources = np.concatenate(all_label_sources)
        labels_baseline = np.concatenate(all_labels_for_baseline)

        # Overall accuracy: how many were correctly labeled by threshold alone
        is_auto = np.isin(label_sources, ['auto', 'threshold'])
        n_auto = is_auto.sum()
        baseline_accuracy = n_auto / len(label_sources) if len(label_sources) > 0 else None

        # Per-class recall: for each class, what fraction was auto-labeled?
        baseline_recall_dict = {}
        for class_val in np.unique(labels_baseline):
            class_mask = (labels_baseline == class_val)
            n_class_auto = (is_auto & class_mask).sum()
            n_class_total = class_mask.sum()
            recall = n_class_auto / n_class_total if n_class_total > 0 else 0
            baseline_recall_dict[str(class_val)] = recall

        if baseline_accuracy is not None:
            n_manual = np.isin(label_sources, ['manual', 'user']).sum()
            print(f"\nBaseline threshold performance:")
            print(f"  Overall accuracy: {baseline_accuracy:.1%}")
            print(f"  Correct by threshold: {n_auto} ({100*baseline_accuracy:.1f}%)")
            print(f"  Manual corrections needed: {n_manual} ({100*n_manual/len(label_sources):.1f}%)")
            print(f"  Per-class recall:")
            for class_val, recall in baseline_recall_dict.items():
                print(f"    Class {class_val}: {recall:.1%}")

    # Handle missing values (NaN)
    # Check for NaN values
    nan_counts = X.isna().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        print(f"Warning: Found {total_nans} NaN values across {(nan_counts > 0).sum()} features")
        print("Features with NaN values:")
        for col in X.columns[nan_counts > 0]:
            print(f"  {col}: {nan_counts[col]} NaNs")

        # Drop rows with any NaN values
        # This is safer than imputation for ML training
        initial_len = len(X)
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        dropped = initial_len - len(X)

        if dropped > 0:
            print(f"Dropped {dropped} samples with NaN values ({dropped/initial_len:.1%} of data)")
            print(f"Remaining: {len(X)} samples")
            print(f"Updated label distribution: {y.value_counts().to_dict()}")

    return X, y, dataset_type, baseline_accuracy, baseline_recall_dict


def save_model(model: Any, filepath: Path, metadata: dict = None):
    """
    Save trained model to disk using pickle with metadata.

    Args:
        model: Trained sklearn/xgboost model
        filepath: Path to save .pkl file
        metadata: Optional dict with 'model_type', 'feature_names', 'model_number', etc.
    """
    import pickle

    # Package model with metadata
    model_package = {
        'model': model,
        'metadata': metadata or {}
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Model saved to {filepath}")
    if metadata:
        print(f"  Model type: {metadata.get('model_type', 'unknown')}")
        print(f"  Model number: {metadata.get('model_number', 'unknown')}")
        print(f"  Features: {len(metadata.get('feature_names', []))} features")


def load_model(filepath: Path) -> tuple:
    """
    Load trained model from disk.

    Returns:
        tuple: (model, metadata_dict) if saved with metadata
               or (model, {}) if old format (backward compatible)
    """
    import pickle

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Handle old format (just model object)
    if not isinstance(data, dict):
        print(f"Model loaded from {filepath} (old format, no metadata)")
        return data, {}

    # Handle new format (model + metadata)
    model = data.get('model')
    metadata = data.get('metadata', {})

    print(f"Model loaded from {filepath}")
    if metadata:
        print(f"  Model type: {metadata.get('model_type', 'unknown')}")
        print(f"  Model number: {metadata.get('model_number', 'unknown')}")
        print(f"  Features: {len(metadata.get('feature_names', []))} features")

    return model, metadata
