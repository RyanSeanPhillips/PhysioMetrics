"""
ML File Type Classifier — classifies files as data, notes, export, etc.

Pure Python service (no Qt). Uses scikit-learn RandomForest trained from
user-labeled data stored in pj_file_labels.

Features:
- File extension, size, modification recency
- Path depth from project root
- Proximity to nearest data file (folder hops)
- Filename keyword scoring (positive/negative)
- Content fingerprint (column count, row count, ABF ref density) for tabular files

Falls back to heuristic scoring when <20 labeled examples.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

# Labels
LABELS = ["data", "notes", "export", "cage_card", "surgery_notes", "other"]

# Keyword sets for heuristic features
NEGATIVE_KEYWORDS = {
    "consolidated", "export", "analysis", "output", "results", "summary",
    "backup", "copy", "old", "archive", "temp", "template", "processed",
}
POSITIVE_KEYWORDS = {
    "log", "notes", "protocol", "record", "experiment", "animal",
    "surgery", "cage", "genotype", "breeding",
}
DATA_EXTENSIONS = {".abf", ".smrx", ".edf", ".mat", ".nwb", ".tif", ".tiff"}
NOTES_EXTENSIONS = {".xlsx", ".xls", ".csv", ".txt", ".docx", ".doc"}


class FileClassifier:
    """
    Classifies files by type using ML (when trained) or heuristics (fallback).

    Usage:
        classifier = FileClassifier()
        classifier.train(labeled_data)  # List of {file_path, label, features}
        result = classifier.classify(path, project_root, data_folders)
    """

    def __init__(self):
        self._model = None
        self._feature_names: List[str] = []
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(
        path: Path,
        project_root: Optional[Path] = None,
        data_folders: Optional[Set[Path]] = None,
    ) -> Dict[str, float]:
        """Extract numeric features from a file for classification."""
        features: Dict[str, float] = {}

        # Extension features (one-hot)
        ext = path.suffix.lower()
        features["ext_xlsx"] = 1.0 if ext in (".xlsx", ".xls") else 0.0
        features["ext_csv"] = 1.0 if ext == ".csv" else 0.0
        features["ext_txt"] = 1.0 if ext == ".txt" else 0.0
        features["ext_docx"] = 1.0 if ext in (".docx", ".doc") else 0.0
        features["ext_data"] = 1.0 if ext in DATA_EXTENSIONS else 0.0
        features["ext_image"] = 1.0 if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp") else 0.0
        features["ext_pdf"] = 1.0 if ext == ".pdf" else 0.0

        # Size features
        try:
            size_bytes = path.stat().st_size
            features["size_kb"] = size_bytes / 1024
            features["size_log"] = _safe_log(size_bytes)
            features["size_small"] = 1.0 if size_bytes < 100_000 else 0.0
            features["size_large"] = 1.0 if size_bytes > 500_000 else 0.0
        except OSError:
            features["size_kb"] = 0.0
            features["size_log"] = 0.0
            features["size_small"] = 0.0
            features["size_large"] = 0.0

        # Path depth
        if project_root:
            try:
                rel = path.relative_to(project_root)
                features["path_depth"] = len(rel.parts) - 1  # Exclude filename
            except ValueError:
                features["path_depth"] = 0.0
        else:
            features["path_depth"] = 0.0

        # Proximity to data files
        if data_folders and project_root:
            features["proximity"] = _folder_distance(
                path.parent.resolve(), data_folders, project_root
            )
        else:
            features["proximity"] = 5.0  # Unknown

        # Keyword features
        stem_lower = path.stem.lower()
        features["neg_keywords"] = sum(1 for kw in NEGATIVE_KEYWORDS if kw in stem_lower)
        features["pos_keywords"] = sum(1 for kw in POSITIVE_KEYWORDS if kw in stem_lower)

        # Name length (longer names tend to be descriptive notes)
        features["name_length"] = len(path.stem)

        # Has numbers in name (data files often have numeric IDs)
        features["has_numbers"] = 1.0 if re.search(r"\d{4,}", path.stem) else 0.0

        return features

    # ------------------------------------------------------------------
    # Heuristic classifier (fallback when <20 labels)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_heuristic(
        path: Path,
        project_root: Optional[Path] = None,
        data_folders: Optional[Set[Path]] = None,
    ) -> Dict[str, Any]:
        """Classify a file using rule-based heuristics. No training needed."""
        ext = path.suffix.lower()

        # Data files — high confidence
        if ext in DATA_EXTENSIONS:
            return {"label": "data", "confidence": 0.95, "method": "heuristic"}

        # Images — could be cage cards
        if ext in (".png", ".jpg", ".jpeg", ".bmp"):
            stem = path.stem.lower()
            if any(kw in stem for kw in ("cage", "card", "genotype", "breeding")):
                return {"label": "cage_card", "confidence": 0.7, "method": "heuristic"}
            return {"label": "other", "confidence": 0.4, "method": "heuristic"}

        # Tabular/text files — need more analysis
        if ext in NOTES_EXTENSIONS:
            features = FileClassifier.extract_features(path, project_root, data_folders)

            score = 0.5  # Start neutral

            # Size
            if features["size_small"]:
                score += 0.15
            if features["size_large"]:
                score -= 0.2

            # Keywords
            score += features["pos_keywords"] * 0.1
            score -= features["neg_keywords"] * 0.15

            # Proximity
            prox = features["proximity"]
            if prox <= 1:
                score += 0.1
            elif prox >= 3:
                score -= 0.1

            score = max(0.0, min(1.0, score))

            if features["neg_keywords"] >= 2:
                return {"label": "export", "confidence": score, "method": "heuristic"}
            if score >= 0.5:
                return {"label": "notes", "confidence": score, "method": "heuristic"}
            return {"label": "other", "confidence": 1.0 - score, "method": "heuristic"}

        return {"label": "other", "confidence": 0.3, "method": "heuristic"}

    # ------------------------------------------------------------------
    # ML classifier
    # ------------------------------------------------------------------

    def train(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the classifier from labeled data.

        Args:
            labeled_data: List of dicts with 'file_path', 'label', and optionally 'features'.

        Returns:
            Training summary with accuracy and feature importance.
        """
        if len(labeled_data) < 20:
            return {
                "trained": False,
                "reason": f"Need at least 20 labeled examples, have {len(labeled_data)}",
                "label_count": len(labeled_data),
            }

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            import numpy as np
        except ImportError:
            return {"trained": False, "reason": "scikit-learn not installed"}

        # Extract features for all labeled files
        X_rows = []
        y_labels = []
        for item in labeled_data:
            feat = item.get("features", {})
            if not feat:
                # Re-extract features if not cached
                p = Path(item["file_path"])
                if p.exists():
                    feat = self.extract_features(p)
                else:
                    continue

            X_rows.append(feat)
            y_labels.append(item["label"])

        if len(X_rows) < 20:
            return {"trained": False, "reason": "Not enough valid examples after filtering"}

        # Align feature names
        self._feature_names = sorted(X_rows[0].keys())
        X = np.array([[row.get(f, 0.0) for f in self._feature_names] for row in X_rows])
        y = np.array(y_labels)

        # Train
        clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)

        # Cross-validation
        scores = cross_val_score(clf, X, y, cv=min(5, len(set(y))), scoring="accuracy")

        # Final fit on all data
        clf.fit(X, y)
        self._model = clf
        self._trained = True

        # Feature importance
        importances = dict(zip(self._feature_names, clf.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "trained": True,
            "samples": len(X),
            "labels": dict(zip(*np.unique(y, return_counts=True))),
            "cv_accuracy": round(float(np.mean(scores)), 3),
            "top_features": [{"name": n, "importance": round(float(v), 3)} for n, v in top_features],
        }

    def classify(
        self,
        path: Path,
        project_root: Optional[Path] = None,
        data_folders: Optional[Set[Path]] = None,
    ) -> Dict[str, Any]:
        """
        Classify a file. Uses ML model if trained, otherwise heuristics.

        Returns:
            Dict with label, confidence, method, and probabilities (if ML).
        """
        if not self._trained or self._model is None:
            return self.classify_heuristic(path, project_root, data_folders)

        try:
            import numpy as np
        except ImportError:
            return self.classify_heuristic(path, project_root, data_folders)

        features = self.extract_features(path, project_root, data_folders)
        X = np.array([[features.get(f, 0.0) for f in self._feature_names]])

        label = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]
        confidence = float(max(proba))

        class_labels = self._model.classes_
        probabilities = {str(c): round(float(p), 3) for c, p in zip(class_labels, proba)}

        return {
            "label": str(label),
            "confidence": round(confidence, 3),
            "method": "ml",
            "probabilities": probabilities,
        }

    def export_model(self) -> Optional[bytes]:
        """Serialize the trained model to bytes (pickle). Returns None if not trained."""
        if not self._trained or self._model is None:
            return None
        import pickle
        return pickle.dumps({"model": self._model, "features": self._feature_names})

    def load_model(self, data: bytes) -> bool:
        """Load a serialized model from bytes."""
        try:
            import pickle
            obj = pickle.loads(data)
            self._model = obj["model"]
            self._feature_names = obj["features"]
            self._trained = True
            return True
        except Exception:
            return False


# === Helpers ===

def _safe_log(x: float) -> float:
    """Safe log10 for feature engineering."""
    import math
    return math.log10(max(x, 1.0))


def _folder_distance(target: Path, data_folders: Set[Path], root: Path) -> float:
    """Minimum folder hops from target to nearest data folder."""
    if not data_folders:
        return 99.0
    try:
        target_parts = target.relative_to(root.resolve()).parts
    except ValueError:
        return 99.0

    min_dist = 99.0
    for df in data_folders:
        try:
            df_parts = df.relative_to(root.resolve()).parts
        except ValueError:
            continue
        common = 0
        for a, b in zip(target_parts, df_parts):
            if a == b:
                common += 1
            else:
                break
        dist = (len(target_parts) - common) + (len(df_parts) - common)
        min_dist = min(min_dist, float(dist))
    return min_dist
