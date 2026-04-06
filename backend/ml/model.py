"""
PhysioTracker v3 — Proper ML Model with sklearn
Trains a RandomForest + GradientBoosting ensemble on synthetic IMU + WiFi features.
Exports predict() and get_model_info() for use by fusion.py.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
import os
import json

# ── Synthetic training data generator ────────────────────────────────────────
ACTIVITIES = ["idle", "walk", "squat", "bend", "lift", "fall_risk"]

def _gen_samples(activity: str, n: int = 300) -> np.ndarray:
    """Generate realistic synthetic feature vectors per activity."""
    rng = np.random.default_rng(hash(activity) % (2**31))

    def noisy(center, scale, size):
        return rng.normal(center, scale, size)

    if activity == "idle":
        accel_rms   = noisy(0.08, 0.03, n).clip(0.01, 0.25)
        gyro_rms    = noisy(0.05, 0.02, n).clip(0.01, 0.15)
        jerk        = noisy(0.02, 0.01, n).clip(0, 0.1)
        cadence     = noisy(0, 2, n).clip(0, 8)
        rep_count   = rng.integers(0, 2, n).astype(float)
        wifi_score  = noisy(5, 3, n).clip(0, 20)
        bt_score    = noisy(3, 2, n).clip(0, 15)
        acou_score  = noisy(2, 2, n).clip(0, 10)
        rf_score    = noisy(2, 2, n).clip(0, 12)

    elif activity == "walk":
        accel_rms   = noisy(0.55, 0.12, n).clip(0.25, 1.0)
        gyro_rms    = noisy(0.35, 0.1, n).clip(0.1, 0.8)
        jerk        = noisy(0.08, 0.03, n).clip(0.02, 0.25)
        cadence     = noisy(95, 12, n).clip(60, 130)
        rep_count   = rng.integers(0, 4, n).astype(float)
        wifi_score  = noisy(25, 7, n).clip(5, 50)
        bt_score    = noisy(18, 6, n).clip(5, 40)
        acou_score  = noisy(20, 8, n).clip(5, 45)
        rf_score    = noisy(15, 6, n).clip(3, 40)

    elif activity == "squat":
        accel_rms   = noisy(0.58, 0.11, n).clip(0.28, 1.05)
        gyro_rms    = noisy(0.78, 0.14, n).clip(0.38, 1.45)
        jerk        = noisy(0.14, 0.05, n).clip(0.05, 0.38)
        cadence     = noisy(24, 6, n).clip(10, 42)
        rep_count   = noisy(10, 3, n).clip(4, 22)
        wifi_score  = noisy(30, 8, n).clip(8, 55)
        bt_score    = noisy(22, 7, n).clip(5, 45)
        acou_score  = noisy(25, 9, n).clip(8, 50)
        rf_score    = noisy(20, 7, n).clip(5, 45)

    elif activity == "bend":
        accel_rms   = noisy(0.32, 0.07, n).clip(0.12, 0.62)
        gyro_rms    = noisy(1.22, 0.18, n).clip(0.65, 2.0)
        jerk        = noisy(0.09, 0.03, n).clip(0.02, 0.26)
        cadence     = noisy(11, 4, n).clip(2, 24)
        rep_count   = noisy(3, 1.4, n).clip(0, 8)
        wifi_score  = noisy(22, 7, n).clip(5, 45)
        bt_score    = noisy(16, 5, n).clip(4, 35)
        acou_score  = noisy(18, 7, n).clip(4, 40)
        rf_score    = noisy(14, 5, n).clip(3, 35)

    elif activity == "lift":
        accel_rms   = noisy(1.0, 0.18, n).clip(0.55, 1.65)
        gyro_rms    = noisy(0.68, 0.14, n).clip(0.25, 1.25)
        jerk        = noisy(0.2, 0.06, n).clip(0.08, 0.55)
        cadence     = noisy(16, 5, n).clip(5, 34)
        rep_count   = noisy(9, 4, n).clip(2, 20)
        wifi_score  = noisy(40, 10, n).clip(10, 65)
        bt_score    = noisy(28, 8, n).clip(8, 55)
        acou_score  = noisy(35, 10, n).clip(10, 65)
        rf_score    = noisy(28, 8, n).clip(8, 55)

    elif activity == "fall_risk":
        accel_rms   = noisy(1.8, 0.4, n).clip(1.0, 3.0)
        gyro_rms    = noisy(1.4, 0.35, n).clip(0.6, 2.5)
        jerk        = noisy(0.65, 0.2, n).clip(0.3, 1.5)
        cadence     = noisy(40, 15, n).clip(10, 80)
        rep_count   = rng.integers(0, 3, n).astype(float)
        wifi_score  = noisy(50, 15, n).clip(15, 90)
        bt_score    = noisy(40, 12, n).clip(10, 75)
        acou_score  = noisy(55, 15, n).clip(20, 90)
        rf_score    = noisy(45, 12, n).clip(15, 80)

    else:
        return np.zeros((n, 9))

    return np.column_stack([
        accel_rms, gyro_rms, jerk, cadence, rep_count,
        wifi_score, bt_score, acou_score, rf_score
    ])


FEATURE_NAMES = [
    "accel_rms", "gyro_rms", "jerk", "cadence", "rep_count",
    "wifi_score", "bt_score", "acou_score", "rf_score"
]

CORPUS_JSONL = os.path.join(os.path.dirname(__file__), "..", "data", "imu_corpus.jsonl")


def _extract_vec_from_samples(samples: list) -> Optional[np.ndarray]:
    if not isinstance(samples, list) or len(samples) < 8:
        return None
    try:
        arr = np.array([
            [
                float(s.get("ax", 0.0)),
                float(s.get("ay", 0.0)),
                float(s.get("az", 0.0)),
                float(s.get("gx", 0.0)),
                float(s.get("gy", 0.0)),
                float(s.get("gz", 0.0)),
                float(s.get("ts", 0.0)),
            ]
            for s in samples
        ], dtype=float)
    except Exception:
        return None

    accel = arr[:, :3]
    gyro = arr[:, 3:6]
    ts = arr[:, 6]

    accel_mag = np.linalg.norm(accel - np.mean(accel, axis=0), axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    accel_rms = float(np.sqrt(np.mean(accel_mag ** 2)))
    gyro_rms = float(np.sqrt(np.mean(gyro_mag ** 2)))
    diffs = np.abs(np.diff(accel_mag))
    jerk = float(np.mean(diffs)) if len(diffs) else 0.0

    dt = np.diff(ts)
    dt = dt[(dt > 1e-4) & (dt < 0.2)]
    sr = 1.0 / float(np.mean(dt)) if len(dt) else 50.0
    sr = max(10.0, min(200.0, sr))

    if len(accel_mag) >= 16:
        fft = np.fft.rfft(accel_mag - np.mean(accel_mag))
        freqs = np.fft.rfftfreq(len(accel_mag), d=1.0 / sr)
        low = np.where((freqs >= 0.3) & (freqs <= 3.5))[0]
        if len(low):
            dom = float(freqs[low[np.argmax(np.abs(fft[low]))]])
            cadence = dom * 60.0
        else:
            cadence = 0.0
    else:
        cadence = 0.0

    if len(gyro_mag) >= 10:
        thr = float(np.mean(gyro_mag) + np.std(gyro_mag) * 0.7)
        peaks = (gyro_mag[1:-1] > gyro_mag[:-2]) & (gyro_mag[1:-1] > gyro_mag[2:]) & (gyro_mag[1:-1] > thr)
        rep_count = float(np.sum(peaks))
    else:
        rep_count = 0.0

    return np.array([
        accel_rms,
        gyro_rms,
        jerk,
        cadence,
        rep_count,
        0.0,
        0.0,
        0.0,
        0.0,
    ], dtype=float)


def _load_real_corpus() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(CORPUS_JSONL):
        return np.zeros((0, 9)), np.zeros((0,), dtype=int)

    X_rows = []
    y_rows = []
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                act = str(rec.get("activity", "")).strip().lower()
                if act not in ACTIVITIES:
                    continue
                vec = _extract_vec_from_samples(rec.get("samples", []))
                if vec is None:
                    continue
                X_rows.append(vec)
                y_rows.append(ACTIVITIES.index(act))
            except Exception:
                continue

    if not X_rows:
        return np.zeros((0, 9)), np.zeros((0,), dtype=int)
    return np.vstack(X_rows), np.array(y_rows, dtype=int)


def _build_training_data():
    X_parts, y_parts = [], []
    for idx, act in enumerate(ACTIVITIES):
        samples = _gen_samples(act, 350)
        X_parts.append(samples)
        y_parts.extend([idx] * len(samples))
    X_syn = np.vstack(X_parts)
    y_syn = np.array(y_parts)

    X_real, y_real = _load_real_corpus()
    if len(y_real):
        # Upweight real windows so model adapts to phone distributions.
        X = np.vstack([X_syn, X_real, X_real])
        y = np.concatenate([y_syn, y_real, y_real])
        return X, y

    return X_syn, y_syn


# ── Model ─────────────────────────────────────────────────────────────────────
class PhysioMLModel:
    """
    Ensemble: RandomForest + GradientBoosting, soft-voted.
    Trains once at startup on synthetic data (~2100 samples).
    """

    def __init__(self):
        self._trained = False
        self._rf = None
        self._gb = None
        self._train_acc = 0.0
        self._label_to_idx = {a: i for i, a in enumerate(ACTIVITIES)}
        self._idx_to_label = {i: a for i, a in enumerate(ACTIVITIES)}
        self._history: deque = deque(maxlen=8)
        self._samples_trained = 0
        self._real_windows = 0
        self._train()

    def _train(self):
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            X, y = _build_training_data()
            self._samples_trained = int(len(y))
            X_real, y_real = _load_real_corpus()
            self._real_windows = int(len(y_real))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self._scaler = StandardScaler()
            X_train_s = self._scaler.fit_transform(X_train)
            X_test_s  = self._scaler.transform(X_test)

            self._rf = RandomForestClassifier(
                n_estimators=120, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=42
            )
            self._rf.fit(X_train_s, y_train)

            self._gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42
            )
            self._gb.fit(X_train_s, y_train)

            rf_acc = self._rf.score(X_test_s, y_test)
            gb_acc = self._gb.score(X_test_s, y_test)
            self._train_acc = round((rf_acc + gb_acc) / 2, 3)
            self._trained = True
            print(f"[ML] Trained — RF={rf_acc:.3f}, GB={gb_acc:.3f}, Ensemble≈{self._train_acc:.3f}")

        except ImportError:
            print("[ML] sklearn not available — falling back to rule-based classifier")
            self._trained = False

    def _extract_vec(self, imu_feat: dict, sensor_scores: dict) -> np.ndarray:
        return np.array([[
            imu_feat.get("accel_rms", 0.0),
            imu_feat.get("gyro_rms", 0.0),
            imu_feat.get("jerk", 0.0),
            imu_feat.get("step_cadence", 0.0),
            float(imu_feat.get("rep_count", 0)),
            sensor_scores.get("wifi", 0.0),
            sensor_scores.get("bluetooth", 0.0),
            sensor_scores.get("acoustic", 0.0),
            sensor_scores.get("ambient_rf", 0.0),
        ]])

    def _posture_bias(self, imu_feat: dict) -> dict:
        accel = float(imu_feat.get("accel_rms", 0.0))
        gyro = float(imu_feat.get("gyro_rms", 0.0))
        jerk = float(imu_feat.get("jerk", 0.0))
        cad = float(imu_feat.get("step_cadence", 0.0))
        reps = float(imu_feat.get("rep_count", 0.0))

        bend_score = 0.0
        squat_score = 0.0
        lift_score = 0.0

        if cad < 35:
            bend_score += max(0.0, (gyro - 0.95) * 0.22)
            bend_score += max(0.0, (0.55 - min(accel, 0.55)) * 0.9)
            bend_score += max(0.0, (5.0 - reps) * 0.05)
            bend_score += max(0.0, (0.16 - min(jerk, 0.16)) * 0.8)

            squat_score += max(0.0, (gyro - 0.68) * 0.18)
            squat_score += max(0.0, (accel - 0.32) * 0.25)
            squat_score += max(0.0, (reps - 3.0) * 0.06)
            squat_score += max(0.0, (28.0 - cad) * 0.01)

            lift_score += max(0.0, (accel - 0.85) * 0.28)
            lift_score += max(0.0, (gyro - 0.55) * 0.08)

        return {
            "bend": min(0.25, bend_score),
            "squat": min(0.25, squat_score),
            "lift": min(0.2, lift_score),
        }

    def predict(self, imu_feat: dict, sensor_scores: dict) -> Tuple[str, float, dict]:
        """
        Returns (activity_label, confidence, proba_dict).
        Falls back to rule-based if sklearn unavailable.
        """
        if not self._trained:
            return self._rule_based(imu_feat)

        vec = self._extract_vec(imu_feat, sensor_scores)
        vec_s = self._scaler.transform(vec)

        rf_proba = self._rf.predict_proba(vec_s)[0]
        gb_proba = self._gb.predict_proba(vec_s)[0]
        ensemble_proba = (rf_proba * 0.55 + gb_proba * 0.45)

        # Temporal smoothing: average with recent history
        self._history.append(ensemble_proba)
        if len(self._history) >= 3:
            smoothed = np.mean(list(self._history), axis=0)
        else:
            smoothed = ensemble_proba

        posture_bias = self._posture_bias(imu_feat)
        for label, boost in posture_bias.items():
            idx = self._label_to_idx.get(label)
            if idx is not None:
                smoothed[idx] += boost
        smoothed = np.clip(smoothed, 1e-6, None)
        smoothed = smoothed / np.sum(smoothed)

        best_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[best_idx])
        activity = self._idx_to_label[best_idx]

        proba_dict = {self._idx_to_label[i]: round(float(p), 4)
                      for i, p in enumerate(smoothed)}
        return activity, round(confidence, 3), proba_dict

    def _rule_based(self, imu_feat: dict) -> Tuple[str, float, dict]:
        """Fallback when sklearn isn't available."""
        accel = imu_feat.get("accel_rms", 0.0)
        gyro  = imu_feat.get("gyro_rms", 0.0)
        jerk  = imu_feat.get("jerk", 0.0)
        cad   = imu_feat.get("step_cadence", 0.0)
        reps  = imu_feat.get("rep_count", 0.0)

        if jerk > 0.5 and accel > 1.5:
            act, conf = "fall_risk", 0.85
        elif accel < 0.18:
            act, conf = "idle", 0.90
        elif 60 <= cad <= 130 and accel > 0.3:
            act, conf = "walk", 0.82
        elif cad < 35 and gyro > 1.0 and reps <= 5:
            act, conf = "bend", 0.80
        elif cad < 40 and gyro > 0.7 and reps >= 4:
            act, conf = "squat", 0.78
        elif accel > 0.9:
            act, conf = "lift", 0.74
        elif accel > 0.25:
            act, conf = "walk", 0.60
        else:
            act, conf = "idle", 0.55

        proba = {a: 0.02 for a in ACTIVITIES}
        proba[act] = conf
        return act, conf, proba

    def get_info(self) -> dict:
        return {
            "trained": self._trained,
            "accuracy": self._train_acc,
            "model": "RF(120)+GB(100) ensemble" if self._trained else "rule-based",
            "features": FEATURE_NAMES,
            "activities": ACTIVITIES,
            "samples_trained": self._samples_trained,
            "real_windows": self._real_windows,
        }

    def get_feature_importances(self) -> dict:
        if not self._trained or self._rf is None:
            return {}
        fi = self._rf.feature_importances_
        return {FEATURE_NAMES[i]: round(float(fi[i]), 4) for i in range(len(fi))}


# Singleton
_model: Optional[PhysioMLModel] = None

def get_model() -> PhysioMLModel:
    global _model
    if _model is None:
        _model = PhysioMLModel()
    return _model