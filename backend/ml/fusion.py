"""
PhysioTracker v3 — Sensor Fusion with proper ML model.
Uses sklearn RF+GB ensemble; falls back to rule-based if unavailable.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


ACTIVITIES = ["idle", "walk", "squat", "bend", "lift", "fall_risk"]

DEFAULT_WEIGHTS = {
    "wifi":       0.20,
    "bluetooth":  0.15,
    "imu":        0.40,
    "acoustic":   0.15,
    "ambient_rf": 0.10,
}


@dataclass
class FusionResult:
    fused_score:      float = 0.0
    activity:         str   = "idle"
    confidence:       float = 0.0
    ml_probabilities: dict  = field(default_factory=dict)
    gait_score:       float = 0.0
    rom_score:        float = 0.0
    symmetry_score:   float = 0.0
    rep_count:        int   = 0
    fatigue_index:    float = 0.0
    step_cadence:     float = 0.0
    fall_risk:        float = 0.0
    sensor_scores:    dict  = field(default_factory=dict)
    imu_features:     dict  = field(default_factory=dict)
    wifi_features:    dict  = field(default_factory=dict)
    bt_features:      dict  = field(default_factory=dict)
    acoustic_features: dict = field(default_factory=dict)
    rf_features:      dict  = field(default_factory=dict)
    model_info:       dict  = field(default_factory=dict)
    effective_weights: dict = field(default_factory=dict)


class SensorFusion:
    def __init__(self, weights=None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._fatigue_buffer = deque(maxlen=30)
        try:
            from ml.model import get_model
            self._ml = get_model()
        except Exception as e:
            print(f"[Fusion] ML model init error: {e}")
            self._ml = None

    def fuse(self, wifi_feat, bt_feat, imu_feat, acoustic_feat, rf_feat) -> FusionResult:
        scores = {
            "wifi":       wifi_feat.get("motion_score", 0),
            "bluetooth":  bt_feat.get("motion_score", 0),
            "imu":        imu_feat.get("motion_score", 0),
            "acoustic":   acoustic_feat.get("motion_score", 0),
            "ambient_rf": rf_feat.get("motion_score", 0),
        }
        effective_weights = self._compute_dynamic_weights(wifi_feat, scores)
        fused = round(min(100.0, sum(scores[k] * effective_weights[k] for k in scores)), 2)

        ml_probabilities = {}
        model_info = {}
        if self._ml is not None:
            activity, confidence, ml_probabilities = self._ml.predict(imu_feat, scores)
            model_info = self._ml.get_info()
        else:
            activity, confidence = self._rule_fallback(imu_feat, fused)

        accel_rms = imu_feat.get("accel_rms", 0.0)
        gyro_rms  = imu_feat.get("gyro_rms", 0.0)
        jerk      = imu_feat.get("jerk", 0.0)
        cadence   = imu_feat.get("step_cadence", 0.0)
        rep_count = imu_feat.get("rep_count", 0)

        gait_score = min(100.0, max(0.0, 50 + (cadence - 80) * 0.3 - jerk * 10)) if activity == "walk" and cadence > 0 else 0.0
        rom_score = min(100.0, gyro_rms * 35)
        sym = 1.0 - min(1.0, abs(accel_rms - gyro_rms * 0.3) / 2.0)
        symmetry_score = round(sym * 100, 1)

        self._fatigue_buffer.append(jerk)
        if len(self._fatigue_buffer) >= 10:
            early = np.mean(list(self._fatigue_buffer)[:5])
            late  = np.mean(list(self._fatigue_buffer)[-5:])
            fatigue_index = min(100.0, max(0.0, (late - early) * 200))
        else:
            fatigue_index = 0.0

        fall_risk_base = min(100.0, jerk * 80 + (fused > 70) * 20)
        if ml_probabilities:
            ml_fall = ml_probabilities.get("fall_risk", 0.0)
            fall_risk = round(fall_risk_base * 0.4 + ml_fall * 100 * 0.6, 1)
        else:
            fall_risk = round(fall_risk_base, 1)

        return FusionResult(
            fused_score=fused, activity=activity, confidence=round(confidence, 2),
            ml_probabilities=ml_probabilities, gait_score=round(gait_score, 1),
            rom_score=round(rom_score, 1), symmetry_score=symmetry_score,
            rep_count=rep_count, fatigue_index=round(fatigue_index, 1),
            step_cadence=round(cadence, 1), fall_risk=fall_risk,
            sensor_scores=scores, imu_features=imu_feat, wifi_features=wifi_feat,
            bt_features=bt_feat, acoustic_features=acoustic_feat, rf_features=rf_feat,
            model_info=model_info, effective_weights=effective_weights,
        )

    def _compute_dynamic_weights(self, wifi_feat: dict, scores: dict) -> dict:
        """Adapt sensor weights with WiFi motion quality while preserving stability."""
        wifi_motion = float(max(0.0, min(100.0, wifi_feat.get("motion_score", 0.0)))) / 100.0
        wifi_real = bool(wifi_feat.get("is_real", False))
        wifi_nets = float(max(0.0, min(12.0, wifi_feat.get("n_networks", 0.0)))) / 12.0
        imu_motion = float(max(0.0, min(100.0, scores.get("imu", 0.0)))) / 100.0

        # WiFi gets stronger influence as RSSI dynamics and AP visibility improve.
        wifi_target = 0.12 + 0.18 * wifi_motion + 0.10 * wifi_nets + (0.06 if wifi_real else 0.0)
        wifi_target = min(0.46, max(0.10, wifi_target))

        raw = {
            "wifi": wifi_target,
            "imu": 0.18 + 0.28 * imu_motion,
            "bluetooth": 0.10,
            "acoustic": 0.12 + 0.04 * wifi_motion,
            "ambient_rf": 0.10 + 0.05 * wifi_motion,
        }

        total = sum(raw.values()) or 1.0
        return {k: raw[k] / total for k in raw}

    def _rule_fallback(self, imu_feat, fused):
        accel = imu_feat.get("accel_rms", 0.0)
        gyro  = imu_feat.get("gyro_rms", 0.0)
        jerk  = imu_feat.get("jerk", 0.0)
        cad   = imu_feat.get("step_cadence", 0.0)
        if jerk > 0.5 and accel > 1.5: return "fall_risk", 0.85
        if accel < 0.18: return "idle", 0.90
        if 60 <= cad <= 130 and accel > 0.3: return "walk", 0.85
        if gyro > 0.8 and cad < 40: return ("squat" if accel < 0.8 else "bend"), 0.75
        if accel > 0.8: return "lift", 0.70
        if accel > 0.25: return "walk", 0.60
        return "idle", 0.55

    def get_model_info(self):
        if self._ml:
            return {**self._ml.get_info(), "importances": self._ml.get_feature_importances()}
        return {"trained": False, "model": "rule-based"}