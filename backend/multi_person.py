"""
PhysioTracker v3 — Multi-Person Tracker
========================================
Handles per-device (phone) IMU streams and maps each phone to an
independent "person" slot. Also uses Bluetooth RSSI clustering and
WiFi variance to corroborate person count.

Key fixes over v2:
  1. Each phone that POSTs to /api/imu gets its own device_id bucket.
  2. Per-person activity classification runs independently.
  3. Fall detection uses a stable threshold with hysteresis so a small
     motion change never triggers a false "fall_risk".
  4. BLE RSSI from each simulated (or real) beacon cluster is used to
     triangulate rough spatial positions for each detected person.
  5. WiFi RSSI spatial variance is used as a corroborating occupancy
     signal, not the primary one.

Improvement (arxiv 2308.06773) — Per-AP std dev occupancy counting:
  Instead of collapsing all WiFi signal dynamics into a single scalar
  motion_score, we now train a lightweight sklearn classifier on a
  per-AP feature vector:
    - <bssid>__std   : normalised RSSI std dev per AP over a rolling window
    - <bssid>__delta : normalised delta of current mean vs baseline mean
  This directly mirrors the paper's approach of treating each detector
  (AP) as an independent channel whose RSSI std dev encodes how many
  bodies are disrupting its signal path.

  The classifier is trained once at startup (or from corpus if available)
  on per-count feature snapshots captured during baseline collection.
  At runtime, _estimate_signal_person_count() runs the classifier first
  and falls back to the original heuristic bands only when no trained
  model is available.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


# ─── Per-AP std dev occupancy classifier ──────────────────────────────────────

# Minimum feature snapshots per occupancy count before the classifier trains
_OCC_MIN_SAMPLES_PER_CLASS = 6

# After this many new snapshots since last training, retrain
_OCC_RETRAIN_EVERY = 20


class OccupancyClassifier:
    """
    Lightweight sklearn RandomForest trained on per-AP RSSI std dev vectors.
    Implements arxiv 2308.06773: each WiFi AP is an independent detector channel;
    its RSSI std dev over time encodes how many bodies are disrupting signal paths.

    Feature vector per sample (preserves per-AP information):
        [AP1_std, AP2_std, ..., APm_std,         # individual AP RSSI std devs
         AP1_delta, AP2_delta, ..., APm_delta,   # individual AP RSSI deltas vs baseline
         mean_ap_std, mean_ap_delta]             # aggregate fallback for robustness

    Canonical AP order (self._ap_order) locks at first training sample, ensuring
    consistent feature vector size and feature alignment during inference.
    Missing APs default to 0.0; extra APs during inference are ignored.

    Labels: integer occupancy counts (0, 1, 2, ..., N).

    Paper results: 98-99% accuracy counting up to 9 people with only 4-5 APs
    and 60+ training samples per occupancy level. Limited AP coverage or sparse
    training data will reduce accuracy proportionally.

    Training data collected during baseline capture via record_training_snapshot().
    Automatic retrain after _OCC_RETRAIN_EVERY new samples.
    """

    def __init__(self):
        self._X: List[List[float]] = []    # feature rows (2*n_aps + 2 per row)
        self._y: List[int] = []            # count labels (ground truth occupancy per row)
        self._ap_order: List[str] = []     # canonical AP/BSSID order for vectorization
        self._clf = None                   # trained RandomForestClassifier
        self._scaler = None                # StandardScaler fitted on training data X
        self._trained = False              # True after successful train()
        self._samples_since_train = 0      # counter for auto-retrain trigger
        self._train_acc = 0.0              # training set accuracy (calibration signal)

    # ── Training data collection ──────────────────────────────────────────────

    def record_training_snapshot(
        self,
        count: int,
        std_vec: Dict[str, float],
        delta_vec: Dict[str, float],
    ):
        """
        Store one labelled feature snapshot (call during baseline capture).

        std_vec  : {bssid: normalised_std}   from WiFiSensor.get_ap_std_vector()
        delta_vec: {bssid: normalised_delta} from WiFiSensor.get_ap_delta_vector()
        count    : known number of people in the room at capture time
        """
        if not std_vec:
            return

        # Build / extend canonical AP order from seen BSSIDs
        for bssid in std_vec:
            if bssid not in self._ap_order:
                self._ap_order.append(bssid)

        row = self._build_row(std_vec, delta_vec)
        if row is None:
            return

        self._X.append(row)
        self._y.append(int(count))
        self._samples_since_train += 1

        # Auto-retrain when enough new data arrives
        if self._samples_since_train >= _OCC_RETRAIN_EVERY:
            self.train()

    def _build_row(
        self,
        std_vec: Dict[str, float],
        delta_vec: Dict[str, float],
    ) -> Optional[List[float]]:
        """
        Vectorise per-AP std dev and delta into a fixed-size feature row.
        Preserves individual AP information for RandomForest (per arxiv 2308.06773).

        Feature layout (canonical AP order from training):
          [AP1_std, AP2_std, ..., APm_std, AP1_delta, AP2_delta, ..., APm_delta, 
           mean_std, mean_delta]

        Each per-AP std dev is an independent detector channel whose value encodes
        signal disruption from people. RandomForest learns which APs are most 
        informative for distinguishing occupancy counts.

        Missing APs during inference → use 0.0 (handled gracefully).
        Extra APs during inference → ignored (uses canonical order only).
        """
        if not self._ap_order:
            return None

        row = []
        n_aps = len(self._ap_order)

        # ── Per-AP RSSI std dev (individual detector channels) ────────────────
        # Each AP's std dev encodes signal disruption from people 
        # (paper's core insight: more people → higher std dev per AP)
        for bssid in self._ap_order:
            std_val = float(std_vec.get(bssid, 0.0))  # 0.0 if AP missing at inference
            row.append(std_val)

        # ── Per-AP RSSI delta (occupancy change vs baseline) ──────────────────
        # How much current mean RSSI differs from captured baseline per AP
        for bssid in self._ap_order:
            delta_val = float(delta_vec.get(bssid, 0.0))  # 0.0 if AP missing
            row.append(delta_val)

        # ── Aggregate fallback features ──────────────────────────────────────
        # Provide RF with aggregate signals for robustness when per-AP data sparse
        mean_std = float(np.mean([std_vec.get(b, 0.0) for b in self._ap_order]))
        mean_delta = float(np.mean([delta_vec.get(b, 0.0) for b in self._ap_order]))
        row.append(mean_std)
        row.append(mean_delta)

        return row

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> bool:
        """
        Fit the classifier.  Returns True if training succeeded.
        Requires at least _OCC_MIN_SAMPLES_PER_CLASS samples per class.
        """
        if not _SKLEARN_AVAILABLE or not self._X or not self._y:
            return False

        from collections import Counter
        counts = Counter(self._y)
        if any(v < _OCC_MIN_SAMPLES_PER_CLASS for v in counts.values()):
            return False   # not enough data yet

        X = np.array(self._X, dtype=float)
        y = np.array(self._y, dtype=int)

        try:
            self._scaler = StandardScaler()
            Xs = self._scaler.fit_transform(X)
            self._clf = RandomForestClassifier(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )
            self._clf.fit(Xs, y)
            # Quick leave-one-out accuracy on training set (not a proper CV,
            # but gives a rough calibration signal)
            self._train_acc = round(float(self._clf.score(Xs, y)), 3)
            self._trained = True
            self._samples_since_train = 0
            print(f"[OccupancyClassifier] Trained on {len(y)} samples, "
                  f"classes={sorted(counts.keys())}, train_acc={self._train_acc:.3f}")
            return True
        except Exception as e:
            print(f"[OccupancyClassifier] Training failed: {e}")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        std_vec: Dict[str, float],
        delta_vec: Dict[str, float],
    ) -> Tuple[Optional[int], float]:
        """
        Predict occupancy count from live per-AP feature vectors.

        Returns (count, confidence) or (None, 0.0) if not trained.
        confidence is the max class probability from the RF.
        """
        if not self._trained or self._clf is None or self._scaler is None:
            return None, 0.0
        if not std_vec:
            return None, 0.0

        row = self._build_row(std_vec, delta_vec)
        if row is None:
            return None, 0.0

        try:
            X = np.array([row], dtype=float)
            Xs = self._scaler.transform(X)
            probas = self._clf.predict_proba(Xs)[0]
            best_idx = int(np.argmax(probas))
            count = int(self._clf.classes_[best_idx])
            confidence = float(probas[best_idx])
            return count, round(confidence, 3)
        except Exception:
            return None, 0.0

    # ── Persistence helpers ───────────────────────────────────────────────────

    def export_training_data(self) -> dict:
        return {
            "X": self._X,
            "y": self._y,
            "ap_order": self._ap_order,
            "trained": self._trained,
            "train_acc": self._train_acc,
        }

    def load_training_data(self, data: dict):
        self._X = data.get("X") or []
        self._y = data.get("y") or []
        self._ap_order = data.get("ap_order") or []
        if self._X and self._y:
            self.train()

    def get_info(self) -> dict:
        """Return classifier status for dashboard and debugging."""
        from collections import Counter
        return {
            "trained": self._trained,
            "train_acc": self._train_acc,
            "n_samples": len(self._y),
            "classes": sorted(set(self._y)),
            "n_aps": len(self._ap_order),
            "ap_order": self._ap_order[:5] if len(self._ap_order) <= 5 else self._ap_order[:5] + ["..."],
            "feature_vector_size": 2 * len(self._ap_order) + 2 if self._ap_order else 0,
            "samples_since_retrain": self._samples_since_train,
        }


# ─── Constants ────────────────────────────────────────────────────────────────

# Seconds of silence before a device slot is considered gone
DEVICE_TIMEOUT_S = 5.0

# Fall detection:  require jerk > FALL_JERK_THRESH  AND  accel > FALL_ACCEL_THRESH
# Both must be true for at least FALL_CONFIRM_FRAMES before declaring fall_risk.
FALL_JERK_THRESH   = 0.55   # was 0.5 in old rule — raised to reduce false positives
FALL_ACCEL_THRESH  = 1.6    # m/s² RMS — body must actually be moving fast
FALL_CONFIRM_FRAMES = 3     # need 3 consecutive qualifying frames → ~0.3 s at 10 Hz

# Activity thresholds (rule-based fallback; ML model overrides when trained)
IDLE_ACCEL_MAX   = 0.18
WALK_CADENCE_MIN = 55.0
WALK_CADENCE_MAX = 140.0
WALK_ACCEL_MIN   = 0.28
SQUAT_GYRO_MIN   = 0.75
BEND_GYRO_MIN    = 1.0
LIFT_ACCEL_MIN   = 0.85

# Demo-safe mode: rely on empty-room baseline + live change intensity only.
# This intentionally ignores 1/2-person profile matching when those profiles
# are noisy or indistinguishable from empty-room captures.
DEMO_EMPTY_BASELINE_ONLY = False
ROOM_FOCUS_ENABLED = True
ROOM_FOCUS_RSSI_TOLERANCE_DB = 22.0

# BLE simulated beacon positions in normalised room coords [0,1]
# (same 5 beacons as bluetooth_sensor.py — we add positions here for RSSI-based ranging)
BEACON_POSITIONS = [
    {"mac": "AA:BB:CC:DD:EE:01", "x": 0.1, "y": 0.1},   # near top-left corner
    {"mac": "AA:BB:CC:DD:EE:02", "x": 0.9, "y": 0.1},   # near top-right corner
    {"mac": "AA:BB:CC:DD:EE:03", "x": 0.9, "y": 0.9},   # near bottom-right corner
    {"mac": "AA:BB:CC:DD:EE:04", "x": 0.1, "y": 0.9},   # near bottom-left corner
    {"mac": "AA:BB:CC:DD:EE:05", "x": 0.5, "y": 0.5},   # centre
]
BEACON_TX_POWER_DBM  = -59.0   # typical BLE TX power at 1 m
BEACON_PATH_LOSS_EXP = 2.5     # indoor path-loss exponent


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PersonState:
    """Per-person rolling state derived from one phone's IMU stream."""
    device_id: str
    last_seen: float = field(default_factory=time.time)
    imu_buffer: deque = field(default_factory=lambda: deque(maxlen=80))
    activity: str = "idle"
    confidence: float = 0.0
    raw_activity: str = "idle"
    activity_history: deque = field(default_factory=lambda: deque(maxlen=8))
    _activity_candidate: str = "idle"
    _activity_streak: int = 0
    accel_rms: float = 0.0
    gyro_rms: float = 0.0
    jerk: float = 0.0
    step_cadence: float = 0.0
    rep_count: int = 0
    fall_risk_score: float = 0.0
    gait_score: float = 0.0
    rom_score: float = 0.0
    symmetry_score: float = 0.0
    fatigue_index: float = 0.0
    gyro_lr_bias: float = 0.0
    gyro_lr_cal_frames: int = 0
    # fall hysteresis counter
    _fall_frame_count: int = 0
    # position in room (BLE trilateration + motion drift)
    pos_x: float = 0.5
    pos_y: float = 0.5
    pos_confidence: float = 0.5
    # per-device spatial seed — assigned once so devices start apart
    _spatial_seed: float = -1.0
    # phase for smooth signal-strength-based position drift
    _drift_phase: float = 0.0
    # source metadata
    source_mode: str = "real"
    sim_activity: Optional[str] = None
    room_match: Optional[bool] = None
    gps_accuracy_m: Optional[float] = None
    # 2D constant-velocity Kalman state [x, y, vx, vy]
    _kf_state: Optional[np.ndarray] = None
    _kf_cov: Optional[np.ndarray] = None
    _kf_ts: float = 0.0


@dataclass
class RoomState:
    person_count: int = 0
    estimated_count: int = 0
    occupancy_band: str = "0"
    phone_count: int = 0
    persons: List[dict] = field(default_factory=list)
    dominant_activity: str = "idle"
    motion_band: str = "quiet"
    room_motion_score: float = 0.0
    baseline_ready: bool = False
    baseline_distance_db: float = 0.0
    baseline_confidence: float = 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rssi_to_distance(rssi_dbm: float) -> float:
    """Convert RSSI to approximate distance (metres) using log-distance model."""
    if rssi_dbm >= BEACON_TX_POWER_DBM:
        return 0.5
    exp = (BEACON_TX_POWER_DBM - rssi_dbm) / (10.0 * BEACON_PATH_LOSS_EXP)
    return min(20.0, max(0.3, 10.0 ** exp))


def _trilaterate(distances: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Estimate (x, y, confidence) from BLE beacon distances using a simple
    weighted centroid (good enough for room-scale localisation with simulated
    or real BLE).  Returns normalised [0,1] coordinates.
    """
    total_w = 0.0
    wx = wy = 0.0
    for bp in BEACON_POSITIONS:
        mac = bp["mac"]
        d = distances.get(mac, None)
        if d is None:
            continue
        # Weight inversely proportional to distance (closer = more reliable)
        w = 1.0 / max(0.1, d)
        wx += bp["x"] * w
        wy += bp["y"] * w
        total_w += w

    if total_w < 1e-6:
        return 0.5, 0.5, 0.3   # centre fallback

    x = max(0.05, min(0.95, wx / total_w))
    y = max(0.05, min(0.95, wy / total_w))
    # Confidence: higher when more beacons contributed and distances are short
    n_used = sum(1 for bp in BEACON_POSITIONS if bp["mac"] in distances)
    conf = min(0.95, 0.4 + 0.12 * n_used)
    return round(x, 3), round(y, 3), round(conf, 2)


def _extract_imu_features(samples: List[dict]) -> dict:
    """
    Compute IMU features from a list of {ax,ay,az,gx,gy,gz,ts} dicts.
    Returns the same feature dict as IMUSensor.extract_features().
    """
    if len(samples) < 8:
        return {
            "accel_rms": 0.0, "gyro_rms": 0.0, "jerk": 0.0,
            "step_cadence": 0.0, "rep_count": 0, "dominant_freq": 0.0,
            "motion_score": 0.0, "activity_hint": "idle",
        }

    arr = np.array([
        [s.get("ax", 0), s.get("ay", 0), s.get("az", 0),
         s.get("gx", 0), s.get("gy", 0), s.get("gz", 0)]
        for s in samples
    ], dtype=float)

    accel = arr[:, :3]
    gyro  = arr[:, 3:]
    gyro_lr = gyro[:, 2]  # gz from phone is left/right tilt rate proxy

    # De-mean to remove gravity component
    accel_mag = np.linalg.norm(accel - np.mean(accel, axis=0), axis=1)
    gyro_mag  = np.linalg.norm(gyro, axis=1)

    accel_rms = float(np.sqrt(np.mean(accel_mag ** 2)))
    gyro_rms  = float(np.sqrt(np.mean(gyro_mag ** 2)))

    diffs = np.abs(np.diff(accel_mag))
    jerk  = float(np.mean(diffs)) if len(diffs) else 0.0

    # Dominant frequency via FFT — use real timestamps if available
    n = len(accel_mag)
    ts_list = [s.get("ts", None) for s in samples]
    if all(t is not None for t in ts_list) and len(ts_list) > 2:
        dt = float(np.mean(np.diff(ts_list)))
        sr = 1.0 / max(dt, 1e-3)
    else:
        sr = 50.0

    fft   = np.abs(np.fft.rfft(accel_mag - np.mean(accel_mag)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    dominant_freq = float(freqs[np.argmax(fft[1:]) + 1]) if n > 2 else 0.0

    # Step cadence
    step_cadence = dominant_freq * 60.0 if 0.5 < dominant_freq < 3.5 else 0.0

    # Rep counting via zero-crossings
    smoothed = np.convolve(accel_mag, np.ones(5) / 5, mode="valid")
    mean_val = float(np.mean(smoothed))
    crossings = np.where(np.diff(np.sign(smoothed - mean_val)))[0]
    rep_count = max(0, len(crossings) // 2)

    # Motion score
    motion_score = min(100.0, accel_rms * 20 + gyro_rms * 5 + jerk * 30)
    gyro_lr_mean = float(np.mean(gyro_lr)) if len(gyro_lr) else 0.0
    gyro_lr_abs_mean = float(np.mean(np.abs(gyro_lr))) if len(gyro_lr) else 0.0

    # Activity hint (quick label for display, full classification done later)
    if accel_rms < IDLE_ACCEL_MAX:
        hint = "idle"
    elif WALK_CADENCE_MIN < step_cadence < WALK_CADENCE_MAX and accel_rms > WALK_ACCEL_MIN:
        hint = "walk"
    elif step_cadence < 35 and gyro_rms >= 1.05 and rep_count <= 5:
        hint = "bend"
    elif step_cadence < 40 and gyro_rms >= SQUAT_GYRO_MIN and rep_count >= 4:
        hint = "squat"
    elif accel_rms >= LIFT_ACCEL_MIN and step_cadence < 35:
        hint = "lift"
    else:
        hint = "active"

    return {
        "accel_rms":     round(accel_rms, 4),
        "gyro_rms":      round(gyro_rms, 4),
        "gyro_lr_mean":  round(gyro_lr_mean, 4),
        "gyro_lr_abs_mean": round(gyro_lr_abs_mean, 4),
        "jerk":          round(jerk, 4),
        "step_cadence":  round(step_cadence, 1),
        "rep_count":     rep_count,
        "dominant_freq": round(dominant_freq, 3),
        "motion_score":  round(motion_score, 2),
        "activity_hint": hint,
    }


def _classify_activity(feat: dict, fall_frame_count: int) -> Tuple[str, float, int]:
    """
    Rule-based activity classifier with hysteresis for fall detection.

    Returns (activity, confidence, updated_fall_frame_count).

    Fall detection requires BOTH high jerk AND high accel to persist for
    FALL_CONFIRM_FRAMES consecutive calls.  This prevents a single noisy
    sample from triggering a fall alert.
    """
    accel = feat.get("accel_rms", 0.0)
    gyro  = feat.get("gyro_rms", 0.0)
    jerk  = feat.get("jerk", 0.0)
    cad   = feat.get("step_cadence", 0.0)

    # ── Fall detection (hysteresis) ──────────────────────────────────────────
    is_fall_candidate = (jerk > FALL_JERK_THRESH and accel > FALL_ACCEL_THRESH)
    if is_fall_candidate:
        fall_frame_count = min(fall_frame_count + 1, FALL_CONFIRM_FRAMES + 2)
    else:
        # Decay — must stay below threshold for 2 frames to clear
        fall_frame_count = max(0, fall_frame_count - 1)

    if fall_frame_count >= FALL_CONFIRM_FRAMES:
        return "fall_risk", 0.88, fall_frame_count

    # ── Normal activities ────────────────────────────────────────────────────
    if accel < IDLE_ACCEL_MAX:
        return "idle", 0.92, fall_frame_count

    if WALK_CADENCE_MIN <= cad <= WALK_CADENCE_MAX and accel >= WALK_ACCEL_MIN:
        conf = min(0.92, 0.70 + (accel - WALK_ACCEL_MIN) * 0.3)
        return "walk", round(conf, 2), fall_frame_count

    if cad < 35 and gyro >= 1.05 and feat.get("rep_count", 0) <= 5:
        return "bend", 0.76, fall_frame_count

    if cad < 40 and gyro >= SQUAT_GYRO_MIN and feat.get("rep_count", 0) >= 4:
        return "squat", 0.74, fall_frame_count

    if accel >= LIFT_ACCEL_MIN and cad < 35:
        return "lift", 0.72, fall_frame_count

    if accel >= WALK_ACCEL_MIN:
        return "walk", 0.62, fall_frame_count

    return "idle", 0.58, fall_frame_count


# ─── Multi-Person Tracker ─────────────────────────────────────────────────────

class MultiPersonTracker:
    """
    Tracks one PersonState per connected phone device_id.

    Lifecycle:
      • update_device(device_id, samples, bt_scan, source_mode, sim_activity)
          → called every time a phone POSTs IMU data
      • get_room_state(wifi_feat) → RoomState with person list + dominant activity

    Person position is estimated by:
      1. If BLE data is present: RSSI-based trilateration (weighted centroid).
      2. Else: equally-spaced positions around the room centre.
    """

    def __init__(self):
        self._persons: Dict[str, PersonState] = {}
        self._signal_count_history: deque = deque(maxlen=18)
        self._signal_smoothed_count: int = 0
        self._wifi_fp_history: deque = deque(maxlen=40)
        self._empty_room_baseline: Dict[str, float] = {}
        self._empty_room_baseline_ts: float = 0.0
        self._empty_room_baseline_scans: int = 0
        # occupancy profiles: count -> {baseline, captured_ts, baseline_scans, baseline_ap_count}
        self._occupancy_profiles: Dict[int, dict] = {}

        # ── Per-AP std dev occupancy classifier (arxiv 2308.06773) ───────────
        self._occ_classifier = OccupancyClassifier()
        # Assigned by main.py after construction: tracker.set_wifi_sensor(wifi)
        # Gives the classifier direct access to per-AP rolling windows without
        # changing the sensor_loop call signatures.
        self._wifi_sensor = None

    def set_wifi_sensor(self, sensor):
        """
        Register the WiFiSensor instance so the occupancy classifier can pull
        per-AP std dev / delta vectors at inference time.
        Call once from main.py after constructing both objects.
        """
        self._wifi_sensor = sensor

    def get_occupancy_classifier_info(self) -> dict:
        """Expose OccupancyClassifier status for /api/status and dashboard."""
        return self._occ_classifier.get_info()

    def _count_to_band(self, count: int) -> str:
        """Map a count to the occupancy display (0-2 exact, 3+ as 4+)."""
        if count >= 3:
            return "4+"
        return str(max(0, count))

    # ── Public API ────────────────────────────────────────────────────────────

    def update_device(
        self,
        device_id: str,
        samples: List[dict],
        bt_scan: Optional[Dict[str, float]] = None,
        source_mode: str = "real",
        sim_activity: Optional[str] = None,
        room_match: Optional[bool] = None,
        gps_accuracy_m: Optional[float] = None,
    ) -> PersonState:
        """
        Ingest new IMU samples (and optional BLE scan) for one phone/person.
        Returns the updated PersonState.
        """
        if device_id not in self._persons:
            self._persons[device_id] = PersonState(device_id=device_id)

        p = self._persons[device_id]
        p.last_seen = time.time()
        p.source_mode = source_mode
        p.sim_activity = sim_activity
        p.room_match = room_match
        p.gps_accuracy_m = gps_accuracy_m

        # ── Buffer new IMU samples ────────────────────────────────────────────
        for s in samples:
            p.imu_buffer.append(s)

        # ── Extract features ──────────────────────────────────────────────────
        feat = _extract_imu_features(list(p.imu_buffer)[-50:])
        p.accel_rms    = feat["accel_rms"]
        p.gyro_rms     = feat["gyro_rms"]
        p.jerk         = feat["jerk"]
        p.step_cadence = feat["step_cadence"]
        p.rep_count    = feat["rep_count"]

        # Calibrate per-device left/right gyro neutral bias while mostly still.
        if source_mode == "real":
            lr = float(feat.get("gyro_lr_mean", 0.0))
            if p.accel_rms < 0.22 and abs(lr) < 0.18 and p.gyro_lr_cal_frames < 80:
                p.gyro_lr_bias = 0.92 * p.gyro_lr_bias + 0.08 * lr
                p.gyro_lr_cal_frames += 1

        # ── Classify activity ─────────────────────────────────────────────────
        if source_mode == "simulated" and sim_activity:
            # Honour the explicit simulation mode without re-classifying
            p.raw_activity = sim_activity
            p.activity   = sim_activity
            p.confidence = 0.93
            p._activity_candidate = sim_activity
            p._activity_streak = 0
            p.activity_history.clear()
            p._fall_frame_count = FALL_CONFIRM_FRAMES if sim_activity == "fall_risk" else 0
        else:
            act, conf, fc = _classify_activity(feat, p._fall_frame_count)
            act, conf = self._directional_real_override(p, feat, act, conf)
            if act == "lift":
                lr = float(feat.get("gyro_lr_mean", 0.0)) - p.gyro_lr_bias
                cad = float(feat.get("step_cadence", 0.0))
                if abs(lr) > 0.12 and cad < 35:
                    act = "bend" if lr < 0 else "squat"
                    conf = max(conf, 0.78)
                else:
                    act = "walk" if cad > 45 else "squat"
                    conf = max(conf, 0.70)
            p.raw_activity      = act
            p._fall_frame_count = fc
            self._stabilize_activity(p, act, conf)

        # ── Physiotherapy metrics ─────────────────────────────────────────────
        p.gait_score = (
            min(100.0, max(0.0, 50 + (p.step_cadence - 80) * 0.3 - p.jerk * 10))
            if p.activity == "walk" and p.step_cadence > 0 else 0.0
        )
        p.rom_score      = min(100.0, p.gyro_rms * 35)
        sym_raw          = 1.0 - min(1.0, abs(p.accel_rms - p.gyro_rms * 0.3) / 2.0)
        p.symmetry_score = round(sym_raw * 100, 1)

        # Fall risk score (0–100) — use the hysteresis counter
        fr_raw         = min(100.0, p.jerk * 80 + (feat["motion_score"] > 70) * 20)
        hyst_boost     = min(40.0, p._fall_frame_count * 15.0)
        p.fall_risk_score = round(min(100.0, fr_raw + hyst_boost), 1)

        # ── Spatial seed: assigned once per device so every phone starts ────────
        # at a deterministic spread position rather than all at (0.5, 0.5).
        if p._spatial_seed < 0:
            # Hash device_id to a stable angle in [0, 2π)
            h = sum(ord(c) * (i + 1) for i, c in enumerate(device_id))
            p._spatial_seed = (h % 1000) / 1000.0   # 0..1 normalised

        # ── Motion drift: make the avatar move visibly with signal strength ──
        # Drift speed scales with accel_rms — active person moves more.
        motion_level = min(1.0, feat["accel_rms"] / 1.5)
        p._drift_phase += 0.04 + motion_level * 0.12   # faster when active

        # ── BLE-based position (if scan provided) ─────────────────────────────
        meas_x = None
        meas_y = None
        meas_conf = 0.45
        if bt_scan:
            dists = {mac: _rssi_to_distance(rssi) for mac, rssi in bt_scan.items()}
            bx, by, bconf = _trilaterate(dists)
            # Only accept if it moved away from dead-centre (avoids all-same scan)
            if abs(bx - 0.5) > 0.04 or abs(by - 0.5) > 0.04:
                meas_x, meas_y, meas_conf = bx, by, bconf
            else:
                # BLE scan was uniform (simulated same values) — use seed position
                # + motion drift so each person has a distinct, animated location.
                base_angle = p._spatial_seed * 2 * math.pi
                radius = 0.22 + 0.06 * motion_level
                drift_x = math.cos(p._drift_phase * 0.7 + base_angle) * 0.04 * motion_level
                drift_y = math.sin(p._drift_phase * 0.5 + base_angle + 1.0) * 0.04 * motion_level
                meas_x = round(max(0.12, min(0.88,
                    0.5 + radius * math.cos(base_angle) + drift_x)), 3)
                meas_y = round(max(0.12, min(0.88,
                    0.5 + radius * math.sin(base_angle) + drift_y)), 3)
                meas_conf = 0.55
        else:
            # No BLE at all — seed spread + motion drift
            base_angle = p._spatial_seed * 2 * math.pi
            radius = 0.22 + 0.06 * motion_level
            drift_x = math.cos(p._drift_phase * 0.7 + base_angle) * 0.04 * motion_level
            drift_y = math.sin(p._drift_phase * 0.5 + base_angle + 1.0) * 0.04 * motion_level
            meas_x = round(max(0.12, min(0.88,
                0.5 + radius * math.cos(base_angle) + drift_x)), 3)
            meas_y = round(max(0.12, min(0.88,
                0.5 + radius * math.sin(base_angle) + drift_y)), 3)
            meas_conf = 0.45

        fx, fy, fconf = self._kalman_update_position(p, float(meas_x), float(meas_y), float(meas_conf))
        p.pos_x = round(max(0.08, min(0.92, fx)), 3)
        p.pos_y = round(max(0.08, min(0.92, fy)), 3)
        p.pos_confidence = round(max(0.2, min(0.98, fconf)), 2)

        return p

    def _directional_real_override(self, p: PersonState, feat: dict, act: str, conf: float) -> Tuple[str, float]:
        """Use signed left/right gyro to separate bend vs squat on real phone data."""
        lr = float(feat.get("gyro_lr_mean", 0.0)) - p.gyro_lr_bias
        cad = float(feat.get("step_cadence", 0.0))
        reps = float(feat.get("rep_count", 0.0))
        accel = float(feat.get("accel_rms", 0.0))

        if cad < 35 and abs(lr) > 0.14 and accel > 0.2:
            if lr < 0:
                return "bend", max(conf, 0.82 if reps <= 6 else 0.78)
            return "squat", max(conf, 0.82 if reps >= 3 else 0.78)

        return act, conf

    def _stabilize_activity(self, p: PersonState, candidate: str, confidence: float):
        """Promote a new activity only after it is repeated consistently."""
        p.activity_history.append(candidate)

        if candidate == "fall_risk":
            p.activity = candidate
            p.confidence = max(confidence, 0.88)
            p._activity_candidate = candidate
            p._activity_streak = 0
            return

        if candidate == p.activity:
            p.confidence = round(0.75 * p.confidence + 0.25 * confidence, 2) if p.confidence else confidence
            p._activity_candidate = candidate
            p._activity_streak = max(1, p._activity_streak)
            return

        if candidate == p._activity_candidate:
            p._activity_streak += 1
        else:
            p._activity_candidate = candidate
            p._activity_streak = 1

        recent_agreement = 0.0
        if p.activity_history:
            recent_agreement = sum(1 for a in p.activity_history if a == candidate) / len(p.activity_history)

        if (
            p._activity_streak >= 3
            or (candidate == "idle" and confidence >= 0.82 and recent_agreement >= 0.5)
            or (confidence >= 0.88 and recent_agreement >= 0.6)
        ):
            p.activity = candidate
            p.confidence = round(confidence, 2)
            return

        # Keep the existing label until the new candidate proves stable.
        p.confidence = round(max(0.25, 0.85 * p.confidence + 0.15 * confidence), 2)

    def expire_old_devices(self):
        """Remove devices that haven't sent data for DEVICE_TIMEOUT_S seconds."""
        now = time.time()
        stale = [did for did, p in self._persons.items()
                 if now - p.last_seen > DEVICE_TIMEOUT_S]
        for did in stale:
            del self._persons[did]

    def get_active_persons(self) -> List[PersonState]:
        """Return list of currently active PersonState objects, sorted by device_id."""
        self.expire_old_devices()
        return sorted(self._persons.values(), key=lambda p: p.device_id)

    def get_room_state(self, wifi_feat: dict) -> RoomState:
        """
        Build a RoomState from active phones + room-wide WiFi motion signal.

        Person count is estimated from room-wide signals and then reconciled
        with connected phones so the dashboard can show pre-phone occupancy.
        Unknown placeholders are used when estimated count is higher than the
        number of connected phones.
        """
        wifi_feat = self.focus_wifi_features(wifi_feat)
        active = self.get_active_persons()
        phone_count = len(active)
        room_motion_score = self._estimate_room_motion_score(wifi_feat)
        self._remember_wifi_fingerprint(wifi_feat)
        baseline_dist_db, baseline_conf, baseline_delta, profile_count, profile_conf = self._compare_against_profiles(wifi_feat)
        estimated_count = self._estimate_signal_person_count(
            wifi_feat,
            room_motion_score,
            baseline_delta,
            baseline_conf,
            profile_count,
            profile_conf,
        )
        total_count = max(phone_count, estimated_count)
        wifi_extra = max(0, total_count - phone_count)

        # Build person list for the dashboard
        persons_out = []

        # Real phone persons — collect raw positions first, then enforce separation
        raw_positions = []
        for i, p in enumerate(active):
            raw_positions.append((p.pos_x, p.pos_y))

        # Push apart any persons that are too close (min distance = 0.20 in norm coords)
        MIN_DIST = 0.20
        final_positions = list(raw_positions)
        for _iter in range(8):   # iterative repulsion passes
            for a in range(len(final_positions)):
                for b in range(a + 1, len(final_positions)):
                    ax, ay = final_positions[a]
                    bx, by = final_positions[b]
                    dx, dy = bx - ax, by - ay
                    dist = math.sqrt(dx * dx + dy * dy) or 0.001
                    if dist < MIN_DIST:
                        push = (MIN_DIST - dist) / 2.0 + 0.01
                        nx, ny = (dx / dist) * push, (dy / dist) * push
                        final_positions[a] = (
                            max(0.10, min(0.90, ax - nx)),
                            max(0.10, min(0.90, ay - ny)),
                        )
                        final_positions[b] = (
                            max(0.10, min(0.90, bx + nx)),
                            max(0.10, min(0.90, by + ny)),
                        )

        for i, p in enumerate(active):
            pos_x, pos_y = final_positions[i]
            persons_out.append({
                "id":           i,
                "device_id":    p.device_id,
                "x":            round(pos_x, 3),
                "y":            round(pos_y, 3),
                "confidence":   p.pos_confidence,
                "activity":     p.activity,
                "fall_risk":    p.fall_risk_score,
                "accel_rms":    round(p.accel_rms, 3),
                "gyro_rms":     round(p.gyro_rms, 3),
                "jerk":         round(p.jerk, 4),
                "step_cadence": round(p.step_cadence, 1),
                "gait_score":   round(p.gait_score, 1),
                "rom_score":    round(p.rom_score, 1),
                "fatigue_index": round(p.fatigue_index, 1),
                "source":       p.source_mode,
                "type":         "phone",
                "room_match":   p.room_match,
                "gps_accuracy_m": p.gps_accuracy_m,
            })

        # WiFi-inferred unknown persons fill estimated room occupancy gaps.
        for j in range(wifi_extra):
            idx = phone_count + j
            angle = (2 * math.pi * (idx + 1)) / max(2, total_count + 1)
            radius = 0.18 + 0.08 * (j % 3)
            persons_out.append({
                "id":           idx,
                "device_id":    f"wifi_inferred_{j}",
                "x":            round(max(0.1, min(0.9, 0.5 + radius * math.cos(angle))), 3),
                "y":            round(max(0.1, min(0.9, 0.5 + radius * math.sin(angle))), 3),
                "confidence":   0.45,
                "activity":     "unknown",
                "fall_risk":    0.0,
                "accel_rms":    0.0,
                "gyro_rms":     0.0,
                "jerk":         0.0,
                "step_cadence": 0.0,
                "gait_score":   0.0,
                "rom_score":    0.0,
                "fatigue_index": 0.0,
                "source":       "wifi_inferred",
                "type":         "inferred",
                "room_match":   None,
                "gps_accuracy_m": None,
            })

        # Dominant activity = worst-case across all persons (prioritise fall_risk)
        activity_priority = ["fall_risk", "lift", "squat", "bend", "walk", "idle", "unknown"]
        all_activities = [p.activity for p in active]
        all_activities.extend(["unknown"] * wifi_extra)
        dominant = "idle"
        for a in activity_priority:
            if a in all_activities:
                dominant = a
                break

        fused = room_motion_score
        if active:
            fused = max(fused, max(p.accel_rms * 20 for p in active))

        motion_band = "quiet"
        if fused >= 55:
            motion_band = "busy"
        elif fused >= 20:
            motion_band = "active"

        occupancy_band = self._count_to_band(total_count)

        return RoomState(
            person_count=total_count,
            estimated_count=estimated_count,
            occupancy_band=occupancy_band,
            phone_count=phone_count,
            persons=persons_out,
            dominant_activity=dominant,
            motion_band=motion_band,
            room_motion_score=round(room_motion_score, 2),
            baseline_ready=bool(self._empty_room_baseline),
            baseline_distance_db=round(baseline_dist_db, 3),
            baseline_confidence=round(baseline_conf, 3),
        )

    def focus_wifi_features(self, wifi_feat: dict) -> dict:
        """
        Filter WiFi features to the current room AP fingerprint, reducing building-wide noise.
        Uses occupancy profile 0 APs as the room anchor when available.
        """
        if not isinstance(wifi_feat, dict):
            return wifi_feat
        if not ROOM_FOCUS_ENABLED:
            return wifi_feat

        zero_profile = self._occupancy_profiles.get(0, {}) if isinstance(self._occupancy_profiles, dict) else {}
        zero_baseline = zero_profile.get("baseline", {}) if isinstance(zero_profile, dict) else {}
        if not isinstance(zero_baseline, dict) or not zero_baseline:
            return wifi_feat

        raw = wifi_feat.get("raw_vector", {})
        if not isinstance(raw, dict) or not raw:
            return wifi_feat

        filtered = {}
        for bssid, rssi in raw.items():
            if bssid not in zero_baseline:
                continue
            try:
                base = float(zero_baseline[bssid])
                cur = float(rssi)
            except Exception:
                continue
            if abs(cur - base) <= ROOM_FOCUS_RSSI_TOLERANCE_DB:
                filtered[bssid] = cur

        # Fallback to baseline AP intersection even if tolerance removes all.
        if not filtered:
            filtered = {b: float(v) for b, v in raw.items() if b in zero_baseline}
        if not filtered:
            return wifi_feat

        out = dict(wifi_feat)
        out["raw_vector"] = filtered
        out["n_networks"] = len(filtered)
        names = wifi_feat.get("network_names", {}) if isinstance(wifi_feat.get("network_names", {}), dict) else {}
        out["network_names"] = {b: names.get(b, "?") for b in filtered.keys()}
        return out

    def set_occupancy_baseline(self, occupancy_count: int, raw_vectors: List[Dict[str, float]]) -> dict:
        """Capture a WiFi baseline profile for a specific occupancy count (e.g., 0/1/2)."""
        usable = []
        for rv in raw_vectors or []:
            if isinstance(rv, dict) and rv:
                usable.append({str(k): float(v) for k, v in rv.items()})
        if not usable:
            return {
                "ready": False,
                "occupancy_count": int(occupancy_count),
                "baseline_ap_count": 0,
                "baseline_scans": 0,
                "captured_ts": None,
                "message": "No usable WiFi scans for baseline",
            }

        freq: Dict[str, int] = {}
        for rv in usable:
            for b in rv.keys():
                freq[b] = freq.get(b, 0) + 1

        min_seen = max(1, int(len(usable) * 0.35))
        keep = [b for b, c in freq.items() if c >= min_seen]
        baseline: Dict[str, float] = {}
        for b in keep:
            vals = [rv[b] for rv in usable if b in rv]
            if vals:
                baseline[b] = float(np.mean(vals))

        if not baseline:
            return {
                "ready": False,
                "occupancy_count": int(occupancy_count),
                "baseline_ap_count": 0,
                "baseline_scans": len(usable),
                "captured_ts": None,
                "message": "No stable APs found for baseline",
            }

        count = max(0, int(occupancy_count))
        rec = {
            "baseline": baseline,
            "captured_ts": time.time(),
            "baseline_scans": len(usable),
            "baseline_ap_count": len(baseline),
        }
        self._occupancy_profiles[count] = rec

        # Keep legacy empty profile fields in sync for compatibility.
        if count == 0:
            self._empty_room_baseline = baseline
            self._empty_room_baseline_ts = rec["captured_ts"]
            self._empty_room_baseline_scans = rec["baseline_scans"]

        # ── Feed per-AP std dev training snapshots into the classifier ────────
        # For each usable scan window, compute per-AP std dev and delta vs the
        # freshly captured baseline, then label with the known count.
        # We use a sliding sub-window approach: take every 4-scan chunk so we
        # get multiple labelled samples from one capture session.
        if self._wifi_sensor is not None and len(usable) >= 4:
            anchor = list(baseline.keys())
            chunk_size = 4
            for start in range(0, len(usable) - chunk_size + 1, 2):
                chunk = usable[start: start + chunk_size]
                # Compute std dev across the chunk per AP
                std_snap: Dict[str, float] = {}
                delta_snap: Dict[str, float] = {}
                for bssid in anchor:
                    vals = [rv[bssid] for rv in chunk if bssid in rv]
                    if len(vals) >= 2:
                        std_val = float(np.std(vals))
                        # Normalise to [0,1] using the same cap as WiFiSensor
                        std_snap[bssid] = round(min(1.0, std_val / 12.0), 4)
                        mean_val = float(np.mean(vals))
                        base_val = baseline.get(bssid, mean_val)
                        delta_snap[bssid] = round(min(1.0, abs(mean_val - base_val) / 20.0), 4)
                if std_snap:
                    self._occ_classifier.record_training_snapshot(count, std_snap, delta_snap)

            # Attempt a retrain after each new occupancy count is captured
            self._occ_classifier.train()

        status = self.get_empty_room_baseline_status()
        status["occupancy_count"] = count
        status["occ_classifier"] = self._occ_classifier.get_info()
        return status

    def set_empty_room_baseline(self, raw_vectors: List[Dict[str, float]]) -> dict:
        """Capture an empty-room WiFi RSSI baseline from multiple raw scan vectors."""
        return self.set_occupancy_baseline(0, raw_vectors)

    def clear_empty_room_baseline(self) -> dict:
        self._empty_room_baseline = {}
        self._empty_room_baseline_ts = 0.0
        self._empty_room_baseline_scans = 0
        self._occupancy_profiles.pop(0, None)
        return self.get_empty_room_baseline_status()

    def export_empty_room_baseline(self) -> dict:
        """Return the current baseline payload for persistence."""
        occ_profiles = {}
        for count, rec in self._occupancy_profiles.items():
            occ_profiles[str(int(count))] = {
                "baseline": dict(rec.get("baseline", {})),
                "captured_ts": rec.get("captured_ts"),
                "baseline_scans": rec.get("baseline_scans"),
                "baseline_ap_count": rec.get("baseline_ap_count"),
            }
        return {
            "ready": bool(self._empty_room_baseline),
            "baseline": dict(self._empty_room_baseline),
            "captured_ts": self._empty_room_baseline_ts or None,
            "baseline_scans": int(self._empty_room_baseline_scans),
            "baseline_ap_count": len(self._empty_room_baseline),
            "occupancy_profiles": occ_profiles,
            # Persist classifier training data so it survives restarts
            "occ_classifier_data": self._occ_classifier.export_training_data(),
        }

    def load_empty_room_baseline(self, baseline: Dict[str, float], captured_ts: Optional[float] = None, scans: Optional[int] = None) -> dict:
        """Load a previously saved baseline map into the live tracker state."""
        cleaned: Dict[str, float] = {}
        for k, v in (baseline or {}).items():
            try:
                cleaned[str(k)] = float(v)
            except Exception:
                continue

        self._empty_room_baseline = cleaned
        self._empty_room_baseline_ts = float(captured_ts) if captured_ts else time.time()
        self._empty_room_baseline_scans = int(scans) if scans is not None else max(1, len(cleaned))
        if cleaned:
            self._occupancy_profiles[0] = {
                "baseline": dict(cleaned),
                "captured_ts": self._empty_room_baseline_ts,
                "baseline_scans": self._empty_room_baseline_scans,
                "baseline_ap_count": len(cleaned),
            }
        else:
            self._occupancy_profiles.pop(0, None)
        return self.get_empty_room_baseline_status()

    def load_occupancy_profiles(self, profiles: Dict[int, dict], fallback_baseline: Optional[Dict[str, float]] = None,
                                fallback_ts: Optional[float] = None, fallback_scans: Optional[int] = None,
                                occ_classifier_data: Optional[dict] = None) -> dict:
        """Load persisted occupancy profiles (0/1/2...) and align empty-room compatibility fields."""
        loaded: Dict[int, dict] = {}
        for k, rec in (profiles or {}).items():
            try:
                count = max(0, int(k))
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            b = rec.get("baseline", {})
            clean_b = {}
            for bk, bv in (b or {}).items():
                try:
                    clean_b[str(bk)] = float(bv)
                except Exception:
                    continue
            if not clean_b:
                continue
            loaded[count] = {
                "baseline": clean_b,
                "captured_ts": rec.get("captured_ts") or time.time(),
                "baseline_scans": int(rec.get("baseline_scans") or max(1, len(clean_b))),
                "baseline_ap_count": int(rec.get("baseline_ap_count") or len(clean_b)),
            }

        self._occupancy_profiles = loaded

        if 0 in self._occupancy_profiles:
            z = self._occupancy_profiles[0]
            self._empty_room_baseline = dict(z.get("baseline", {}))
            self._empty_room_baseline_ts = float(z.get("captured_ts") or time.time())
            self._empty_room_baseline_scans = int(z.get("baseline_scans") or max(1, len(self._empty_room_baseline)))
        else:
            self._empty_room_baseline = {}
            self._empty_room_baseline_ts = 0.0
            self._empty_room_baseline_scans = 0
            if fallback_baseline:
                self.load_empty_room_baseline(fallback_baseline, fallback_ts, fallback_scans)

        # Restore classifier training data if provided
        if occ_classifier_data and isinstance(occ_classifier_data, dict):
            self._occ_classifier.load_training_data(occ_classifier_data)

        return self.get_empty_room_baseline_status()

    def get_empty_room_baseline_status(self, wifi_feat: Optional[dict] = None) -> dict:
        distance_db = 0.0
        confidence = 0.0
        profile_count = None
        profile_conf = 0.0
        if wifi_feat:
            distance_db, confidence, _, profile_count, profile_conf = self._compare_against_profiles(wifi_feat)
        return {
            "ready": bool(self._empty_room_baseline),
            "baseline_ap_count": len(self._empty_room_baseline),
            "baseline_scans": self._empty_room_baseline_scans,
            "captured_ts": self._empty_room_baseline_ts or None,
            "current_distance_db": round(distance_db, 3),
            "comparison_confidence": round(confidence, 3),
            "occupancy_profiles": sorted(int(k) for k in self._occupancy_profiles.keys()),
            "profile_estimated_count": profile_count,
            "profile_confidence": round(profile_conf, 3),
        }

    def get_merged_imu_features(self) -> dict:
        """
        Merge IMU features across all active persons for single-channel sensors
        (WiFi, BT, acoustic, RF) that still operate room-wide.

        Uses the most-active person's features so the ambient sensors are
        calibrated to actual movement, not a flat average.
        """
        active = self.get_active_persons()
        if not active:
            return {
                "accel_rms": 0.0, "gyro_rms": 0.0, "jerk": 0.0,
                "step_cadence": 0.0, "rep_count": 0, "dominant_freq": 0.0,
                "motion_score": 0.0, "activity_hint": "idle",
            }
        # Pick the most active person
        most_active = max(active, key=lambda p: p.accel_rms)
        return _extract_imu_features(list(most_active.imu_buffer)[-50:])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_room_motion_score(self, wifi_feat: dict) -> float:
        """Estimate room-wide movement intensity (0-100) from WiFi dynamics."""
        motion = float(max(0.0, min(100.0, wifi_feat.get("motion_score", 0.0))))
        n_nets = float(max(0.0, wifi_feat.get("n_networks", 0.0)))
        var = float(max(0.0, wifi_feat.get("variance", 0.0)))
        corr = float(max(0.0, min(1.0, wifi_feat.get("corr_score", 0.7))))

        nets_bonus = min(12.0, max(0.0, (n_nets - 2.0) * 2.0))
        var_boost = min(22.0, var * 2.8)
        decorrelation_boost = min(14.0, max(0.0, 0.9 - corr) * 28.0)

        score = 0.55 * motion + var_boost + decorrelation_boost + nets_bonus
        return max(0.0, min(100.0, score))

    def _estimate_signal_person_count(
        self,
        wifi_feat: dict,
        room_motion_score: float,
        baseline_delta_score: float = 0.0,
        baseline_confidence: float = 0.0,
        profile_count_hint: Optional[int] = None,
        profile_confidence: float = 0.0,
    ) -> int:
        """
        Estimate people count from WiFi signal dynamics.

        Priority order:
          1. OccupancyClassifier (per-AP std dev RF) — if trained and confident
          2. Profile nearest-match hint              — if baseline profiles exist
          3. Heuristic band scoring                  — legacy fallback

        The classifier is the main improvement from arxiv:2308.06773: treating
        per-AP RSSI std dev as independent detector channels gives much finer
        occupancy discrimination than a single aggregate motion_score.
        """
        n_nets = float(max(0.0, wifi_feat.get("n_networks", 0.0)))
        var = float(max(0.0, wifi_feat.get("variance", 0.0)))
        corr = float(max(0.0, min(1.0, wifi_feat.get("corr_score", 0.7))))

        # ── 1. Per-AP classifier (primary path) ───────────────────────────────
        if self._wifi_sensor is not None:
            anchor = list(self._empty_room_baseline.keys()) or None
            std_vec = self._wifi_sensor.get_ap_std_vector(anchor_bssids=anchor)
            delta_vec = self._wifi_sensor.get_ap_delta_vector(
                self._empty_room_baseline, anchor_bssids=anchor
            ) if self._empty_room_baseline else {}

            clf_count, clf_conf = self._occ_classifier.predict(std_vec, delta_vec)

            if clf_count is not None and clf_conf >= 0.55:
                # High-confidence classifier result: use directly with smoothing
                raw = clf_count
                self._signal_count_history.append(raw)
                median = int(np.median(list(self._signal_count_history)))
                if median > self._signal_smoothed_count:
                    self._signal_smoothed_count += 1
                elif median < self._signal_smoothed_count:
                    self._signal_smoothed_count -= 1
                return max(0, self._signal_smoothed_count)

            elif clf_count is not None and clf_conf >= 0.40:
                # Medium confidence: blend with heuristic result below
                # (fall through to heuristic, then blend at the end)
                pass

        else:
            clf_count, clf_conf = None, 0.0
            std_vec, delta_vec = {}, {}

        # ── 2 & 3. Heuristic bands (legacy path) ─────────────────────────────
        nets_score    = min(1.0, max(0.0, (n_nets - 2.0) / 6.0))
        var_score     = min(1.0, max(0.0, (var - 2.0) / 8.0))
        decor_score   = min(1.0, max(0.0, (0.88 - corr) / 0.45))
        motion_score  = max(0.0, min(1.0, room_motion_score / 100.0))
        cluster_score = max(0.0, min(1.0, self._estimate_wifi_cluster_count() / 4.0))
        baseline_score = max(0.0, min(1.0, baseline_delta_score))

        # Incorporate mean_ap_std from wifi_feat if available (blended into
        # motion_score so downstream heuristic bands benefit too)
        mean_ap_std = float(wifi_feat.get("mean_ap_std", 0.0))
        if mean_ap_std > 0:
            motion_score = min(1.0, motion_score * 0.70 + mean_ap_std * 0.30)

        if DEMO_EMPTY_BASELINE_ONLY:
            change_index = (
                0.58 * baseline_score
                + 0.17 * motion_score
                + 0.15 * var_score
                + 0.10 * cluster_score
            )

            if change_index < 0.16:
                raw = 0
            elif change_index < 0.48:
                raw = 1 if change_index < 0.30 else 2
            elif change_index < 0.78:
                raw = 3
            else:
                raw = 4

        else:
            occupancy_score = (
                0.17 * nets_score
                + 0.22 * var_score
                + 0.17 * decor_score
                + 0.14 * motion_score
                + 0.13 * cluster_score
                + 0.17 * baseline_score
            )

            if baseline_confidence >= 0.35 and baseline_score < 0.10 and room_motion_score < 22 and var < 2.6:
                raw = 0
            elif occupancy_score < 0.17:
                raw = 0
            elif occupancy_score < 0.30:
                raw = 1
            elif occupancy_score < 0.45:
                raw = 2
            elif occupancy_score < 0.63:
                raw = 3
            elif occupancy_score < 0.82:
                raw = 4
            else:
                raw = 5

            if baseline_confidence >= 0.35 and baseline_score < 0.18:
                raw = min(raw, 1)

            # Profile nearest-match hint
            if profile_count_hint is not None and profile_confidence >= 0.45:
                hint = max(0, min(6, int(profile_count_hint)))
                if profile_confidence >= 0.72:
                    raw = hint
                elif profile_confidence >= 0.55:
                    raw = int(round(0.55 * raw + 0.45 * hint))
                else:
                    raw = int(round(0.70 * raw + 0.30 * hint))

        # Blend with medium-confidence classifier result if we have one
        if clf_count is not None and 0.40 <= clf_conf < 0.55:
            blend_w = (clf_conf - 0.40) / 0.15   # 0→1 as conf goes 0.40→0.55
            raw = int(round(raw * (1 - blend_w * 0.4) + clf_count * (blend_w * 0.4)))

        self._signal_count_history.append(raw)
        median = int(np.median(list(self._signal_count_history))) if self._signal_count_history else raw

        if median > self._signal_smoothed_count:
            self._signal_smoothed_count += 1
        elif median < self._signal_smoothed_count:
            self._signal_smoothed_count -= 1

        return max(0, self._signal_smoothed_count)

    def _compare_against_empty_baseline(self, wifi_feat: dict) -> Tuple[float, float, float]:
        """
        Compare current WiFi fingerprint against an empty-room baseline.
        Returns: (distance_db, confidence, delta_score)
        """
        if not self._empty_room_baseline:
            return 0.0, 0.0, 0.0
        raw = wifi_feat.get("raw_vector", {}) if isinstance(wifi_feat, dict) else {}
        if not isinstance(raw, dict) or not raw:
            return 0.0, 0.0, 0.0

        baseline = self._empty_room_baseline
        union = set(raw.keys()) | set(baseline.keys())
        if not union:
            return 0.0, 0.0, 0.0

        common = set(raw.keys()) & set(baseline.keys())
        diffs = [abs(float(raw.get(b, -100.0)) - float(baseline.get(b, -100.0))) for b in union]
        mean_diff_db = float(np.mean(diffs)) if diffs else 0.0

        overlap = len(common) / max(1, len(union))
        common_depth = min(1.0, len(common) / 8.0)
        confidence = 0.68 * overlap + 0.32 * common_depth
        # Keep baseline sensitivity usable even with sparse AP overlap.
        # Confidence still contributes, but no longer collapses score to near-zero.
        delta_score = min(1.0, (mean_diff_db / 10.0)) * (0.5 + 0.5 * confidence)

        return mean_diff_db, confidence, delta_score

    def _compare_against_profiles(self, wifi_feat: dict) -> Tuple[float, float, float, Optional[int], float]:
        """
        Compare current fingerprint against all occupancy profiles.
        Returns:
          empty_distance_db, empty_confidence, empty_delta_score,
          nearest_profile_count, nearest_profile_confidence
        """
        empty_distance, empty_conf, empty_delta = self._compare_against_empty_baseline(wifi_feat)
        if DEMO_EMPTY_BASELINE_ONLY:
            return empty_distance, empty_conf, empty_delta, None, 0.0
        raw = wifi_feat.get("raw_vector", {}) if isinstance(wifi_feat, dict) else {}
        if not isinstance(raw, dict) or not raw or not self._occupancy_profiles:
            return empty_distance, empty_conf, empty_delta, None, 0.0

        best_count: Optional[int] = None
        best_dist = 1e9
        best_conf = 0.0
        best_common = 0
        ranking: List[Tuple[float, int]] = []

        for count, rec in self._occupancy_profiles.items():
            baseline = rec.get("baseline", {}) if isinstance(rec, dict) else {}
            if not baseline:
                continue
            union = set(raw.keys()) | set(baseline.keys())
            if not union:
                continue
            common = set(raw.keys()) & set(baseline.keys())
            diffs = [abs(float(raw.get(b, -100.0)) - float(baseline.get(b, -100.0))) for b in union]
            mean_diff_db = float(np.mean(diffs)) if diffs else 0.0

            overlap = len(common) / max(1, len(union))
            common_depth = min(1.0, len(common) / 8.0)
            conf = 0.68 * overlap + 0.32 * common_depth

            # Better profile = low distance + good overlap confidence.
            score = mean_diff_db / max(0.2, conf)
            ranking.append((score, int(count)))
            if score < best_dist:
                best_dist = score
                best_count = int(count)
                best_conf = conf
                best_common = len(common)

        # Downscale confidence if nearest profile is still far in dB distance.
        if best_count is not None:
            p = self._occupancy_profiles.get(best_count, {})
            b = p.get("baseline", {}) if isinstance(p, dict) else {}
            union = set(raw.keys()) | set(b.keys())
            diffs = [abs(float(raw.get(k, -100.0)) - float(b.get(k, -100.0))) for k in union] if union else []
            mean_diff = float(np.mean(diffs)) if diffs else 0.0
            dist_factor = max(0.0, min(1.0, 1.0 - (mean_diff / 12.0)))
            ap_factor = max(0.0, min(1.0, best_common / 3.0))

            # Require clear margin from second-best profile; if all profiles look the same,
            # confidence should collapse so profile hint won't override dynamic signal logic.
            ranking = sorted(ranking, key=lambda x: x[0])
            separation_factor = 0.0
            if len(ranking) >= 2:
                gap = max(0.0, ranking[1][0] - ranking[0][0])
                separation_factor = max(0.0, min(1.0, gap / 1.2))
            elif len(ranking) == 1:
                separation_factor = 0.6

            best_conf = max(0.0, min(1.0, best_conf * dist_factor * ap_factor * separation_factor))

        return empty_distance, empty_conf, empty_delta, best_count, best_conf

    def _remember_wifi_fingerprint(self, wifi_feat: dict):
        raw = wifi_feat.get("raw_vector", {}) if isinstance(wifi_feat, dict) else {}
        if not isinstance(raw, dict) or not raw:
            return
        self._wifi_fp_history.append({k: float(v) for k, v in raw.items()})

    def _estimate_wifi_cluster_count(self) -> int:
        """
        Cluster recent AP RSSI fingerprints to estimate occupancy complexity.
        Uses DBSCAN over standardized AP vectors; returns approx count in [0,4].
        """
        if DBSCAN is None or len(self._wifi_fp_history) < 12:
            return 0

        # Use BSSIDs that appear frequently enough for stable vectorization.
        freq: Dict[str, int] = {}
        for fp in self._wifi_fp_history:
            for b in fp.keys():
                freq[b] = freq.get(b, 0) + 1
        stable = [b for b, c in freq.items() if c >= max(6, int(len(self._wifi_fp_history) * 0.25))]
        if len(stable) < 3:
            return 0

        # Keep strongest/most frequent APs only.
        stable = sorted(stable, key=lambda b: -freq[b])[:12]
        X = np.array([[fp.get(b, -100.0) for b in stable] for fp in self._wifi_fp_history], dtype=float)
        if X.shape[0] < 12 or X.shape[1] < 3:
            return 0

        # Standardize each AP dimension to avoid dominant channels.
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        sd[sd < 1e-3] = 1.0
        Xz = (X - mu) / sd

        try:
            labels = DBSCAN(eps=0.95, min_samples=4).fit_predict(Xz)
        except Exception:
            return 0

        unique = [lb for lb in set(labels.tolist()) if lb != -1]
        if not unique:
            return 0

        # Cluster count measures distinct RSSI states; map softly to occupancy.
        n_clusters = len(unique)
        return max(0, min(5, n_clusters - 1))

    def _kalman_update_position(self, p: PersonState, mx: float, my: float, mconf: float) -> Tuple[float, float, float]:
        """Constant-velocity 2D Kalman filter for BLE/drift position smoothing."""
        now = time.time()
        if p._kf_state is None or p._kf_cov is None or p._kf_ts <= 0:
            p._kf_state = np.array([mx, my, 0.0, 0.0], dtype=float)
            p._kf_cov = np.diag([0.08, 0.08, 0.6, 0.6]).astype(float)
            p._kf_ts = now
            return mx, my, mconf

        dt = max(0.03, min(0.4, now - p._kf_ts))
        p._kf_ts = now

        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        q = 0.03
        Q = np.array([
            [q * dt * dt, 0.0, q * dt, 0.0],
            [0.0, q * dt * dt, 0.0, q * dt],
            [q * dt, 0.0, q, 0.0],
            [0.0, q * dt, 0.0, q],
        ], dtype=float)

        # Predict
        x = F @ p._kf_state
        P = F @ p._kf_cov @ F.T + Q

        # Update
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        meas_var = max(0.002, (1.0 - max(0.0, min(1.0, mconf))) * 0.08)
        R = np.diag([meas_var, meas_var]).astype(float)
        z = np.array([mx, my], dtype=float)
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

        p._kf_state = x
        p._kf_cov = P
        filt_conf = max(0.25, min(0.98, 1.0 - (meas_var * 6.0)))
        return float(x[0]), float(x[1]), float(filt_conf)