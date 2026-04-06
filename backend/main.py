"""
PhysioTracker v3 — FastAPI backend (Multi-Person Edition)
==========================================================
Key improvements over v2:
  • Each phone that POSTs to /api/imu is tracked as a SEPARATE person.
  • Fall detection uses hysteresis (3 consecutive frames) — no more false
    positives from a single noisy sample.
  • BLE RSSI trilateration gives each person an estimated room position.
  • WiFi variance is used as a corroborating signal, not the primary
    person-count estimator.
  • /api/imu now accepts an optional ble_scan payload for real/simulated BLE.
  • Dashboard receives a `persons` list (one entry per phone), not a
    single merged result.
"""
import asyncio, json, time, os, math
from contextlib import asynccontextmanager
from collections import deque
from typing import Optional, List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sensors.wifi_sensor import WiFiSensor
from sensors.bluetooth_sensor import BluetoothSensor
from sensors.imu_sensor import IMUSensor, ActivityMode
from sensors.acoustic_rf_sensors import AcousticSensor, AmbientRFSensor
from ml.fusion import SensorFusion, FusionResult
from multi_person import MultiPersonTracker

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_JSONL = os.path.join(DATA_DIR, "imu_corpus.jsonl")
ROOM_BASELINES_JSON = os.path.join(DATA_DIR, "room_baselines.json")

# ── Sensor instances (ambient / room-wide) ────────────────────────────────────
wifi     = WiFiSensor(window=20)
bt       = BluetoothSensor(window=20)
imu      = IMUSensor(window=50, sample_rate=50.0)
acoustic = AcousticSensor(window=30)
rf       = AmbientRFSensor(window=30)
fusion   = SensorFusion()

tracker  = MultiPersonTracker()

# ── Global state ──────────────────────────────────────────────────────────────
_latest: Optional[FusionResult] = None
_activity_timeline: list = []
_connected_clients: set = set()
_loop_tick = 0
_loop_last_ts = 0.0
_loop_last_error = ""
_strict_real_mode = False
_last_wifi_networks: list = []
_room_target_count = 1
_room_force_target = False
_room_cfg_path = os.path.join(os.path.dirname(__file__), "room_config.json")
_room_baselines: Dict[str, dict] = {}
_active_room_name: Optional[str] = None


def _load_room_config():
    global _room_target_count, _room_force_target, _strict_real_mode
    try:
        if not os.path.isfile(_room_cfg_path):
            return
        with open(_room_cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _room_target_count = max(1, min(10, int(data.get("target_count", 1))))
        _room_force_target = bool(data.get("force_target", False))
        _strict_real_mode  = bool(data.get("strict_real_mode", False))
    except Exception:
        pass


def _save_room_config():
    try:
        with open(_room_cfg_path, "w", encoding="utf-8") as f:
            json.dump({"target_count": _room_target_count,
                       "force_target": _room_force_target,
                       "strict_real_mode": _strict_real_mode}, f)
    except Exception:
        pass


def _normalize_room_name(name: Optional[str]) -> Optional[str]:
    room = str(name or "").strip()
    if not room:
        return None
    room = room[:40]
    cleaned = "".join(ch for ch in room if ch.isalnum() or ch in ("_", "-", " ")).strip()
    return cleaned or None


def _save_room_baselines():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = {
            "active_room": _active_room_name,
            "rooms": _room_baselines,
        }
        with open(ROOM_BASELINES_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
    except Exception:
        pass


def _load_room_baselines():
    global _room_baselines, _active_room_name
    _room_baselines = {}
    _active_room_name = None
    try:
        if not os.path.isfile(ROOM_BASELINES_JSON):
            return
        with open(ROOM_BASELINES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        rooms = data.get("rooms", {}) if isinstance(data, dict) else {}
        if isinstance(rooms, dict):
            _room_baselines = rooms
        active = _normalize_room_name(data.get("active_room") if isinstance(data, dict) else None)
        if active and active in _room_baselines:
            _active_room_name = active
            rec = _room_baselines.get(active, {})
            profiles = rec.get("occupancy_profiles", {})
            if isinstance(profiles, dict) and profiles:
                tracker.load_occupancy_profiles(
                    profiles,
                    fallback_baseline=rec.get("baseline", {}),
                    fallback_ts=rec.get("captured_ts"),
                    fallback_scans=rec.get("baseline_scans"),
                )
            else:
                tracker.load_empty_room_baseline(
                    rec.get("baseline", {}),
                    captured_ts=rec.get("captured_ts"),
                    scans=rec.get("baseline_scans"),
                )
    except Exception:
        _room_baselines = {}
        _active_room_name = None


def _room_baseline_summary(wifi_feat: Optional[dict] = None) -> dict:
    status = tracker.get_empty_room_baseline_status(wifi_feat)
    status["active_room"] = _active_room_name
    status["stored_rooms"] = sorted(_room_baselines.keys())
    return status


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _sanitize_for_json(value.item())
        except Exception:
            return 0.0
    return value


async def sensor_loop():
    global _latest, _last_wifi_networks, _loop_tick, _loop_last_ts, _loop_last_error
    global _connected_clients, _strict_real_mode, _room_target_count, _room_force_target
    wifi_counter = 0
    wifi_feat = {"motion_score": 0.0, "n_networks": 0}
    focused_wifi_feat = wifi_feat

    while True:
        t0 = time.time()
        try:
            if _strict_real_mode:
                imu_feat  = tracker.get_merged_imu_features()
                bt_feat   = {"motion_score": 0.0, "variance": 0.0,
                             "total_change": 0.0, "n_beacons": 0, "raw_vector": {}}
                acou_feat = {"motion_score": 0.0, "echo_variance": 0.0,
                             "doppler_mean": 0.0, "phase_variance": 0.0}
                rf_feat   = {"motion_score": 0.0, "doppler_variance": 0.0,
                             "entropy_change": 0.0, "power_fluctuation": 0.0}
            else:
                merged = tracker.get_merged_imu_features()
                room_motion_score = float(max(0.0, min(100.0, wifi_feat.get("motion_score", 0.0))))
                motion_level = room_motion_score / 50.0
                bt.set_motion_level(motion_level)
                acoustic.set_motion_level(motion_level)
                rf.set_motion_level(motion_level)

                bt.add_scan(); acoustic.read(); rf.read(); imu.read()
                bt_feat   = bt.extract_features()
                acou_feat = acoustic.extract_features()
                rf_feat   = rf.extract_features()
                imu_feat  = merged if tracker.get_active_persons() else imu.extract_features()

            wifi_counter += 1
            if wifi_counter >= 3:
                wifi_counter = 0
                loop = asyncio.get_running_loop()
                wifi.set_simulated_fallback(not _strict_real_mode)
                await loop.run_in_executor(None, wifi.add_scan)
                wifi_feat = wifi.extract_features()
                focused_wifi_feat = tracker.focus_wifi_features(wifi_feat)
                raw_for_ui = focused_wifi_feat.get("raw_vector", {}) if isinstance(focused_wifi_feat, dict) else {}
                if raw_for_ui:
                    _last_wifi_networks = [
                        {"bssid": b, "ssid": wifi.known_networks.get(b, "?"),
                         "rssi": round(v, 1),
                         "signal_pct": max(0, min(100, int((v + 100) * 2)))}
                        for b, v in sorted(raw_for_ui.items(), key=lambda x: -x[1])
                    ]
                elif wifi.history:
                    raw = wifi.history[-1]
                    _last_wifi_networks = [
                        {"bssid": b, "ssid": wifi.known_networks.get(b, "?"),
                         "rssi": round(v, 1),
                         "signal_pct": max(0, min(100, int((v + 100) * 2)))}
                        for b, v in sorted(raw.items(), key=lambda x: -x[1])
                    ]

            result = fusion.fuse(focused_wifi_feat, bt_feat, imu_feat, acou_feat, rf_feat)
            _latest = result

            room_state = tracker.get_room_state(focused_wifi_feat)
            dominant = room_state.dominant_activity

            if (not _activity_timeline or
                    _activity_timeline[-1]["activity"] != dominant):
                _activity_timeline.append({
                    "time": time.time(), "activity": dominant,
                    "confidence": result.confidence, "fused_score": result.fused_score,
                    "person_count": room_state.person_count,
                })
                if len(_activity_timeline) > 200:
                    _activity_timeline.pop(0)

            if _connected_clients:
                payload = _sanitize_for_json(_build_payload(result, room_state))
                msg = json.dumps(payload, allow_nan=False)
                dead = set()
                for ws in _connected_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                _connected_clients.difference_update(dead)

            _loop_tick += 1
            _loop_last_ts = time.time()
            _loop_last_error = ""

        except Exception as e:
            _loop_last_error = f"{type(e).__name__}: {e}"
            print(f"[sensor_loop] {_loop_last_error}")

        elapsed = time.time() - t0
        await asyncio.sleep(max(0.05, 0.1 - elapsed))


def _build_payload(r: FusionResult, room_state) -> dict:
    active = tracker.get_active_persons()
    first  = active[0] if active else None
    physio = {
        "gait_score":    first.gait_score      if first else r.gait_score,
        "rom_score":     first.rom_score       if first else r.rom_score,
        "symmetry_score":first.symmetry_score  if first else r.symmetry_score,
        "rep_count":     first.rep_count       if first else r.rep_count,
        "fatigue_index": first.fatigue_index   if first else r.fatigue_index,
        "step_cadence":  first.step_cadence    if first else r.step_cadence,
        "fall_risk":     first.fall_risk_score if first else r.fall_risk,
    }
    return {
        "ts":              time.time(),
        "activity":        room_state.dominant_activity,
        "confidence":      r.confidence,
        "fused_score":     r.fused_score,
        "ml_probabilities": r.ml_probabilities,
        "person_count":    room_state.person_count,
        "persons":         room_state.persons,
        "room_state": {
            "target_count":    _room_target_count,
            "force_target":    _room_force_target,
            "estimated_count": room_state.estimated_count,
            "occupancy_band":  room_state.occupancy_band,
            "tracked_phone_count": room_state.phone_count,
            "room_motion_score": room_state.room_motion_score,
            "baseline_ready": room_state.baseline_ready,
            "baseline_distance_db": room_state.baseline_distance_db,
            "baseline_confidence": room_state.baseline_confidence,
            "occupancy_state": (
                "matches target" if room_state.person_count == _room_target_count
                else ("below target" if room_state.person_count < _room_target_count
                      else "above target")
            ),
            "motion_band":       room_state.motion_band,
            "dominant_activity": room_state.dominant_activity,
            "activity_hint": {
                "idle":      "mostly still",
                "walk":      "walking or pacing",
                "squat":     "lower-body exercise",
                "bend":      "bending / reach motion",
                "lift":      "lifting or overhead work",
                "fall_risk": "abrupt / unstable motion",
                "unknown":   "no phone data",
            }.get(room_state.dominant_activity, room_state.dominant_activity),
            "confidence":        round(r.confidence, 2),
        },
        "physio": physio,
        "sensors": {
            "wifi": r.wifi_features, "bt": r.bt_features,
            "imu": r.imu_features, "acoustic": r.acoustic_features,
            "rf": r.rf_features,
        },
        "sensor_sources": {
            "wifi_real":        bool(r.wifi_features.get("is_real", False)),
            "phone_count":      len(active),
            "simulated_core":   not _strict_real_mode,
            "strict_real_mode": _strict_real_mode,
        },
        "sensor_scores": r.sensor_scores,
        "wifi_networks": _last_wifi_networks[:12],
        "model_info":    r.model_info,
    }


@asynccontextmanager
async def lifespan(app):
    _load_room_config()
    _load_room_baselines()
    wifi.set_simulated_fallback(not _strict_real_mode)
    task = asyncio.create_task(sensor_loop())
    yield
    task.cancel()


app = FastAPI(title="PhysioTracker v3 Multi-Person", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
async def serve_frontend():
    html_path = os.path.join(frontend_dir, "index.html")
    if os.path.isfile(html_path): return FileResponse(html_path)
    return {"message": "PhysioTracker v3 API"}


@app.get("/phone")
async def serve_phone():
    html_path = os.path.join(frontend_dir, "phone.html")
    if os.path.isfile(html_path): return FileResponse(html_path)
    return {"error": "not found"}


@app.get("/api/status")
async def status():
    active = tracker.get_active_persons()
    room_state = tracker.get_room_state(_latest.wifi_features if _latest else {})
    baseline = _room_baseline_summary(_latest.wifi_features if _latest else None)
    return {
        "status":           "running",
        "wifi_networks":    _latest.wifi_features.get("n_networks", 0) if _latest else 0,
        "activity":         _latest.activity if _latest else "idle",
        "person_count":     room_state.person_count,
        "estimated_count":  room_state.estimated_count,
        "phone_count":      room_state.phone_count,
        "connected_phones": [p.device_id for p in active],
        "room_baseline": baseline,
        "room_target_count": _room_target_count,
        "strict_real_mode": _strict_real_mode,
        "ml":               fusion.get_model_info(),
        "loop": {
            "tick":     _loop_tick,
            "last_ts":  _loop_last_ts,
            "age_s":    round((time.time() - _loop_last_ts), 3) if _loop_last_ts else None,
            "last_error": _loop_last_error,
        },
    }


@app.get("/api/snapshot")
async def snapshot():
    if not _latest: raise HTTPException(503, "Not ready")
    room_state = tracker.get_room_state(_latest.wifi_features)
    return _sanitize_for_json(_build_payload(_latest, room_state))


@app.get("/api/timeline")
async def timeline():
    return {"timeline": _activity_timeline[-50:]}


@app.get("/api/wifi/networks")
async def wifi_networks():
    return {"networks": _last_wifi_networks}


@app.get("/api/ml/info")
async def ml_info():
    return fusion.get_model_info()


@app.get("/api/persons")
async def get_persons():
    active = tracker.get_active_persons()
    return {
        "count": len(active),
        "persons": [
            {"device_id": p.device_id, "activity": p.activity,
             "confidence": p.confidence, "fall_risk": p.fall_risk_score,
             "accel_rms": p.accel_rms, "jerk": p.jerk,
             "step_cadence": p.step_cadence, "pos_x": p.pos_x, "pos_y": p.pos_y}
            for p in active
        ],
    }


# ── Phone IMU — accepts per-device_id streams ──────────────────────────────────

class PhoneIMUData(BaseModel):
    samples: list
    device_id: Optional[str] = "phone_default"
    source_mode: Optional[str] = "real"
    sim_activity: Optional[str] = None
    ble_scan: Optional[Dict[str, float]] = None   # {mac: rssi_dBm} optional
    room_match: Optional[bool] = None
    gps_accuracy_m: Optional[float] = None


class LabeledCaptureData(BaseModel):
    activity: str
    device_id: Optional[str] = "phone_default"
    samples: list
    source_mode: Optional[str] = "real"
    room_match: Optional[bool] = None
    gps_accuracy_m: Optional[float] = None


@app.post("/api/imu")
async def receive_phone_imu(data: PhoneIMUData):
    """Each unique device_id is tracked as an independent person."""
    device_id   = str(data.device_id or "phone_default").strip() or "phone_default"
    source_mode = "simulated" if str(data.source_mode or "").lower() == "sim" else "real"
    sim_act     = str(data.sim_activity) if data.sim_activity else None

    person = tracker.update_device(
        device_id=device_id,
        samples=data.samples,
        bt_scan=data.ble_scan,
        source_mode=source_mode,
        sim_activity=sim_act,
        room_match=data.room_match,
        gps_accuracy_m=data.gps_accuracy_m,
    )
    room_state = tracker.get_room_state(_latest.wifi_features if _latest else {})
    return {
        "ok":           True,
        "device_id":    device_id,
        "buffered":     len(data.samples),
        "activity":     person.activity,
        "confidence":   person.confidence,
        "fall_risk":    person.fall_risk_score,
        "person_count": room_state.person_count,
        "estimated_count": room_state.estimated_count,
        "room_match": person.room_match,
    }


@app.post("/api/dataset/append")
async def append_labeled_capture(data: LabeledCaptureData):
    """Append one labeled IMU window to local corpus JSONL for offline retraining."""
    activity = str(data.activity or "").strip().lower()
    if activity not in {"idle", "walk", "squat", "bend", "lift", "fall_risk"}:
        raise HTTPException(400, "Invalid activity label")
    if not isinstance(data.samples, list) or len(data.samples) < 8:
        raise HTTPException(400, "Need at least 8 IMU samples")

    os.makedirs(DATA_DIR, exist_ok=True)
    rec = {
        "ts": time.time(),
        "activity": activity,
        "device_id": str(data.device_id or "phone_default"),
        "source_mode": str(data.source_mode or "real"),
        "room_match": data.room_match,
        "gps_accuracy_m": data.gps_accuracy_m,
        "samples": data.samples,
    }
    with open(DATASET_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    return {"ok": True, "activity": activity, "samples": len(data.samples)}


@app.get("/api/dataset/stats")
async def dataset_stats():
    if not os.path.isfile(DATASET_JSONL):
        return {"exists": False, "records": 0, "by_activity": {}}
    by_activity = {"idle": 0, "walk": 0, "squat": 0, "bend": 0, "lift": 0, "fall_risk": 0}
    records = 0
    with open(DATASET_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                act = json.loads(line).get("activity", "")
                if act in by_activity:
                    by_activity[act] += 1
            except Exception:
                continue
    return {"exists": True, "records": records, "by_activity": by_activity}


class SimulateRequest(BaseModel):
    activity: str
    device_id: Optional[str] = "sim_phone"


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    import random, math as _m
    mode_map = {
        "idle": ActivityMode.IDLE, "walk": ActivityMode.WALK,
        "squat": ActivityMode.SQUAT, "bend": ActivityMode.BEND,
        "lift": ActivityMode.LIFT, "fall_risk": ActivityMode.IDLE,
    }
    act = req.activity.lower()
    if act not in mode_map:
        raise HTTPException(400, f"Choose from: {list(mode_map)}")
    imu.set_mode(mode_map[act])

    p = {
        "idle":      (0.05, 0.02, 0.2,  0.02),
        "walk":      (0.9,  1.2,  1.8,  0.15),
        "squat":     (0.6,  2.0,  0.6,  0.10),
        "bend":      (0.5,  2.5,  0.4,  0.08),
        "lift":      (1.2,  1.8,  0.5,  0.20),
        "fall_risk": (2.2,  1.8,  1.5,  0.30),
    }[act]
    a_amp, g_amp, freq, noise = p

    phase = random.uniform(0, _m.pi * 2)
    sr = 50.0
    samples = []
    for i in range(25):
        phase += 2 * _m.pi * freq / sr
        samples.append({
            "ax": round(_m.sin(phase) * a_amp + random.gauss(0, noise), 4),
            "ay": round(_m.cos(phase * 1.3) * a_amp * 0.7 + random.gauss(0, noise), 4),
            "az": round(9.81 + _m.sin(phase * 0.5) * a_amp * 0.3, 4),
            "gx": round(_m.sin(phase + 0.5) * g_amp + random.gauss(0, noise * 2), 4),
            "gy": round(_m.cos(phase + 1.0) * g_amp * 0.8 + random.gauss(0, noise * 2), 4),
            "gz": round(_m.sin(phase * 1.2) * g_amp * 0.5 + random.gauss(0, noise), 4),
            "ts": time.time() + i / sr,
        })

    device_id = str(req.device_id or "sim_phone")
    tracker.update_device(device_id=device_id, samples=samples,
                          bt_scan=None, source_mode="simulated", sim_activity=act)
    return {"ok": True, "mode": act, "device_id": device_id}


class RoomTargetRequest(BaseModel):
    count: int
    force_target: Optional[bool] = None


class BaselineCaptureRequest(BaseModel):
    scans: Optional[int] = 8
    confirm_empty: Optional[bool] = True
    room_name: Optional[str] = None
    save_room: Optional[bool] = True
    occupancy_count: Optional[int] = 0


class RoomBaselineSelectRequest(BaseModel):
    room_name: str


@app.get("/api/room/target")
async def get_room_target():
    room_state = tracker.get_room_state(_latest.wifi_features if _latest else {})
    baseline = _room_baseline_summary(_latest.wifi_features if _latest else None)
    return {
        "target_count": _room_target_count,
        "force_target": _room_force_target,
        "strict_real_mode": _strict_real_mode,
        "estimated_count": room_state.estimated_count,
        "occupancy_band": room_state.occupancy_band,
        "room_state": {
            "occupancy_state": (
                "matches target" if room_state.person_count == _room_target_count
                else ("below target" if room_state.person_count < _room_target_count else "above target")
            ),
            "activity_hint": room_state.dominant_activity,
        },
        "baseline": baseline,
    }


@app.post("/api/room/target")
async def set_room_target(req: RoomTargetRequest):
    global _room_target_count, _room_force_target
    _room_target_count = max(1, min(10, int(req.count)))
    if req.force_target is not None:
        _room_force_target = bool(req.force_target)
    _save_room_config()
    return {"ok": True, "target_count": _room_target_count, "force_target": _room_force_target}


@app.get("/api/room/baseline")
async def get_room_baseline():
    wifi_feat = _latest.wifi_features if _latest else wifi.extract_features()
    return _room_baseline_summary(wifi_feat)


@app.get("/api/room/baselines")
async def list_room_baselines():
    wifi_feat = _latest.wifi_features if _latest else None
    current = tracker.get_empty_room_baseline_status(wifi_feat)
    rooms = []
    for name in sorted(_room_baselines.keys()):
        rec = _room_baselines.get(name, {})
        rooms.append({
            "room_name": name,
            "baseline_ap_count": int(rec.get("baseline_ap_count", 0) or 0),
            "baseline_scans": int(rec.get("baseline_scans", 0) or 0),
            "captured_ts": rec.get("captured_ts"),
            "occupancy_profiles": sorted(int(k) for k in (rec.get("occupancy_profiles", {}) or {}).keys()),
            "is_active": name == _active_room_name,
        })
    return {
        "active_room": _active_room_name,
        "current": current,
        "rooms": rooms,
    }


@app.post("/api/room/baseline/capture")
async def capture_room_baseline(req: BaselineCaptureRequest):
    global _active_room_name
    if req.confirm_empty is False:
        raise HTTPException(400, "confirm_empty must be true to avoid accidental baseline capture")

    scans = max(4, min(20, int(req.scans or 8)))
    occupancy_count = max(0, min(6, int(req.occupancy_count or 0)))
    loop = asyncio.get_running_loop()
    vectors = []
    for _ in range(scans):
        await loop.run_in_executor(None, wifi.add_scan)
        feat = wifi.extract_features()
        raw = feat.get("raw_vector", {})
        if isinstance(raw, dict) and raw:
            vectors.append(raw)
        await asyncio.sleep(0.12)

    baseline = tracker.set_occupancy_baseline(occupancy_count, vectors)
    room_name = _normalize_room_name(req.room_name)
    saved_room = None
    if baseline.get("ready") and (req.save_room is not False):
        if room_name:
            exp = tracker.export_empty_room_baseline()
            _room_baselines[room_name] = {
                "baseline": exp.get("baseline", {}),
                "captured_ts": exp.get("captured_ts"),
                "baseline_scans": exp.get("baseline_scans"),
                "baseline_ap_count": exp.get("baseline_ap_count"),
                "occupancy_profiles": exp.get("occupancy_profiles", {}),
            }
            _active_room_name = room_name
            saved_room = room_name
            _save_room_baselines()
    return {
        "ok": bool(baseline.get("ready")),
        "captured_scans": scans,
        "used_vectors": len(vectors),
        "baseline": baseline,
        "saved_room": saved_room,
        "active_room": _active_room_name,
        "occupancy_count": occupancy_count,
    }


@app.delete("/api/room/baseline")
async def clear_room_baseline():
    global _active_room_name
    baseline = tracker.clear_empty_room_baseline()
    _active_room_name = None
    _save_room_baselines()
    return {"ok": True, "baseline": _room_baseline_summary()}


@app.post("/api/room/baseline/select")
async def select_room_baseline(req: RoomBaselineSelectRequest):
    global _active_room_name
    room_name = _normalize_room_name(req.room_name)
    if not room_name or room_name not in _room_baselines:
        raise HTTPException(404, "Room baseline not found")
    rec = _room_baselines[room_name]
    profiles = rec.get("occupancy_profiles", {})
    if isinstance(profiles, dict) and profiles:
        baseline = tracker.load_occupancy_profiles(
            profiles,
            fallback_baseline=rec.get("baseline", {}),
            fallback_ts=rec.get("captured_ts"),
            fallback_scans=rec.get("baseline_scans"),
        )
    else:
        baseline = tracker.load_empty_room_baseline(
            rec.get("baseline", {}),
            captured_ts=rec.get("captured_ts"),
            scans=rec.get("baseline_scans"),
        )
    _active_room_name = room_name
    _save_room_baselines()
    return {"ok": True, "active_room": _active_room_name, "baseline": baseline}


@app.delete("/api/room/baseline/{room_name}")
async def delete_room_baseline(room_name: str):
    global _active_room_name
    room = _normalize_room_name(room_name)
    if not room or room not in _room_baselines:
        raise HTTPException(404, "Room baseline not found")
    del _room_baselines[room]
    if _active_room_name == room:
        _active_room_name = None
        tracker.clear_empty_room_baseline()
    _save_room_baselines()
    return {"ok": True, "deleted": room, "active_room": _active_room_name}


class ModeRequest(BaseModel):
    strict_real_mode: bool


@app.get("/api/mode")
async def get_mode():
    return {"strict_real_mode": _strict_real_mode}


@app.post("/api/mode")
async def set_mode(req: ModeRequest):
    global _strict_real_mode
    _strict_real_mode = bool(req.strict_real_mode)
    wifi.set_simulated_fallback(not _strict_real_mode)
    _save_room_config()
    return {"ok": True, "strict_real_mode": _strict_real_mode}


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    import requests

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    ctx = req.context or {}
    active = tracker.get_active_persons()
    if active:
        ctx.update({"person_count": len(active), "persons": [
            {"device_id": p.device_id, "activity": p.activity, "fall_risk": p.fall_risk_score}
            for p in active
        ]})
    elif _latest:
        ctx.update({"activity": _latest.activity, "fall_risk": _latest.fall_risk,
                    "fatigue": _latest.fatigue_index})
    sys_p = (f"You are an expert AI physiotherapy assistant. "
             f"Current metrics: {json.dumps(ctx)}. "
             "Be concise (2-4 sentences). Focus on actionable advice.")
    try:
        payload = {
            "model": ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": req.message},
            ],
            "options": {"num_predict": 220, "temperature": 0.3},
        }
        resp = requests.post(f"{ollama_base}/api/chat", json=payload, timeout=30)
        if resp.status_code != 200:
            return {"reply": f"Ollama error ({resp.status_code}): {resp.text[:200]}"}
        data = resp.json()
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            text = "No response from Ollama model."
        return {"reply": text}
    except Exception as e:
        return {"reply": f"Ollama unreachable or failed: {e}"}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    global _connected_clients
    await websocket.accept()
    _connected_clients.add(websocket)
    try:
        while True: await websocket.receive_text()
    except Exception:
        _connected_clients.discard(websocket)