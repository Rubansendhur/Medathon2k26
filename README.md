# PhysioTracker v3 — Multi-Person Edition

AI-Powered Contactless Human Motion Analysis · Physiotherapy & Sports Medicine

---

## What's New in This Version

### Multi-Person Tracking (Fixed)
- Each phone that opens `/phone` and sends data is tracked as a **separate person**
- Each phone gets a stable unique Device ID stored in its browser (localStorage)
- You can also type a custom name like `phone_alice` or `phone_bob`
- Up to 10 simultaneous phones supported
- The dashboard shows **per-person activity, fall risk, cadence, and room position**

### False Fall Detection Fixed
- Old behaviour: a single noisy IMU sample could trigger "FALL RISK"
- New behaviour: fall detection requires **3 consecutive qualifying frames** (hysteresis)
  - Both `jerk > 0.55` AND `accel_rms > 1.6` must be true for ~0.3 s before alerting
  - Counter decays when readings return to normal — no sticky false alarms

### Simulated Bluetooth for Position Estimation
- The phone page now simulates **5 BLE beacon RSSI values** (room corners + centre)
- These are sent alongside IMU data to the backend
- Backend uses a **log-distance RSSI model + weighted centroid trilateration** to estimate each person's position in the room
- The radar/room canvas now shows each person at their estimated position with their device name and activity label

### Dashboard Improvements
- **Persons card**: shows a mini-card per connected phone with activity, fall risk, accel, jerk, cadence
- **Phone Streams card**: lists all live devices with current activity and fall risk
- **Room canvas**: each person has their own colour, name label, and activity label; fall risk triggers a red `⚠ FALL` annotation
- **Radar canvas**: colour-coded dots per person, activity label under each dot, red ring for fall risk

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1) Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 2) Run backend

macOS/Linux:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell:

```powershell
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in a browser for the dashboard.

### Testing Multiple Persons

1. Open `http://<your-ip>:8000/phone` in **two different browser tabs or devices**
2. Each will auto-generate a unique Device ID (or type your own, e.g. `phone_alice`)
3. Start a Sim Stream on each with a different activity (e.g. one WALK, one SQUAT)
4. The dashboard will show 2 persons with different colours, positions, and activities

---

## Architecture

```
Phone A (device_id=phone_alice) ──┐
Phone B (device_id=phone_bob)  ──┤──► /api/imu ──► MultiPersonTracker
Phone C (device_id=phone_carol) ─┘                      │
                                                         ▼
WiFi RSSI (real or simulated) ────────────────► SensorFusion (room-wide)
BLE sim (5 beacons, per phone) ──► trilaterate ──► PersonState.pos_x/y
                                                         │
                                                         ▼
                                              /ws/stream (10 Hz WebSocket)
                                                         │
                                                         ▼
                                                  Dashboard (index.html)
```

### Person Count Sources (Priority Order)
1. **Connected phones** — each phone = 1 person (primary source)
2. **WiFi RSSI dynamics** — provides a room-wide motion corroboration signal (`room_motion_score`), not direct counting
3. **Phone-absent fallback** — if no phones are connected but WiFi variance/motion is high, dashboard shows one inferred `?` person

### Fall Detection Thresholds
| Parameter | Value | Meaning |
|---|---|---|
| `FALL_JERK_THRESH` | 0.55 | Min jerk (mean |Δaccel|) to start counting |
| `FALL_ACCEL_THRESH` | 1.6 m/s² | Min accel RMS to start counting |
| `FALL_CONFIRM_FRAMES` | 3 | Consecutive frames both must be true |
| Decay rate | −1/frame | Resets after 1 quiet frame |

### Sensor Signal Summary
| Signal | Source | Role |
|---|---|---|
| WiFi RSSI | Real (all BSSIDs) | Room-wide motion corroboration (`room_motion_score`) |
| Bluetooth BLE | Simulated (5 beacons) | Per-person room position via RSSI trilateration |
| IMU Accel+Gyro | Phone real or simulated | Per-person activity classification, fall detection |
| Acoustic Doppler | Simulated | Room-wide motion score contribution |
| Ambient RF | Simulated | Room-wide motion score contribution |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard |
| `/phone` | GET | Phone sensor page |
| `/api/imu` | POST | Receive IMU + BLE scan from one phone (device_id identifies person) |
| `/api/persons` | GET | Current state of all active person slots |
| `/api/simulate` | POST | Inject simulated activity for a named device |
| `/api/status` | GET | System status + connected phones list |
| `/api/snapshot` | GET | Full current payload |
| `/api/timeline` | GET | Activity history |
| `/api/room/target` | GET/POST | Room occupancy target |
| `/api/mode` | GET/POST | Strict-real vs mixed mode |
| `/api/chat` | POST | AI physiotherapy coach (needs ANTHROPIC_API_KEY) |
| `/ws/stream` | WS | 10 Hz real-time stream |

### POST /api/imu payload
```json
{
  "samples": [{"ax":0.1,"ay":0.2,"az":9.8,"gx":0.01,"gy":0.02,"gz":0.01,"ts":1234567890.0}],
  "device_id": "phone_alice",
  "source_mode": "real",
  "sim_activity": null,
  "ble_scan": {
    "AA:BB:CC:DD:EE:01": -58.3,
    "AA:BB:CC:DD:EE:02": -72.1,
    "AA:BB:CC:DD:EE:03": -81.0,
    "AA:BB:CC:DD:EE:04": -65.4,
    "AA:BB:CC:DD:EE:05": -77.2
  }
}
```

---

## Limitations & Next Steps

- Person count is currently phone-driven; missing phones can only be flagged as inferred `?`, not precisely counted
- WiFi provides corroborating room-motion only in Phase 1
- CSI-based multi-person counting is planned as a **Phase 2** upgrade
- BLE trilateration quality depends on browser BLE support and beacon visibility
- Activity classifier is rule-based — train a CNN-LSTM model on collected ground-truth data for better accuracy
