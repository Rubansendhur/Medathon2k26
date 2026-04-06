PhysioTracker v3 - Multi-Person Edition
AI-powered contactless human motion analysis for physiotherapy and sports medicine.

Project Overview
PhysioTracker is a real-time monitoring platform that combines phone IMU signals, BLE RSSI estimates, and room-level wireless sensing to understand movement activity, fall risk, and approximate location of people in a room.

It is designed for:

Physiotherapy progress monitoring
Rehab session observation
Sports movement tracking
Safety-oriented fall-risk alerts in shared spaces
Core capabilities:

Multi-person live tracking
Per-person activity classification
Fall-risk detection with temporal confirmation
Real-time dashboard and websocket stream
API-first backend for mobile and web integration
Why the /phone route is important
The /phone page is the entry point that turns each mobile device into an active sensing node for one person.

Why this matters:

Device identity: each phone gets a stable device_id, so the backend can keep person data separated
Person mapping: in multi-person scenarios, one phone corresponds to one tracked person slot
Sensor stream source: IMU samples and BLE scan values are sent from this route to /api/imu
Real + simulated testing: it supports both live motion data and simulation mode for demos or development
Data quality: reliable, continuous input from /phone is what enables stable activity labels, cadence, and fall-risk logic
Without /phone, the system cannot provide robust per-person tracking; it would only have room-level inference signals, which are less precise for individual monitoring.

What is improved in v3
Multi-person tracking (fixed)
Each phone that opens /phone and sends data is tracked as a separate person
Each phone gets a stable unique device ID stored in browser local storage
You can also type a custom name like phone_alice or phone_bob
Up to 10 simultaneous phones are supported
Dashboard shows per-person activity, fall risk, cadence, and room position
False fall detection reduced
Previous behavior: a single noisy IMU sample could trigger a fall alert
Current behavior: fall detection requires 3 consecutive qualifying frames (hysteresis)
Both jerk > 0.55 and accel_rms > 1.6 must stay true for about 0.3 seconds before alerting
Counter decays when readings normalize to avoid sticky false alarms
BLE-based position estimation added
The /phone page can simulate 5 BLE beacon RSSI values (room corners plus center)
BLE values are transmitted with IMU data to backend
Backend applies log-distance RSSI modeling and weighted centroid trilateration
Radar and room canvas show each person with estimated position, name, and activity label
Dashboard upgrades
Persons card: mini-card for each connected phone with activity, fall risk, accel, jerk, cadence
Phone streams card: all live devices with current activity and fall risk
Room canvas: unique color and labels per person, with clear fall warning marker
Radar canvas: per-person dots, activity labels, and fall-risk highlighting

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
