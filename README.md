# PhysioTracker v3 - Multi-Person Edition

AI-powered contactless human motion analysis for physiotherapy and sports medicine.

---

## Project Overview

PhysioTracker is a real-time monitoring platform that combines phone IMU signals, BLE RSSI estimates, and room-level wireless sensing to understand movement activity, fall risk, and approximate location of people in a room.

It is designed for:
- Physiotherapy progress monitoring
- Rehab session observation
- Sports movement tracking
- Safety-oriented fall-risk alerts in shared spaces

Core capabilities:
- Multi-person live tracking
- Per-person activity classification
- Fall-risk detection with temporal confirmation
- Real-time dashboard and websocket stream
- API-first backend for mobile and web integration

### Why the /phone route is important

The /phone page is the entry point that turns each mobile device into an active sensing node for one person.

Why this matters:
- Device identity: each phone gets a stable device_id, so the backend can keep person data separated
- Person mapping: in multi-person scenarios, one phone corresponds to one tracked person slot
- Sensor stream source: IMU samples and BLE scan values are sent from this route to /api/imu
- Real plus simulated testing: it supports both live motion data and simulation mode for demos and development
- Data quality: reliable, continuous input from /phone is what enables stable activity labels, cadence, and fall-risk logic

Without /phone, the system cannot provide robust per-person tracking; it would only have room-level inference signals, which are less precise for individual monitoring.

---

## Improvements in v3

### Multi-Person Tracking (Fixed)
- Each phone that opens /phone and sends data is tracked as a separate person
- Each phone gets a stable unique Device ID stored in its browser (localStorage)
- You can also type a custom name like phone_alice or phone_bob
- Up to 10 simultaneous phones supported
- The dashboard shows per-person activity, fall risk, cadence, and room position

### False Fall Detection Fixed
- Old behaviour: a single noisy IMU sample could trigger FALL RISK
- New behaviour: fall detection requires 3 consecutive qualifying frames (hysteresis)
- Both jerk > 0.55 and accel_rms > 1.6 must be true for about 0.3 s before alerting
- Counter decays when readings return to normal, so false alarms are reduced

### Simulated Bluetooth for Position Estimation
- The phone page now simulates 5 BLE beacon RSSI values (room corners plus center)
- These are sent alongside IMU data to the backend
- Backend uses a log-distance RSSI model plus weighted centroid trilateration to estimate each person's position in the room
- The radar and room canvas now shows each person at their estimated position with their device name and activity label

### Dashboard Improvements
- Persons card: shows a mini-card per connected phone with activity, fall risk, accel, jerk, cadence
- Phone Streams card: lists all live devices with current activity and fall risk
- Room canvas: each person has their own colour, name label, and activity label; fall risk triggers a red FALL annotation
- Radar canvas: colour-coded dots per person, activity label under each dot, red ring for fall risk

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

Open http://localhost:8000 in a browser for the dashboard.

### Testing Multiple Persons

1. Open http://<your-ip>:8000/phone in two different browser tabs or devices
2. Each will auto-generate a unique Device ID (or type your own, for example phone_alice)
3. Start a Sim Stream on each with a different activity (for example one WALK, one SQUAT)
4. The dashboard will show 2 persons with different colours, positions, and activities

---

## Architecture

```text
Phone A (device_id=phone_alice) --+
Phone B (device_id=phone_bob)   --+--> /api/imu --> MultiPersonTracker
Phone C (device_id=phone_carol) --+                      |
                                                         v
WiFi RSSI (real or simulated) ----------------> SensorFusion (room-wide)
BLE sim (5 beacons, per phone) --> trilaterate --> PersonState.pos_x/y
                                                         |
                                                         v
                                              /ws/stream (10 Hz WebSocket)
                                                         |
                                                         v
                                                  Dashboard (index.html)
```

### Person Count Sources (Priority Order)
1. Connected phones - each phone = 1 person (primary source)
2. WiFi RSSI dynamics - provides a room-wide motion corroboration signal (room_motion_score), not direct counting
3. Phone-absent fallback - if no phones are connected but WiFi variance or motion is high, dashboard shows one inferred ? person

### Fall Detection Thresholds

| Parameter | Value | Meaning |
|---|---|---|
| FALL_JERK_THRESH | 0.55 | Min jerk (mean abs delta accel) to start counting |
| FALL_ACCEL_THRESH | 1.6 m/s^2 | Min accel RMS to start counting |
| FALL_CONFIRM_FRAMES | 3 | Consecutive frames where both conditions are true |
| Decay rate | -1/frame | Resets after 1 quiet frame |

### Sensor Signal Summary

| Signal | Source | Role |
|---|---|---|
| WiFi RSSI | Real (all BSSIDs) | Room-wide motion corroboration (room_motion_score) |
| Bluetooth BLE | Simulated (5 beacons) | Per-person room position via RSSI trilateration |
| IMU Accel+Gyro | Phone real or simulated | Per-person activity classification, fall detection |
| Acoustic Doppler | Simulated | Room-wide motion score contribution |
| Ambient RF | Simulated | Room-wide motion score contribution |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| / | GET | Dashboard |
| /phone | GET | Phone sensor page |
| /api/imu | POST | Receive IMU plus BLE scan from one phone (device_id identifies person) |
| /api/persons | GET | Current state of all active person slots |
| /api/simulate | POST | Inject simulated activity for a named device |
| /api/status | GET | System status plus connected phones list |
| /api/snapshot | GET | Full current payload |
| /api/timeline | GET | Activity history |
| /api/room/target | GET/POST | Room occupancy target |
| /api/mode | GET/POST | Strict-real vs mixed mode |
| /api/chat | POST | AI physiotherapy coach (needs ANTHROPIC_API_KEY) |
| /ws/stream | WS | 10 Hz real-time stream |

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
