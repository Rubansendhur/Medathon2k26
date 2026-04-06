"""
Bluetooth BLE RSSI Sensor — simulated.
Simulates 3-6 BLE beacons whose RSSI varies based on motion state.
"""
import numpy as np
import time
from collections import deque


BEACON_POOL = [
    {"mac": "AA:BB:CC:DD:EE:01", "name": "BLE-Beacon-1", "base_rssi": -55},
    {"mac": "AA:BB:CC:DD:EE:02", "name": "BLE-Beacon-2", "base_rssi": -62},
    {"mac": "AA:BB:CC:DD:EE:03", "name": "BLE-Beacon-3", "base_rssi": -70},
    {"mac": "AA:BB:CC:DD:EE:04", "name": "Fitness-Band",  "base_rssi": -48},
    {"mac": "AA:BB:CC:DD:EE:05", "name": "BLE-Beacon-5", "base_rssi": -75},
]


class BluetoothSensor:
    def __init__(self, window: int = 20):
        self.history: deque = deque(maxlen=window)
        self._phase = 0.0
        self._motion_level = 0.0   # 0=idle  1=walking  2=active

    def set_motion_level(self, level: float):
        self._motion_level = max(0.0, min(2.0, level))

    def scan(self) -> dict:
        """Simulate BLE scan — returns {mac: rssi}."""
        self._phase += 0.15
        results = {}
        noise_scale = 1.5 + self._motion_level * 3.0

        for beacon in BEACON_POOL:
            # drift + noise scaled to motion
            drift = np.sin(self._phase + hash(beacon["mac"]) % 10) * self._motion_level * 4
            noise = np.random.normal(0, noise_scale)
            rssi = beacon["base_rssi"] + drift + noise
            rssi = max(-100, min(-30, rssi))
            results[beacon["mac"]] = round(rssi, 1)

        return results

    def add_scan(self, scan: dict | None = None) -> dict:
        if scan is None:
            scan = self.scan()
        self.history.append(scan)
        return scan

    def extract_features(self) -> dict:
        base = {
            "motion_score": 0.0,
            "variance": 0.0,
            "total_change": 0.0,
            "n_beacons": 0,
            "raw_vector": {},
        }
        if len(self.history) < 5:
            return base

        macs = list(BEACON_POOL[i]["mac"] for i in range(len(BEACON_POOL)))
        matrix = np.array([
            [sc.get(m, -100) for m in macs]
            for sc in self.history
        ], dtype=float)

        var = float(np.mean(np.var(matrix, axis=0)))
        diffs = float(np.mean(np.abs(np.diff(matrix, axis=0))))
        motion_score = min(100.0, var * 3.0 + diffs * 12.0)

        return {
            "motion_score": round(motion_score, 2),
            "variance": round(min(var, 15), 3),
            "total_change": round(min(diffs, 10), 3),
            "n_beacons": len(macs),
            "raw_vector": {m: round(matrix[-1, i], 1) for i, m in enumerate(macs)},
        }
