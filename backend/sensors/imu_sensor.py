"""
IMU Sensor (Accelerometer + Gyroscope) — simulated.
Generates realistic motion patterns for different activities.
"""
import numpy as np
from collections import deque
from enum import Enum


class ActivityMode(Enum):
    IDLE = "idle"
    WALK = "walk"
    SQUAT = "squat"
    BEND = "bend"
    LIFT = "lift"


# Per-activity parameters: (accel_amp, gyro_amp, freq_hz, noise)
ACTIVITY_PARAMS = {
    ActivityMode.IDLE:  (0.05, 0.02, 0.2,  0.02),
    ActivityMode.WALK:  (0.9,  1.2,  1.8,  0.15),
    ActivityMode.SQUAT: (0.6,  2.0,  0.6,  0.10),
    ActivityMode.BEND:  (0.5,  2.5,  0.4,  0.08),
    ActivityMode.LIFT:  (1.2,  1.8,  0.5,  0.20),
}


class IMUSensor:
    def __init__(self, window: int = 50, sample_rate: float = 50.0):
        self.window = window
        self.sample_rate = sample_rate
        self.history: deque = deque(maxlen=window)
        self._phase = 0.0
        self._mode = ActivityMode.IDLE
        self._transition = 0.0   # blend factor for smooth mode changes

    def set_mode(self, mode: ActivityMode):
        self._mode = mode

    def read(self) -> dict:
        """Return one IMU sample {ax, ay, az, gx, gy, gz}."""
        amp_a, amp_g, freq, noise = ACTIVITY_PARAMS[self._mode]
        self._phase += 2 * np.pi * freq / self.sample_rate

        # Gravity component
        ax = np.sin(self._phase) * amp_a + np.random.normal(0, noise)
        ay = np.cos(self._phase * 1.3) * amp_a * 0.7 + np.random.normal(0, noise)
        az = 9.81 + np.sin(self._phase * 0.5) * amp_a * 0.3 + np.random.normal(0, noise * 0.5)

        gx = np.sin(self._phase + 0.5) * amp_g + np.random.normal(0, noise * 2)
        gy = np.cos(self._phase + 1.0) * amp_g * 0.8 + np.random.normal(0, noise * 2)
        gz = np.sin(self._phase * 1.2) * amp_g * 0.5 + np.random.normal(0, noise)

        sample = {
            "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
            "gx": round(gx, 4), "gy": round(gy, 4), "gz": round(gz, 4),
        }
        self.history.append(sample)
        return sample

    def extract_features(self) -> dict:
        if len(self.history) < 10:
            return {
                "motion_score": 0.0, "jerk": 0.0, "rep_count": 0,
                "step_cadence": 0.0, "accel_rms": 0.0, "gyro_rms": 0.0,
                "dominant_freq": 0.0, "activity_hint": "idle",
            }

        arr = np.array([[s["ax"], s["ay"], s["az"], s["gx"], s["gy"], s["gz"]]
                        for s in self.history])

        accel = arr[:, :3]
        gyro = arr[:, 3:]

        # RMS
        accel_mag = np.linalg.norm(accel - np.mean(accel, axis=0), axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)
        accel_rms = float(np.sqrt(np.mean(accel_mag ** 2)))
        gyro_rms = float(np.sqrt(np.mean(gyro_mag ** 2)))

        # Jerk (rate of change of acceleration)
        jerk = float(np.mean(np.abs(np.diff(accel_mag))))

        # Dominant frequency via FFT
        n = len(accel_mag)
        fft = np.abs(np.fft.rfft(accel_mag - np.mean(accel_mag)))
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        dominant_freq = float(freqs[np.argmax(fft[1:]) + 1]) if n > 2 else 0.0

        # Rep counting — zero-crossings of smoothed accel
        smoothed = np.convolve(accel_mag, np.ones(5) / 5, mode="valid")
        mean_val = np.mean(smoothed)
        crossings = np.where(np.diff(np.sign(smoothed - mean_val)))[0]
        rep_count = max(0, len(crossings) // 2)

        # Step cadence (steps/min)
        step_cadence = dominant_freq * 60 if 0.5 < dominant_freq < 3.5 else 0.0

        # Motion score
        motion_score = min(100.0, accel_rms * 20 + gyro_rms * 5 + jerk * 30)

        # Activity hint
        if accel_rms < 0.15:
            hint = "idle"
        elif 0.8 < dominant_freq < 2.5:
            hint = "walk"
        elif dominant_freq < 0.8 and gyro_rms > 0.8:
            hint = "squat/bend"
        else:
            hint = "active"

        return {
            "motion_score": round(motion_score, 2),
            "jerk": round(jerk, 4),
            "rep_count": rep_count,
            "step_cadence": round(step_cadence, 1),
            "accel_rms": round(accel_rms, 4),
            "gyro_rms": round(gyro_rms, 4),
            "dominant_freq": round(dominant_freq, 3),
            "activity_hint": hint,
        }
