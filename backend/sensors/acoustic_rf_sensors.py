"""
Acoustic Sensor (speaker + mic echo) — simulated.
Ambient RF Sensor (RTL-SDR Doppler proxy) — simulated.
"""
import numpy as np
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
#  Acoustic Sensor
# ══════════════════════════════════════════════════════════════════════════════

class AcousticSensor:
    """
    Simulates echo-based motion detection.
    A speaker emits an inaudible chirp; the mic captures reflections.
    Movement causes Doppler shifts & multipath changes.
    """
    def __init__(self, window: int = 30, sample_rate: float = 50.0):
        self.window = window
        self.sample_rate = sample_rate
        self.history: deque = deque(maxlen=window)
        self._phase = 0.0
        self._motion_level = 0.0  # 0–2

    def set_motion_level(self, level: float):
        self._motion_level = max(0.0, min(2.0, level))

    def read(self) -> dict:
        self._phase += 0.1
        # Simulate echo delay variance and amplitude modulation
        base_echo = 0.85 + np.sin(self._phase * 0.3) * 0.05
        motion_distortion = self._motion_level * 0.15 * np.sin(self._phase * 2.1)
        noise = np.random.normal(0, 0.02 + self._motion_level * 0.03)

        echo_amplitude = base_echo + motion_distortion + noise
        doppler_shift = self._motion_level * 12.0 * np.sin(self._phase * 1.5) + np.random.normal(0, 2)
        phase_diff = np.random.normal(0, 5 + self._motion_level * 15)

        sample = {
            "echo_amplitude": round(float(np.clip(echo_amplitude, 0, 1)), 4),
            "doppler_shift_hz": round(float(doppler_shift), 3),
            "phase_diff_deg": round(float(phase_diff), 2),
        }
        self.history.append(sample)
        return sample

    def extract_features(self) -> dict:
        if len(self.history) < 5:
            return {"motion_score": 0.0, "echo_variance": 0.0, "doppler_mean": 0.0}

        arr = np.array([
            [s["echo_amplitude"], s["doppler_shift_hz"], s["phase_diff_deg"]]
            for s in self.history
        ])

        echo_var = float(np.var(arr[:, 0]))
        doppler_mean = float(np.mean(np.abs(arr[:, 1])))
        phase_var = float(np.var(arr[:, 2]))

        motion_score = min(100.0, echo_var * 800 + doppler_mean * 3 + phase_var * 0.5)

        return {
            "motion_score": round(motion_score, 2),
            "echo_variance": round(echo_var, 6),
            "doppler_mean": round(doppler_mean, 3),
            "phase_variance": round(phase_var, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Ambient RF Sensor  (RTL-SDR simulated)
# ══════════════════════════════════════════════════════════════════════════════

class AmbientRFSensor:
    """
    Simulates RTL-SDR ambient RF measurements.
    Monitors TV/FM broadcast bands — human motion causes multipath fading.
    FFT features + Doppler proxy extracted from IQ samples.
    """
    def __init__(self, window: int = 30, fft_bins: int = 64):
        self.window = window
        self.fft_bins = fft_bins
        self.history: deque = deque(maxlen=window)
        self._phase = 0.0
        self._motion_level = 0.0

    def set_motion_level(self, level: float):
        self._motion_level = max(0.0, min(2.0, level))

    def read(self) -> dict:
        self._phase += 0.08
        n = self.fft_bins

        # Simulate power spectral density
        base_psd = np.random.exponential(1.0, n)
        # Add motion-induced peaks (Doppler bumps)
        if self._motion_level > 0.1:
            doppler_bin = int(n * 0.15 + np.sin(self._phase) * n * 0.05)
            doppler_bin = max(1, min(n - 2, doppler_bin))
            base_psd[doppler_bin] += self._motion_level * 5.0
            base_psd[doppler_bin - 1] += self._motion_level * 2.0
            base_psd[doppler_bin + 1] += self._motion_level * 2.0

        psd_db = 10 * np.log10(base_psd + 1e-9) - 60  # normalize to ~-60 dBm

        # Doppler proxy: centroid shift
        freqs = np.linspace(-500, 500, n)  # Hz around center freq
        power_norm = np.abs(base_psd) / (np.sum(np.abs(base_psd)) + 1e-9)
        doppler_proxy = float(np.sum(freqs * power_norm))

        # Spectral entropy
        spectral_entropy = float(-np.sum(power_norm * np.log(power_norm + 1e-9)))

        sample = {
            "doppler_proxy_hz": round(doppler_proxy, 2),
            "spectral_entropy": round(spectral_entropy, 4),
            "peak_power_db": round(float(np.max(psd_db)), 2),
            "mean_power_db": round(float(np.mean(psd_db)), 2),
            "psd_snapshot": psd_db.tolist(),
        }
        self.history.append(sample)
        return sample

    def extract_features(self) -> dict:
        if len(self.history) < 5:
            return {
                "motion_score": 0.0, "doppler_variance": 0.0,
                "entropy_change": 0.0, "power_fluctuation": 0.0,
            }

        arr = np.array([
            [s["doppler_proxy_hz"], s["spectral_entropy"], s["peak_power_db"]]
            for s in self.history
        ])

        doppler_var = float(np.var(arr[:, 0]))
        entropy_change = float(np.mean(np.abs(np.diff(arr[:, 1]))))
        power_fluct = float(np.var(arr[:, 2]))

        motion_score = min(100.0, doppler_var * 0.05 + entropy_change * 20 + power_fluct * 2)

        return {
            "motion_score": round(motion_score, 2),
            "doppler_variance": round(doppler_var, 3),
            "entropy_change": round(entropy_change, 4),
            "power_fluctuation": round(power_fluct, 3),
        }
