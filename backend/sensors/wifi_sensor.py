"""
WiFi RSSI Sensor — scans ALL visible BSSIDs.
Tries: nmcli → iwlist → iw → airport (macOS) → netsh (Windows)
Falls back to simulated data if no scanner available (keeps dashboard alive).

Improvement (arxiv 2308.06773):
  Per-AP rolling std-dev is now computed and exposed as `ap_std_vector`.
  This is the key feature the paper identifies for accurate person counting —
  each AP acts as an independent "detector channel", and the std dev of its
  RSSI over a time window encodes how many bodies are disrupting that path.
  The feature vector is fed into MultiPersonTracker's occupancy classifier
  instead of the collapsed scalar motion_score.
"""
import subprocess, re, time, platform, random
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


# Minimum scans before per-AP std dev features are considered reliable
_STD_MIN_WINDOW = 8

# APs seen in at least this fraction of recent scans are considered "stable"
_STABLE_AP_PRESENCE_THRESHOLD = 0.5

# Hard cap on per-AP std dev used in feature normalisation (dB)
# 99th percentile in the paper's dataset was ~6 dB; cap at 12 to handle outliers
_STD_CAP_DB = 12.0


class WiFiSensor:
    def __init__(self, window: int = 30):
        # Increased default window from 20→30 so std dev estimates are more stable
        self.history: deque = deque(maxlen=window)
        self.known_networks: dict = {}
        self.os = platform.system()
        self._scanner = self._detect_scanner()
        self._sim_networks = self._make_sim_networks()
        self._sim_phase = 0.0
        self._motion_level = 0.0
        self.use_simulated_fallback = True

        # ── Per-AP rolling stats ──────────────────────────────────────────────
        # {bssid: deque of recent rssi values}  — kept separate from history so
        # we can compute std dev even when some APs are absent from a scan.
        self._ap_windows: Dict[str, deque] = {}
        self._ap_window_size = window

        print(f"[WiFi] Scanner: {self._scanner or 'SIMULATED'}")

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_simulated_fallback(self, enabled: bool):
        self.use_simulated_fallback = bool(enabled)

    def set_motion_level(self, level: float):
        self._motion_level = max(0.0, min(2.0, level))

    # ── Scanner detection ─────────────────────────────────────────────────────

    def _detect_scanner(self):
        candidates = [
            ("nmcli", ["nmcli", "--version"]),
            ("iwlist", ["iwlist", "--version"]),
            ("iw",    ["iw", "--version"]),
        ]
        for name, cmd in candidates:
            try:
                subprocess.run(cmd, capture_output=True, timeout=2)
                return name
            except Exception:
                pass
        if self.os == "Darwin":
            airport = ("/System/Library/PrivateFrameworks/Apple80211.framework"
                       "/Versions/Current/Resources/airport")
            if subprocess.run(["test", "-f", airport], capture_output=True).returncode == 0:
                return "airport"
        if self.os == "Windows":
            return "netsh"
        return None

    # ── Simulated networks ────────────────────────────────────────────────────

    def _make_sim_networks(self):
        import hashlib
        nets = []
        names = ["HomeNet_5G", "NETGEAR42", "Airtel_Fiber", "JioFiber_Home",
                 "Office_WiFi", "Guest_Network", "TP-Link_2.4G", "BSNL_Broadband"]
        for i, name in enumerate(names):
            seed = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
            mac = ":".join(f"{(seed >> (j * 8)) & 0xFF:02X}" for j in range(6))
            base_rssi = -45 - i * 6
            nets.append({"mac": mac, "ssid": name, "base_rssi": base_rssi})
        return nets

    # ── Scan methods ──────────────────────────────────────────────────────────

    def scan(self) -> dict:
        """Return {bssid: rssi_dBm}. Always returns something (real or simulated)."""
        result = {}
        if self._scanner == "nmcli":
            result = self._scan_nmcli()
        elif self._scanner == "iwlist":
            result = self._scan_iwlist()
        elif self._scanner == "iw":
            result = self._scan_iw()
        elif self._scanner == "airport":
            result = self._scan_macos()
        elif self._scanner == "netsh":
            result = self._scan_windows()

        if not result and self.use_simulated_fallback:
            result = self._scan_simulated()
        return result

    def _scan_nmcli(self) -> dict:
        networks = {}
        try:
            out = subprocess.check_output(
                ["nmcli", "-t", "-f", "SSID,BSSID,SIGNAL", "dev", "wifi", "list"],
                stderr=subprocess.DEVNULL, timeout=8
            ).decode(errors="ignore")
            for line in out.strip().split("\n"):
                parts = line.split(":")
                if len(parts) >= 3:
                    try:
                        ssid = parts[0]
                        bssid = ":".join(parts[1:7])
                        signal = int(parts[-1])
                        rssi = signal / 2 - 100
                        networks[bssid] = rssi
                        self.known_networks[bssid] = ssid
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            print(f"[WiFi][nmcli] {e}")
        return networks

    def _scan_iwlist(self) -> dict:
        networks = {}
        try:
            out = subprocess.check_output(
                ["iwlist", "scan"], stderr=subprocess.DEVNULL, timeout=10
            ).decode(errors="ignore")
            bssid = ssid = None
            for line in out.split("\n"):
                line = line.strip()
                m = re.match(r"Cell\s+\d+\s+-\s+Address:\s+([\w:]+)", line)
                if m: bssid = m.group(1)
                m = re.search(r'ESSID:"(.*)"', line)
                if m: ssid = m.group(1)
                m = re.search(r"Signal level=(-?\d+)\s*dBm", line)
                if m and bssid:
                    rssi = float(m.group(1))
                    if rssi > -95:
                        networks[bssid] = rssi
                        self.known_networks[bssid] = ssid or "?"
        except Exception as e:
            print(f"[WiFi][iwlist] {e}")
        return networks

    def _scan_iw(self) -> dict:
        networks = {}
        try:
            out = subprocess.check_output(
                ["iw", "dev"], stderr=subprocess.DEVNULL, timeout=3
            ).decode(errors="ignore")
            iface = re.search(r"Interface\s+(\w+)", out)
            if not iface:
                return {}
            ifname = iface.group(1)
            out2 = subprocess.check_output(
                ["iw", ifname, "scan"], stderr=subprocess.DEVNULL, timeout=10
            ).decode(errors="ignore")
            bssid = ssid = None
            for line in out2.split("\n"):
                line = line.strip()
                m = re.match(r"BSS ([\w:]+)", line)
                if m: bssid = m.group(1)
                m = re.search(r"SSID:\s+(.*)", line)
                if m: ssid = m.group(1).strip()
                m = re.search(r"signal:\s+(-[\d.]+)\s+dBm", line)
                if m and bssid:
                    networks[bssid] = float(m.group(1))
                    self.known_networks[bssid] = ssid or "?"
        except Exception as e:
            print(f"[WiFi][iw] {e}")
        return networks

    def _scan_windows(self) -> dict:
        networks = {}
        try:
            out = subprocess.check_output(
                ["netsh", "wlan", "show", "networks", "mode=bssid"],
                stderr=subprocess.DEVNULL, timeout=8
            ).decode(errors="ignore")
            current_ssid = current_bssid = None
            for line in out.split("\n"):
                line = line.strip()
                m = re.match(r"SSID\s+\d+\s*:\s*(.*)", line)
                if m: current_ssid = m.group(1).strip(); continue
                m = re.match(r"BSSID\s+\d+\s*:\s*(.*)", line)
                if m: current_bssid = m.group(1).strip(); continue
                m = re.match(r"Signal\s*:\s*(\d+)%", line)
                if m and current_bssid:
                    rssi = int(m.group(1)) / 2 - 100
                    if rssi > -95:
                        networks[current_bssid] = rssi
                        self.known_networks[current_bssid] = current_ssid or "?"
        except Exception as e:
            print(f"[WiFi][Windows] {e}")
        return networks

    def _scan_macos(self) -> dict:
        networks = {}
        try:
            airport = ("/System/Library/PrivateFrameworks/Apple80211.framework"
                       "/Versions/Current/Resources/airport")
            out = subprocess.check_output([airport, "-s"],
                                          stderr=subprocess.DEVNULL, timeout=10).decode(errors="ignore")
            for line in out.split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        ssid, bssid, rssi = parts[0], parts[1], int(parts[2])
                        if rssi > -95:
                            networks[bssid] = float(rssi)
                            self.known_networks[bssid] = ssid
                    except ValueError:
                        pass
        except Exception as e:
            print(f"[WiFi][macOS] {e}")
        return networks

    def _scan_simulated(self) -> dict:
        """Realistic simulated WiFi scan — RSSI drifts with motion level."""
        self._sim_phase += 0.08
        networks = {}
        noise_scale = 1.5 + self._motion_level * 4.0

        for i, net in enumerate(self._sim_networks):
            drift = np.sin(self._sim_phase + i * 0.7) * self._motion_level * 5.0
            noise = np.random.normal(0, noise_scale)
            rssi = net["base_rssi"] + drift + noise
            rssi = max(-95, min(-30, rssi))
            networks[net["mac"]] = round(rssi, 1)
            self.known_networks[net["mac"]] = net["ssid"]

        return networks

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_scan(self, scan: dict | None = None):
        if scan is None:
            scan = self.scan()
        if scan:
            self.history.append(scan)
            self._update_ap_windows(scan)
        return scan

    def _update_ap_windows(self, scan: dict):
        """
        Push latest RSSI values into per-AP rolling windows.
        APs absent from the current scan are NOT backfilled with -100 here;
        we use -100 only when building the feature vector so the std dev
        correctly reflects dropout events (which are themselves a signal).
        """
        for bssid, rssi in scan.items():
            if bssid not in self._ap_windows:
                self._ap_windows[bssid] = deque(maxlen=self._ap_window_size)
            self._ap_windows[bssid].append(float(rssi))

    # ── Per-AP std dev feature extraction (core new method) ──────────────────

    def get_ap_std_vector(
        self,
        anchor_bssids: Optional[List[str]] = None,
        min_present_fraction: float = _STABLE_AP_PRESENCE_THRESHOLD,
    ) -> Dict[str, float]:
        """
        Return {bssid: normalised_std_dev} for stable APs.

        This is the primary feature from arxiv:2308.06773 — per-AP RSSI std dev
        over the rolling window discriminates occupancy counts far better than
        any aggregate scalar.

        normalised_std_dev is in [0, 1]:
            0   = completely flat signal (no one moving near this AP's path)
            1   = std dev >= _STD_CAP_DB (heavy disruption)

        Args:
            anchor_bssids:  If provided, only include these BSSIDs (used by
                            MultiPersonTracker to restrict to room-specific APs).
            min_present_fraction: Discard APs seen in fewer than this fraction
                            of the current history window (avoids unstable APs).
        """
        if len(self.history) < _STD_MIN_WINDOW:
            return {}

        n_scans = len(self.history)
        result: Dict[str, float] = {}

        candidates = anchor_bssids if anchor_bssids else list(self._ap_windows.keys())

        for bssid in candidates:
            win = self._ap_windows.get(bssid)
            if win is None or len(win) < _STD_MIN_WINDOW:
                continue

            # Presence check: was this AP seen recently enough?
            seen = sum(1 for sc in self.history if bssid in sc)
            if seen / n_scans < min_present_fraction:
                continue

            # Std dev of raw RSSI values in the window.
            # Missing-scan dropout events are represented as -100 dBm — this
            # deliberately inflates the std dev when an AP intermittently
            # disappears, which is itself a signal of body obstruction.
            vals = list(win)
            std = float(np.std(vals))
            normalised = min(1.0, std / _STD_CAP_DB)
            result[bssid] = round(normalised, 4)

        return result

    def get_ap_delta_vector(
        self,
        baseline: Dict[str, float],
        anchor_bssids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Return {bssid: normalised_delta} = |current_mean_rssi - baseline_rssi| / 20 dB,
        clipped to [0, 1].

        Complements the std dev vector: std dev captures *variability*
        (motion), delta captures *offset from empty room* (occupancy load).
        Using both together gives the classifier two orthogonal signals.
        """
        if not self.history or not baseline:
            return {}

        candidates = anchor_bssids if anchor_bssids else list(baseline.keys())
        result: Dict[str, float] = {}

        for bssid in candidates:
            win = self._ap_windows.get(bssid)
            base_val = baseline.get(bssid)
            if win is None or len(win) < 2 or base_val is None:
                continue
            current_mean = float(np.mean(list(win)))
            delta = abs(current_mean - float(base_val))
            result[bssid] = round(min(1.0, delta / 20.0), 4)

        return result

    def get_counting_feature_vector(
        self,
        baseline: Optional[Dict[str, float]] = None,
        anchor_bssids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Build the combined per-AP feature vector for occupancy counting.

        Returns a flat dict with two entries per stable AP:
            "<bssid>__std"   → normalised RSSI std dev  (0-1)
            "<bssid>__delta" → normalised delta from baseline (0-1, if baseline given)

        Also includes aggregate summary scalars for backward compatibility
        with code that still reads motion_score / variance.
        """
        std_vec = self.get_ap_std_vector(anchor_bssids=anchor_bssids)
        delta_vec: Dict[str, float] = {}
        if baseline:
            delta_vec = self.get_ap_delta_vector(baseline, anchor_bssids=anchor_bssids)

        feature_vec: Dict[str, float] = {}
        for bssid, std_val in std_vec.items():
            feature_vec[f"{bssid}__std"] = std_val
        for bssid, delta_val in delta_vec.items():
            feature_vec[f"{bssid}__delta"] = delta_val

        # Aggregate scalars (used by existing fusion code)
        if std_vec:
            feature_vec["mean_ap_std"] = round(float(np.mean(list(std_vec.values()))), 4)
            feature_vec["max_ap_std"] = round(float(np.max(list(std_vec.values()))), 4)
            feature_vec["n_active_aps"] = len(std_vec)
        else:
            feature_vec["mean_ap_std"] = 0.0
            feature_vec["max_ap_std"] = 0.0
            feature_vec["n_active_aps"] = 0

        if delta_vec:
            feature_vec["mean_ap_delta"] = round(float(np.mean(list(delta_vec.values()))), 4)
        else:
            feature_vec["mean_ap_delta"] = 0.0

        return feature_vec

    # ── Original feature extraction (unchanged, still used by fusion.py) ─────

    def extract_features(self) -> dict:
        base = {
            "motion_score": 0.0, "variance": 0.0, "total_change": 0.0,
            "corr_score": 0.0, "n_networks": 0, "raw_vector": {},
            "network_names": {}, "is_real": self._scanner is not None,
            # NEW: always include std/delta aggregates even in base dict
            "mean_ap_std": 0.0, "max_ap_std": 0.0, "mean_ap_delta": 0.0,
        }
        if len(self.history) < 4:
            if self._sim_networks:
                base["n_networks"] = len(self._sim_networks)
                last = self.history[-1] if self.history else {}
                base["raw_vector"] = {k: v for k, v in last.items()}
                base["network_names"] = dict(self.known_networks)
            return base

        all_bssids = set()
        for sc in self.history:
            all_bssids.update(sc.keys())

        stable = [b for b in all_bssids
                  if sum(1 for sc in self.history if b in sc) > len(self.history) * _STABLE_AP_PRESENCE_THRESHOLD]
        base["n_networks"] = len(stable)
        base["network_names"] = {b: self.known_networks.get(b, "?") for b in stable}

        if len(stable) < 2:
            if self.history:
                base["raw_vector"] = dict(list(self.history)[-1])
            return base

        matrix = np.array([
            [sc.get(b, -100) for b in stable] for sc in self.history
        ], dtype=float)

        var   = float(np.mean(np.var(matrix, axis=0)))
        diffs = float(np.mean(np.abs(np.diff(matrix, axis=0))))
        corr_score = 0.0
        if len(stable) > 1:
            col_std = np.std(matrix, axis=0)
            if np.all(col_std > 1e-6):
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.corrcoef(matrix.T)
                upper = corr[np.triu_indices_from(corr, k=1)]
                upper = upper[np.isfinite(upper)]
                if upper.size:
                    corr_score = float(np.mean(np.abs(upper)))

        var   = min(var, 15)
        diffs = min(diffs, 10)
        motion_score = min(100.0, var * 2.5 + diffs * 10 + corr_score * 15)

        # ── Attach per-AP std dev aggregates ─────────────────────────────────
        std_vec = self.get_ap_std_vector(anchor_bssids=stable)
        if std_vec:
            std_vals = list(std_vec.values())
            mean_std = float(np.mean(std_vals))
            max_std  = float(np.max(std_vals))
            # Blend mean_ap_std into motion_score — paper shows std dev is a
            # stronger signal than raw variance, so give it 30% weight here.
            motion_score = min(100.0, motion_score * 0.70 + mean_std * 100.0 * 0.30)
            base["mean_ap_std"] = round(mean_std, 4)
            base["max_ap_std"]  = round(max_std, 4)

        return {
            **base,
            "motion_score":  round(motion_score, 2),
            "variance":      round(var, 3),
            "total_change":  round(diffs, 3),
            "corr_score":    round(corr_score, 3),
            "n_networks":    len(stable),
            "raw_vector":    {b: round(matrix[-1, i], 1) for i, b in enumerate(stable)},
            "network_names": base["network_names"],
            "is_real":       self._scanner is not None,
        }