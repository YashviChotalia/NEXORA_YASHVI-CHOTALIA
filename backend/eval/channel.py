"""Helpline-audio simulation shared by the SER evaluation and the telephone-adapted head training.

  trim_silence   cut leading / trailing silence, as the live voice-activity cut does before a window is scored
  telephone      16 kHz -> narrow-band phone line (8 kHz, 300-3400 Hz, G.711 mu-law) -> 16 kHz
  degrade        one of the named CONDITIONS (clean, phone, phone + pink noise at 20 / 10 / 5 dB SNR)
"""

from __future__ import annotations

import numpy as np

SR = 16000
CONDITIONS = ("clean", "phone", "phone_snr20", "phone_snr10", "phone_snr5")
CONDITION_LABEL = {"clean": "Clean 16 kHz (studio)", "phone": "Telephone channel (8 kHz, G.711)",
                   "phone_snr20": "Telephone + noise, 20 dB SNR", "phone_snr10": "Telephone + noise, 10 dB SNR",
                   "phone_snr5": "Telephone + noise, 5 dB SNR"}


def trim_silence(x: np.ndarray, sr: int = SR, pad_s: float = 0.15) -> np.ndarray:
    n = int(0.02 * sr)
    if len(x) < 4 * n:
        return x
    e = np.sqrt(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(e)
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    on = np.where(db > lo + 0.3 * (hi - lo))[0]
    if len(on) == 0:
        return x
    a = max(0, on[0] * n - int(pad_s * sr))
    b = min(len(x), (on[-1] + 1) * n + int(pad_s * sr))
    return x[a:b]


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise, a stand-in for street / room / crowd noise heard on phone calls."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def _mulaw(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711 mu-law: compand to 8 bits and expand back (the quantisation a phone line applies)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu).astype(np.float32)


def telephone(x: np.ndarray, sr: int = SR) -> np.ndarray:
    from scipy.signal import butter, resample_poly, sosfiltfilt
    y = resample_poly(x, 1, 2)
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr // 2, output="sos")
    y = sosfiltfilt(sos, y)
    peak = float(np.max(np.abs(y))) or 1.0
    y = _mulaw(y / peak * 0.9) * peak / 0.9
    return resample_poly(y, 2, 1).astype(np.float32)[: len(x)]


def degrade(x: np.ndarray, condition: str, seed: int) -> np.ndarray:
    if condition == "clean":
        return x
    y = telephone(x)
    if "_snr" in condition:
        snr = float(condition.split("_snr")[1])
        level = max(float(np.sqrt(np.mean(y ** 2))), 1e-4)
        noise = _pink_noise(len(y), np.random.default_rng(seed))
        y = telephone(np.clip(y + noise * level / (10 ** (snr / 20)), -1, 1))  # the noise goes down the line too
    return np.clip(y, -1, 1).astype(np.float32)
