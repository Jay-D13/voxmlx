import os
import unittest

import mlx.core as mx
import numpy as np

from voxmlx.audio import (
    GLOBAL_LOG_MEL_MAX, HOP_LENGTH, N_FFT, log_mel_spectrogram,
    log_mel_spectrogram_step, mel_filter_bank,
)


def reference(audio, full=False):
    if full:
        audio = np.pad(audio, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP_LENGTH]
    if full:
        frames = frames[:-1]
    spectrum = np.fft.rfft(frames * np.hanning(N_FFT + 1)[:-1])
    mel = np.abs(spectrum) ** 2 @ mel_filter_bank().T
    log = np.log10(np.maximum(mel, 1e-10))
    return ((np.maximum(log, GLOBAL_LOG_MEL_MAX - 8) + 4) / 4).T


# M5 defaults to reduced-precision float32 matmul; test both runtime modes.
ATOL = 1e-4 if os.environ.get("MLX_ENABLE_TF32") == "0" else 3e-4


class AudioTests(unittest.TestCase):
    def test_fft_matches_numpy(self):
        n = 16000
        impulse = np.zeros(n, dtype=np.float32)
        impulse[[0, 160, 400, 8000, n - 1]] = 1
        signals = [np.zeros(n), impulse,
                   np.sin(2 * np.pi * 440 * np.arange(n) / 16000),
                   np.random.default_rng(42).normal(0, .1, n)]
        for index, signal in enumerate(signals):
            signal = signal.astype(np.float32)
            with self.subTest(signal=index):
                actual = np.array(log_mel_spectrogram(signal))
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, reference(signal, full=True), atol=ATOL, rtol=3e-5)
                for count in (1280, 2560, 5120):
                    tail = None
                    for start in range(0, n - count + 1, count):
                        chunk = signal[start:start + count]
                        combined = np.concatenate([np.zeros(200, np.float32) if tail is None else tail, chunk])
                        mel, tail = log_mel_spectrogram_step(chunk, tail)
                        np.testing.assert_allclose(np.array(mel), reference(combined), atol=ATOL, rtol=3e-5)
                        np.testing.assert_array_equal(tail, combined[-240:])
                        self.assertTrue(np.isfinite(np.array(mel)).all())

    def test_short_chunk(self):
        mel, tail = log_mel_spectrogram_step(np.zeros(80, np.float32), None)
        self.assertEqual(mel.shape, (128, 0))
        self.assertEqual(tail.shape, (240,))


if __name__ == '__main__':
    unittest.main()
