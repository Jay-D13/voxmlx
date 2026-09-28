import os
import unittest

import mlx.core as mx
import numpy as np

from voxmlx.audio import log_mel_spectrogram_step
from voxmlx.cache import RotatingKVCache
from voxmlx.encoder import CausalWhisperEncoder
from voxmlx.model import VoxtralRealtime


# Strict oracle checks run with MLX_ENABLE_TF32=0; normal M5 kernels use TF32.
ATOL = 1e-4 if os.environ.get("MLX_ENABLE_TF32") == "0" else 3e-3


def tiny_model(sliding_window):
    model = VoxtralRealtime({
        'dim': 16, 'n_layers': 1, 'n_heads': 2, 'n_kv_heads': 1,
        'head_dim': 8, 'hidden_dim': 32, 'vocab_size': 64, 'rope_theta': 10000,
        'multimodal': {'whisper_model_args': {
            'encoder_args': {'audio_encoding_args': {'num_mel_bins': 128},
                             'dim': 16, 'n_layers': 2, 'n_heads': 2, 'head_dim': 8,
                             'hidden_dim': 32, 'rope_theta': 10000,
                             'sliding_window': sliding_window},
            'downsample_args': {'downsample_factor': 4}}},
    })
    # Convolutions initialize to zero; use nonzero weights to test state handling.
    for conv in (model.encoder.conv1, model.encoder.conv2):
        conv.weight = mx.random.normal(conv.weight.shape) * .03
    return model


class EncoderBatchTests(unittest.TestCase):
    def test_window_matches_full_reference_across_partitions(self):
        mx.random.seed(7)
        encoder = CausalWhisperEncoder(dim=16, n_layers=2, n_heads=2, head_dim=8,
                                       hidden_dim=32, sliding_window=17)
        x = mx.random.normal((1, 212, 16))
        positions = mx.arange(x.shape[1])
        distance = positions[:, None] - positions[None, :]
        mask = (distance >= 0) & (distance < encoder.sliding_window)
        expected = x
        for layer in encoder.layers:
            expected = layer(expected, offset=0, mask=mask)
        expected = np.array(encoder.norm(expected))
        for chunk_size in (1, 4, 8, 16):
            with self.subTest(chunk_size=chunk_size):
                cache = [RotatingKVCache(17) for _ in encoder.layers]
                # Startup is larger than the window; subsequent calls wrap it repeatedly.
                outputs = [encoder.forward_transformer(x[:, :132], cache)]
                for start in range(132, x.shape[1], chunk_size):
                    outputs.append(encoder.forward_transformer(x[:, start:start + chunk_size], cache))
                    mx.eval(outputs[-1])
                actual = np.array(mx.concatenate(outputs, axis=1))
                np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=1e-4)
                self.assertEqual(cache[0].offset, 212)
                self.assertLessEqual(cache[0].keys.shape[2], 17 + chunk_size - 1)

    def test_audio_conv_downsampling_and_adapter_are_partition_independent(self):
        mx.random.seed(9)
        model = tiny_model(sliding_window=17)
        audio = np.random.default_rng(10).normal(0, .1, 33 * 1280).astype(np.float32)
        outputs = []
        for tokens in (1, 2, 4):
            tail = c1 = c2 = cache = ds = None
            chunks = [np.concatenate([np.zeros(32 * 1280, np.float32), audio[:1280]])]
            chunks += [audio[i:i + tokens * 1280] for i in range(1280, len(audio), tokens * 1280)]
            result = []
            for chunk in chunks:
                mel, tail = log_mel_spectrogram_step(chunk, tail)
                out, c1, c2, cache, ds = model.encode_step(mel, c1, c2, cache, ds)
                mx.eval(out)
                result.append(out)
            outputs.append(np.array(mx.concatenate(result)))
        for actual in outputs[1:]:
            np.testing.assert_allclose(actual, outputs[0], atol=ATOL, rtol=1e-4)

    def test_file_encode_matches_streaming_beyond_window(self):
        mx.random.seed(11)
        model = tiny_model(sliding_window=750)
        # 1648 encoder frames: more than twice the window and not a multiple of it.
        # A multiple of 8 mel frames, so encode() trims nothing for stride/downsampling.
        mel = mx.random.normal((128, 3296))
        expected = np.array(model.encode(mel))

        # Streaming: a startup chunk (left padding plus one token), then 80 ms chunks.
        c1 = c2 = cache = ds = None
        result = []
        for start, stop in [(0, 264)] + [(s, s + 8) for s in range(264, mel.shape[1], 8)]:
            out, c1, c2, cache, ds = model.encode_step(mel[:, start:stop], c1, c2, cache, ds)
            mx.eval(out)
            result.append(out)
        actual = np.array(mx.concatenate(result))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=1e-4)

        # Guard that the fixture exercises the window: unwindowed causal attention
        # agrees for the first 750 frames only.
        unwindowed = np.array(model.encoder.forward_transformer(model.encoder.forward_conv(mel)))
        windowed = np.array(model.encoder(mel))
        np.testing.assert_allclose(windowed[:, :750], unwindowed[:, :750], atol=ATOL, rtol=1e-4)
        self.assertGreater(np.abs(windowed[:, 750:] - unwindowed[:, 750:]).max(), 10 * ATOL)


if __name__ == '__main__':
    unittest.main()
