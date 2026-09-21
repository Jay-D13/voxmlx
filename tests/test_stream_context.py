import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from voxmlx.model import VoxtralRealtime
from voxmlx.stream import stream_transcribe
from voxmlx import _build_prompt_tokens, main


class StreamContextTests(unittest.TestCase):
    def test_context_must_fit_prompt(self):
        with self.assertRaisesRegex(ValueError, "at least 39"):
            stream_transcribe(context_size=38)

    def test_quality_delay_must_fit_prompt(self):
        with self.assertRaisesRegex(ValueError, "at least 63"):
            stream_transcribe(context_size=62, delay_ms=2400)

    def test_invalid_delay_fails_before_loading_model(self):
        with patch("voxmlx.stream.load_model") as load:
            for delay in (0, -80, 100, 1280, 2480):
                with self.subTest(delay=delay), self.assertRaises(ValueError):
                    stream_transcribe(delay_ms=delay)
            load.assert_not_called()

    def test_quality_prompt_contains_delay_tokens(self):
        tokenizer = SimpleNamespace(bos_id=1, get_special_token=lambda _: 7)
        tokens, delay = _build_prompt_tokens(tokenizer, num_delay_tokens=30)
        self.assertEqual(delay, 30)
        self.assertEqual(tokens, [1] + [7] * 62)

    def test_cli_forwards_quality_settings(self):
        with patch("sys.argv", ["voxmlx", "--context-size", "1024", "--delay-ms", "2400"]):
            with patch("voxmlx.stream.stream_transcribe") as stream:
                main()
        self.assertEqual(stream.call_args.kwargs["context_size"], 1024)
        self.assertEqual(stream.call_args.kwargs["delay_ms"], 2400)

    def test_encoder_cache_uses_configured_window(self):
        model = VoxtralRealtime({
            "dim": 8, "n_layers": 1, "n_heads": 1, "n_kv_heads": 1,
            "head_dim": 8, "hidden_dim": 16, "vocab_size": 64,
            "rope_theta": 10000,
            "multimodal": {"whisper_model_args": {
                "encoder_args": {
                    "audio_encoding_args": {"num_mel_bins": 4},
                    "dim": 8, "n_layers": 1, "n_heads": 1, "head_dim": 8,
                    "hidden_dim": 16, "rope_theta": 10000, "sliding_window": 8,
                },
                "downsample_args": {"downsample_factor": 4},
            }},
        })
        conv1 = conv2 = cache = remainder = None
        for _ in range(20):
            embeds, conv1, conv2, cache, remainder = model.encode_step(
                mx.zeros((4, 8)), conv1, conv2, cache, remainder
            )
            if embeds is not None:
                mx.eval(embeds)
            self.assertEqual(cache[0].max_size, 8)
            # Chunked updates retain the window plus the current chunk.
            self.assertLessEqual(cache[0].keys.shape[2], 11)
        self.assertGreater(cache[0].offset, 8)


if __name__ == "__main__":
    unittest.main()
