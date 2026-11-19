import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

spec = importlib.util.spec_from_file_location(
    'accuracy', Path(__file__).resolve().parents[1] / 'benchmarks/accuracy.py')
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)


class AccuracyBenchmarkTests(unittest.TestCase):
    def test_normalize(self):
        cases = {
            "Bonjour, je m'appelle Claire.": 'bonjour je m appelle claire',
            'l’homme': 'l homme',
            'Saint-Étienne': 'saint étienne',
            '« Les Misérables » — dit-il…': 'les misérables dit il',
            '3,5 %': '3 5',
            'été': 'été',  # Decomposed accents compose.
            ' Où\n est-il ?  ': 'où est il',
            'Le cœur': 'le cœur',
            '': '',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(accuracy.normalize(text), expected)

    def test_edit_distance(self):
        self.assertEqual(accuracy.edit_distance('le chat dort'.split(), 'le chien dort bien'.split()), 2)
        self.assertEqual(accuracy.edit_distance('le chat dort'.split(), []), 3)  # Empty hypothesis.
        self.assertEqual(accuracy.edit_distance('chat', 'chien'), 3)
        self.assertEqual(accuracy.edit_distance('', ''), 0)

    def test_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'a.wav').write_bytes(b'')
            manifest = root / 'data/manifest.jsonl'
            manifest.write_text(json.dumps({'id': 'a', 'audio': '../a.wav', 'text': 'x'}) + '\n\n')
            row, = accuracy.read_manifest(manifest)
            self.assertEqual(row['audio'], str((root / 'a.wav').resolve()))
            manifest.write_text(json.dumps({'id': 'b', 'audio': 'b.wav', 'text': 'x'}))
            with self.assertRaisesRegex(FileNotFoundError, 'Audio for b'):
                accuracy.read_manifest(manifest)

    def test_stream_replay_runs_the_live_loop_to_the_final_flush(self):
        chunks = []

        def encode(mel, *state):
            chunks.append(np.array(mel)[0])
            return mx.zeros((mel.shape[1] // 1280, 1)), None, None, [], None

        model = SimpleNamespace(
            time_embedding=lambda _: mx.zeros((1, 1)),
            language_model=SimpleNamespace(layers=[], embed=lambda ids: mx.zeros((*ids.shape, 1))),
            encode_step=encode, decode=lambda *a, **k: mx.array([[[0., 0., 0., 1.]]]))
        tokenizer = SimpleNamespace(bos_id=1, eos_id=2, get_special_token=lambda _: 0,
                                    decode=lambda ids, **kwargs: ' mot')
        audio = np.arange(1, 10 * 1280 + 101, dtype=np.float32)
        device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            # The fake mel passes samples through, so chunk sizes are visible.
            with patch('voxmlx.stream.log_mel_spectrogram_step',
                       side_effect=lambda chunk, tail: (mx.array(chunk)[None, :], tail)):
                text = accuracy.replay_stream((model, tokenizer, {}), audio, model_path='/m', delay_ms=480,
                                              audio_batch_ms=160, context_size=512)
        finally:
            mx.set_default_device(device)
        # In tokens: 80 ms plus the 32-token left pad, 160 ms batches, then the flush.
        self.assertEqual([len(c) // 1280 for c in chunks], [33, 2, 2, 2, 2, 4, 4, 4, 4, 3])
        fed = np.concatenate(chunks)
        np.testing.assert_array_equal(fed[fed != 0], audio)
        self.assertTrue(text.startswith(' mot mot'))
        self.assertTrue(text.endswith('\n'))


if __name__ == '__main__':
    unittest.main()
