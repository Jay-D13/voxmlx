from contextlib import redirect_stdout
from io import StringIO
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from voxmlx import main
from voxmlx.stream import stream_transcribe, main as stream_main


class StreamBatchTests(unittest.TestCase):
    def replay(self, initial, arrivals=(), batch_ms=80, eos_once=False):
        """Drive the real loop with a cheap model and an event-driven fake mic."""
        chunks, emitted, waits, shown = [], [], [], []
        arrivals = iter(arrivals)
        initial = np.asarray(initial, dtype=np.float32)
        callback = None
        eos_sent = False
        stopped = False
        prefills = []

        def decode(x, *args, **kwargs):
            nonlocal eos_sent
            if x.shape[1] > 1:
                prefills.append(x.shape[1])
            token = 2 if eos_once and not eos_sent else 3
            eos_sent = True
            return mx.array([[[0., 0., float(token == 2), float(token == 3)]]])

        def encode(mel, *state):
            chunks.append(np.array(mel)[0])
            return mx.zeros((mel.shape[1] // 1280, 1)), None, None, [], None

        model = SimpleNamespace(
            time_embedding=lambda _: mx.zeros((1, 1)),
            language_model=SimpleNamespace(layers=[], embed=lambda ids: mx.zeros((*ids.shape, 1))),
            encode_step=encode, decode=decode,
        )
        tokenizer = SimpleNamespace(bos_id=1, eos_id=2, get_special_token=lambda _: 0,
                                    decode=lambda ids, **kwargs: 'word' if ids[0] == 3 else '')

        class Mic:
            def __init__(self, **kwargs):
                nonlocal callback
                callback = kwargs['callback']
            def start(self):
                if len(initial):
                    callback(initial[:, None], len(initial), None, None)
            def stop(self):
                nonlocal stopped
                stopped = True
            def close(self):
                pass

        # Save the original class: patching threading.Condition also affects Thread.
        real_condition = threading.Condition
        class Condition(real_condition):
            def wait(self, timeout=None):
                waits.append(len(chunks))
                shown.append(emitted.count('word'))
                try:
                    data = np.asarray(next(arrivals), dtype=np.float32)
                except StopIteration:
                    raise KeyboardInterrupt
                # Callback notifies while the consumer is waiting; no polling needed.
                producer = threading.Thread(target=lambda: callback(data[:, None], len(data), None, None))
                producer.start()
                return super().wait(timeout)

        with redirect_stdout(StringIO()), patch('voxmlx.stream.load_model', return_value=(model, tokenizer, {})), \
             patch('voxmlx.stream.sd.InputStream', Mic), \
             patch('voxmlx.stream.threading', SimpleNamespace(Condition=Condition)), \
             patch('voxmlx.stream.log_mel_spectrogram_step', side_effect=lambda chunk, tail: (mx.array(chunk)[None, :], tail)):
            stream_transcribe(audio_batch_ms=batch_ms, on_text=emitted.append)
        self.assertTrue(stopped)
        self.shown_at_wait = shown
        return chunks, waits, emitted, prefills

    def test_backlog_is_drained_before_waiting_and_batches_are_capped(self):
        chunks, waits, _, _ = self.replay(np.ones(13 * 1280))
        self.assertEqual(waits, [4])
        self.assertEqual([len(c) for c in chunks[:4]], [33 * 1280, 4 * 1280, 4 * 1280, 4 * 1280])

    def test_newest_prediction_is_shown_before_waiting_for_audio(self):
        # The 39-token prompt fits in 7 audio tokens (32 are padding) and predicts one
        # token; each later position predicts one more. Nothing waits for the next chunk.
        for audio_tokens, predictions in ((7, 1), (13, 7)):
            self.replay(np.ones(audio_tokens * 1280))
            self.assertEqual(self.shown_at_wait, [predictions])

    def test_audio_after_eos_opens_the_next_session(self):
        # The prefill's prediction is EOS; the next arrival must not join the ended session.
        chunks, _, _, prefills = self.replay(np.ones(7 * 1280), arrivals=[np.ones(1280)], eos_once=True)
        self.assertEqual([len(c) // 1280 for c in chunks[:4]], [33, 4, 2, 33])
        self.assertEqual(prefills, [39, 39])  # The new session completes during shutdown.

    def test_wait_notification_and_partial_shutdown(self):
        chunks, waits, text, _ = self.replay(np.ones(1280),
            arrivals=[np.ones(1280), np.ones(1280), np.ones(123)], batch_ms=160)
        self.assertEqual(waits, [1, 1, 2, 2])
        self.assertEqual(len(chunks[1]), 2560)
        self.assertEqual(sum(np.count_nonzero(c) for c in chunks), 3 * 1280 + 123)
        self.assertIn('word', text)  # Shutdown completes prefill even for short captures.

    def test_eos_retains_unencoded_backlog_and_resets_prefix(self):
        chunks, waits, _, prefills = self.replay(np.ones(25 * 1280), eos_once=True)
        self.assertGreaterEqual(len(prefills), 2)
        self.assertTrue(all(n == 39 for n in prefills))
        self.assertEqual(sum(np.count_nonzero(c) for c in chunks), 25 * 1280)
        self.assertGreater(sum(len(c) == 33 * 1280 for c in chunks), 1)

    def test_shutdown_before_initial_80ms_flushes_capture(self):
        chunks, _, text, _ = self.replay(np.ones(123), batch_ms=320)
        self.assertEqual(sum(np.count_nonzero(c) for c in chunks), 123)
        self.assertIn('word', text)

    def test_eos_during_shutdown_retains_raw_audio(self):
        with patch('threading.Condition.wait_for', side_effect=[True, KeyboardInterrupt]):
            chunks, _, _, prefills = self.replay(np.ones(25 * 1280), eos_once=True)
        self.assertEqual(sum(np.count_nonzero(c) for c in chunks), 25 * 1280)
        self.assertGreaterEqual(len(prefills), 2)

    def test_no_audio_shutdown(self):
        chunks, waits, text, _ = self.replay([])
        self.assertEqual(chunks, [])
        self.assertEqual(waits, [0])
        self.assertEqual(text, ['\n'])

    def test_320ms_waits_for_four_tokens_after_startup(self):
        chunks, waits, _, _ = self.replay(np.ones(1280),
            arrivals=[np.ones(1280)] * 4, batch_ms=320)
        self.assertEqual(waits, [1, 1, 1, 1, 2])
        self.assertEqual(len(chunks[1]), 4 * 1280)

    def test_no_audio_warning_uses_bounded_wait(self):
        with patch('voxmlx.stream.time.monotonic', side_effect=[0., 3.]), \
             patch('threading.Condition.wait_for', side_effect=[False, KeyboardInterrupt]) as wait, \
             patch('builtins.print') as output:
            self.replay([])
        self.assertTrue(any('No audio received' in str(c) for c in output.call_args_list))
        self.assertTrue(all(c.kwargs['timeout'] == .25 for c in wait.call_args_list))

    def test_validation_and_both_entrypoints(self):
        with patch('voxmlx.stream.load_model') as load:
            for value in (-80, 0, 81, 240, 640):
                with self.assertRaisesRegex(ValueError, 'audio_batch_ms'):
                    stream_transcribe(audio_batch_ms=value)
            load.assert_not_called()
        for entrypoint in (main, stream_main):
            for batch in (80, 160, 320):
                with patch('sys.argv', ['voxmlx', '--audio-batch-ms', str(batch)]), \
                     patch('voxmlx.stream.stream_transcribe') as stream:
                    entrypoint()
                self.assertEqual(stream.call_args.kwargs['audio_batch_ms'], batch)


if __name__ == '__main__':
    unittest.main()
