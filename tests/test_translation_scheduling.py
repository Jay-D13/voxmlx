from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from queue import Empty
import tempfile
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from voxmlx import main
from voxmlx.translation import LiveTranslation, _Pending, load_french_english


class SchedulingTests(unittest.TestCase):
    def run_replay(self, records, idle=1.5, cost=None, translations=None):
        """Replay (text, arrival) records against a simulated clock; English is uppercase."""
        clock, waits, events, self.calls = [0.], [], [], []
        records = list(records)

        class Queue:
            def get(self, timeout=None):
                waits.append(timeout)
                if len(waits) > 100:
                    raise AssertionError('Scheduler is spinning')
                text, arrived = records[0]
                deadline = float('inf') if timeout is None else clock[0] + timeout
                if arrived > deadline:
                    clock[0] = deadline
                    raise Empty
                records.pop(0)
                clock[0] = max(clock[0], arrived)
                return text, arrived

            def empty(self):
                return not records or records[0][1] > clock[0]

        def translate(text, prefix=''):
            self.calls.append((text, prefix))
            clock[0] += (cost or {}).get(text, 0)
            english = (translations or {}).get(text, text.upper())
            assert english.startswith(prefix), (english, prefix)
            return english

        live = LiveTranslation(translate, idle, on_timing=events.append)
        live.queue = Queue()
        output = StringIO()
        with redirect_stdout(output), patch('voxmlx.translation.time', SimpleNamespace(monotonic=lambda: clock[0])):
            live._run()
        self.output = output.getvalue()
        return events, waits

    def test_idle_wait_and_no_pending_blocks_indefinitely(self):
        events, waits = self.run_replay([('Bonjour', 0), (None, 3)])
        self.assertEqual(events[0]['ready_at'], 1.5)
        self.assertEqual(events[0]['reason'], 'idle')
        self.assertEqual(waits, [None, 1.5, None])

    def test_english_streams_once_consecutive_translations_agree(self):
        events, _ = self.run_replay([(word + ' ', i) for i, word in enumerate(
            ('un', 'deux', 'trois', 'quatre'))] + [('cinq', 4), (None, 8)])
        self.assertEqual(self.output, 'UN DEUX TROIS QUATRE CINQ\n')
        # "UN" is shown when the second translation confirms it, not at the pause.
        self.assertEqual(events[0]['first_english'], 1)
        self.assertEqual((events[0]['reason'], events[0]['ready_at']), ('idle', 5.5))
        # The completed sentence continues the English already shown.
        self.assertEqual(self.calls[-1], ('un deux trois quatre cinq', 'UN DEUX TROIS'))

    def test_disagreeing_words_wait_and_shown_words_are_kept(self):
        translations = {'le': 'the', 'le chat': 'the cat.', 'le chat noir': 'the black cat.',
                        'le chat noir dort': 'the black cat sleeps.'}
        self.run_replay([('le ', 0), ('chat ', 1), ('noir ', 2), ('dort', 3), (None, 9)],
                        translations=translations)
        # "cat" was never agreed before "black", so it was not shown early.
        self.assertEqual(self.output, 'the black cat sleeps.\n')
        self.assertEqual(self.calls[-1], ('le chat noir dort', 'the'))

    def test_backlog_uses_arrival_order_not_processing_time(self):
        events, _ = self.run_replay([('Bonjour. ', 0), ('un ', .1), ('deux', 1),
                                     ('trois', 3), (None, 6)], cost={'Bonjour.': 4})
        self.assertEqual([e['source'] for e in events], ['Bonjour.', 'un deux', 'trois'])
        self.assertEqual(events[1]['ready_at'], 2.5)
        self.assertEqual(events[1]['inference_start'], 4)
        self.assertEqual(events[2]['ready_at'], 4.5)
        # Queued fragments were not retranslated one by one while the worker was behind.
        self.assertEqual([c for c, _ in self.calls], ['Bonjour.', 'un deux', 'trois'])

    def test_fragments_before_at_and_after_deadline(self):
        for arrival, expected in [(1.49, ['ab']), (1.5, ['a', 'b']), (1.51, ['a', 'b'])]:
            with self.subTest(arrival=arrival):
                events, _ = self.run_replay([('a', 0), ('b', arrival), (None, 4)])
                self.assertEqual([e['source'] for e in events], expected)

    def test_pending_keeps_arrivals_and_whole_words(self):
        pending = _Pending(10)
        pending.append('bonjour ', 0)
        pending.append('mon', 2)
        self.assertEqual(pending.complete(), 'bonjour')
        self.assertEqual(pending.deadline(), 12)
        self.assertIsNone(pending.ready(11))
        self.assertEqual(pending.ready(12), (11, 'idle', 12))
        pending.take(8)
        self.assertEqual((pending.arrivals, pending.complete()), ([2, 2, 2], ''))

    def test_long_word_has_no_busy_loop(self):
        events, waits = self.run_replay([('x' * 300, 0), (None, 12)], idle=10)
        self.assertEqual(waits, [None, 10, None])
        self.assertEqual(events[0]['source'], 'x' * 300)
        self.assertEqual(events[0]['reason'], 'idle')

    def test_length_punctuation_shutdown_preserve_words_and_order(self):
        source = 'Bonjour.\n' + 'bonjour ' * 40 + 'fin'
        events, _ = self.run_replay([(source[i:i+11], i / 100) for i in range(0, len(source), 11)] + [(None, 4)])
        self.assertEqual(' '.join(e['source'] for e in events).split(), source.split())
        self.assertIn('length', [e['reason'] for e in events])
        self.assertEqual(events[-1]['reason'], 'shutdown')
        self.assertEqual(self.output.split(), source.upper().split())
        empty, _ = self.run_replay([(None, 0)])
        self.assertEqual(empty, [])

    def test_actual_worker_does_not_wait_again_after_slow_translation(self):
        entered, release, finished = Event(), Event(), Event()
        events = []

        def translate(text, prefix=''):
            if text == 'Bonjour.':
                entered.set()
                self.assertTrue(release.wait(2))
            elif text == 'Au revoir':
                finished.set()
            return text

        with redirect_stdout(StringIO()), LiveTranslation(translate, idle_seconds=.2, on_timing=events.append) as live:
            live.write('Bonjour. ')
            self.assertTrue(entered.wait(2))
            live.write('Au revoir')
            time.sleep(.25)
            release.set()
            self.assertTrue(finished.wait(.15))
        self.assertEqual(events[1]['reason'], 'idle')
        self.assertGreater(events[1]['inference_start'] - events[1]['ready_at'], .04)

    def test_cli_validation_and_forwarding(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with patch('sys.argv', ['voxmlx', '--translation-idle-ms', value]), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    main()
        with patch('sys.argv', ['voxmlx', '--translate-en', '--translation-idle-ms', '750']), \
             patch('voxmlx.translation.load_french_english', return_value=str), \
             patch('voxmlx.translation.LiveTranslation') as live, \
             patch('voxmlx.stream.stream_transcribe'):
            main()
            live.assert_called_once_with(str, idle_seconds=.75, record=None)
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LiveTranslation(str, idle_seconds=timeout)

    def test_offline_loader_requires_cached_model_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'VOXMLX_CACHE_DIR': directory}), \
             patch.dict('sys.modules', {'ctranslate2': SimpleNamespace()}), \
             patch('voxmlx.translation.urllib.request.urlopen') as urlopen, \
             patch('voxmlx.translation.FrenchEnglish') as model:
            with self.assertRaisesRegex(RuntimeError, 'Cached French-to-English'):
                load_french_english(allow_download=False)
            (Path(directory) / 'translate-fr_en-1_9' / 'model').mkdir(parents=True)
            self.assertIs(load_french_english(allow_download=False, warmup=False), model.return_value)
            model.return_value.assert_not_called()
        urlopen.assert_not_called()

    def test_timing_callback_failure_still_drains_text(self):
        chunks = []
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(RuntimeError):
                with LiveTranslation(lambda t, prefix='': chunks.append(t) or t,
                                     on_timing=Mock(side_effect=ValueError('broken'))) as live:
                    live.write('Bonjour.\nAu revoir')
        self.assertEqual((chunks[0], chunks[-1]), ('Bonjour.', 'Au revoir'))
