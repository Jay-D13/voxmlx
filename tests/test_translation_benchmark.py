import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('translation_benchmark', Path(__file__).resolve().parents[1] / 'benchmarks/translation.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class TranslationBenchmarkTests(unittest.TestCase):
    def test_fixture_validation(self):
        self.assertIsNone(benchmark.read_fixture()[-1]['text'])
        invalid = [[], [None], [{'at_ms': 0, 'text': '  '}, {'at_ms': 1, 'text': None}], [{'at_ms': 0, 'text': 'bonjour'}],
                   [{'at_ms': -1, 'text': 'a'}, {'at_ms': 1, 'text': None}],
                   [{'at_ms': 2, 'text': 'a'}, {'at_ms': 1, 'text': None}],
                   [{'at_ms': 0, 'text': None}, {'at_ms': 1, 'text': 'a'}],
                   [{'at_ms': float('nan'), 'text': 'a'}, {'at_ms': 1, 'text': None}],
                   [{'at_ms': 0, 'text': 123}, {'at_ms': 1, 'text': None}]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.jsonl'
            for rows in invalid:
                path.write_text('\n'.join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    benchmark.read_fixture(path)
            path.write_text('{"at_ms":0,"text":"bonjour"}\n{"at_ms":1,"text":null}')
            self.assertEqual(len(benchmark.read_fixture(path)), 2)

    def test_metrics_and_json_preserve_unavailable_values(self):
        event = dict(first_arrival=1, ready_at=2, inference_start=3, inference_end=4, output_end=5)
        metrics = benchmark.phrase_metrics([event])
        self.assertEqual(metrics['buffering_ms']['median'], 1000)
        self.assertEqual(metrics['ready_wait_ms']['p95'], 1000)
        self.assertEqual(metrics['output_latency_ms']['median'], 4000)
        baseline = benchmark.phrase_metrics([dict(inference_start=1, inference_end=2)])
        self.assertIsNone(baseline['buffering_ms']['median'])
        rows = [dict(variant='baseline', threads=2, beam=4, idle_ms=1500, metrics=baseline, failure=None)]
        result = json.loads(json.dumps(benchmark.summarize(rows)))
        self.assertIsNone(result[0]['first_english_ms'])
        self.assertEqual(result[0]['metrics']['inference_ms']['median'], 1000)
        rows[0]['failure'] = ''
        self.assertEqual(benchmark.summarize(rows)[0]['failures'], 1)

    def test_first_english_metric_skips_sentences_without_english(self):
        events = [dict(first_arrival=1, first_english=1.25), dict(first_arrival=2, first_english=None)]
        self.assertEqual(benchmark.phrase_metrics(events)['first_english_ms'], {'median': 250, 'p95': 250})
