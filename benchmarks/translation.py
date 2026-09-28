"""Offline translation benchmarks: inference, timestamped text, or paced audio.

Each variant runs in a fresh process. Audio uses this checkout's ASR for all
variants; only the translation scheduler is loaded from the baseline revision.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# Pauses, sentence boundaries, continuous speech spanning five seconds, and EOF.
DEFAULT_FIXTURE = [
    {'at_ms': 0, 'text': 'Bonjour. '},
    {'at_ms': 100, 'text': 'Nous testons '},
    {'at_ms': 300, 'text': 'la traduction'},
    {'at_ms': 2000, 'text': ' en direct. '},
    *[{'at_ms': 2200 + i * 600, 'text': word + ' '}
      for i, word in enumerate('nous continuons à parler sans ponctuation pour vérifier la limite de durée'.split())],
    {'at_ms': 10100, 'text': 'Merci pour votre attention'},
    {'at_ms': 10400, 'text': None},
]
REFERENCE_PHRASES = ['Bonjour.', 'Nous testons la traduction en direct.',
                     'Nous continuons à parler sans ponctuation pour vérifier la limite de durée.',
                     'Merci pour votre attention']


def read_fixture(path=None):
    rows = ([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            if path else DEFAULT_FIXTURE)
    previous = -1
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('Fixture records must be JSON objects')
        at = row.get('at_ms')
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at) or at < 0 or at < previous:
            raise ValueError('Fixture at_ms must be finite, nonnegative, and nondecreasing')
        previous = at
        if 'text' not in row or not (isinstance(row['text'], str) or row['text'] is None):
            raise ValueError('Fixture text must be a string or final null EOF')
        if row['text'] is None and i != len(rows) - 1:
            raise ValueError('Only the final fixture record may be EOF')
    if not rows or rows[-1]['text'] is not None or not any(isinstance(r['text'], str) and r['text'].strip() for r in rows):
        raise ValueError('Fixture requires text and a final null EOF')
    return rows


def distribution(values):
    if not values:
        return {'median': None, 'p95': None}
    ordered = sorted(values)
    return {'median': statistics.median(values), 'p95': ordered[math.ceil(.95 * len(values)) - 1]}


def phrase_metrics(events):
    pairs = {'buffering_ms': ('first_arrival', 'ready_at'),
             'ready_wait_ms': ('ready_at', 'inference_start'),
             'inference_ms': ('inference_start', 'inference_end'),
             'first_english_ms': ('first_arrival', 'first_english'),
             'output_latency_ms': ('first_arrival', 'output_end')}
    return {name: distribution([(e[b] - e[a]) * 1000 for e in events
                                if e.get(a) is not None and e.get(b) is not None])
            for name, (a, b) in pairs.items()}


def worker(args):
    # Match the launcher's cache location, while respecting an explicit one.
    os.environ.setdefault('VOXMLX_CACHE_DIR', str(ROOT / '.cache/voxmlx'))
    sys.path.insert(0, str(ROOT))
    from voxmlx.translation import LiveTranslation, load_french_english, model_dir
    tick = time.monotonic()
    if args.variant == 'asr-only':
        translate = None
        load_ms = warmup_ms = 0
    else:
        # Loading creates the CTranslate2 model; time the first phrase separately.
        translate = load_french_english(allow_download=False, warmup=False,
                                        threads=args.threads, beam_size=args.beam)
        load_ms = (time.monotonic() - tick) * 1000
        tick = time.monotonic()
        translate('Bonjour, ceci est un exercice de préparation.')
        warmup_ms = (time.monotonic() - tick) * 1000
    if args.variant == 'baseline':
        spec = importlib.util.spec_from_file_location('baseline_translation', args.baseline_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        LiveTranslation = module.LiveTranslation

    rows = read_fixture(args.fixture)
    events, calls = [], []
    outstanding = ''
    max_backlog = 0
    first_english = None
    lock = threading.Lock()

    def timed_translate(text, prefix=''):
        # Includes partial-sentence calls; baseline schedulers never pass a prefix.
        event = {'source': text, 'prefix': prefix, 'inference_start': time.monotonic()}
        try:
            event.update(english=translate(text, prefix), error=None)
            return event['english']
        except Exception as exc:
            event.update(english=None, error=str(exc))
            raise
        finally:
            event['inference_end'] = time.monotonic()
            calls.append(event)

    def consume(text):
        nonlocal outstanding
        with lock:
            # Include whitespace skipped by the sentence splitter.
            pos = outstanding.find(text.strip())
            if pos < 0:
                raise RuntimeError('Emitted phrase does not match queued source')
            outstanding = outstanding[pos + len(text.strip()):]
            outstanding = outstanding.lstrip()

    def on_sentence(event):
        nonlocal first_english
        events.append(event)
        if event['first_english'] is not None and (first_english is None or event['first_english'] < first_english):
            first_english = event['first_english']
        consume(event['source'])

    options = dict(idle_seconds=args.idle_ms / 1000)
    if args.variant != 'baseline':
        options['on_timing'] = on_sentence
    live = LiveTranslation(timed_translate, **options)
    if args.variant == 'baseline':
        # Phrase-at-a-time schedulers print each phrase's English in _emit.
        original_emit = live._emit

        def emit(text, *values):
            nonlocal first_english
            original_emit(text, *values)
            if text.strip():
                ended = time.monotonic()
                if calls[-1]['error'] is None and calls[-1]['english'].strip() and first_english is None:
                    first_english = ended
                events.append(dict(calls[-1], output_end=ended))
                consume(text)
        live._emit = emit

    def write(text):
        nonlocal outstanding, max_backlog
        if text:
            with lock:
                outstanding += text
                max_backlog = max(max_backlog, len(outstanding))
            live.write(text)

    audio_result = None
    if args.mode == 'audio':
        import mlx.core as mx
        import numpy as np
        from voxmlx import load_model
        from voxmlx.audio import load_audio
        from streaming import paced_replay
        tick = time.monotonic()
        model, tokenizer, config = load_model(args.model)
        audio = load_audio(args.audio)
        if len(audio) < 1280:
            raise ValueError('Audio must contain at least 80 ms')
        audio = np.pad(audio, (0, -len(audio) % 1280))
        asr_load_ms = (time.monotonic() - tick) * 1000
        tick = time.monotonic()
        paced_replay(args, model, tokenizer, config, audio[:16000])
        mx.synchronize()
        asr_warmup_ms = (time.monotonic() - tick) * 1000
        mx.reset_peak_memory()

    started = time.monotonic()
    failure = None
    eof = started
    with redirect_stdout(io.StringIO()):
        if args.mode == 'inference':
            phrases = (REFERENCE_PHRASES if not args.fixture else
                       [p.strip() for p in ''.join(r['text'] or '' for r in rows).splitlines() if p.strip()])
            for phrase in phrases:
                try:
                    timed_translate(phrase)
                except Exception as exc:
                    failure = str(exc)
            events = calls
            eof = time.monotonic()
        else:
            try:
                if args.variant == 'asr-only':
                    audio_result = paced_replay(args, model, tokenizer, config, audio)
                    eof = time.monotonic()
                else:
                    with live:
                        if args.mode == 'audio':
                            audio_result = paced_replay(args, model, tokenizer, config, audio, on_text=write)
                            eof = time.monotonic()
                        else:
                            for row in rows:
                                time.sleep(max(0, started + row['at_ms'] / 1000 - time.monotonic()))
                                if row['text'] is not None:
                                    write(row['text'])
                            eof = time.monotonic()
            except Exception as exc:
                failure = str(exc)
    ended = time.monotonic()
    if args.mode != 'inference' and args.variant != 'asr-only' and outstanding.strip():
        failure = failure or 'Replay finished with unemitted source text'
    metadata = model_dir() / 'metadata.json'
    effective_compute_type = getattr(getattr(translate, 'translator', None), 'compute_type', None)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
    result = dict(mode=args.mode, variant=args.variant, threads=args.threads, beam=args.beam,
                  idle_ms=args.idle_ms, load_ms=load_ms, warmup_ms=warmup_ms,
                  total_ms=(ended - started) * 1000,
                  drain_ms=(ended - eof) * 1000 if args.mode != 'inference' else None,
                  first_english_ms=(first_english - started) * 1000 if first_english else None,
                  max_outstanding_source_chars=max_backlog if args.mode != 'inference' else None,
                  phrases=len(events), translation_calls=len(calls), failure=failure, peak_rss_bytes=rss_bytes,
                  metrics=phrase_metrics(events), events=events,
                  hardware=platform.machine(), os=platform.platform(), python=platform.python_version(),
                  versions={p: importlib.metadata.version(p) for p in ('ctranslate2', 'sentencepiece', 'mlx')},
                  translation_model=json.loads(metadata.read_text()) if metadata.exists() else None,
                  effective_compute_type=effective_compute_type,
                  input_sha256=hashlib.sha256(Path(args.audio).read_bytes() if args.mode == 'audio' else
                                            json.dumps(phrases if args.mode == 'inference' else rows, sort_keys=True).encode()).hexdigest())
    if sys.platform == 'darwin':
        result['hardware'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if args.mode == 'inference':
        seconds = sum(e['inference_end'] - e['inference_start'] for e in calls)
        result['source_chars_per_second'] = sum(len(e['source']) for e in calls) / seconds
    if args.mode == 'audio':
        result.update(audio=audio_result, asr_load_ms=asr_load_ms, asr_warmup_ms=asr_warmup_ms,
                      asr_model=str(Path(args.model).resolve()), audio_source=args.audio_source,
                      peak_mlx_bytes=mx.get_peak_memory(), context_size=args.context_size,
                      audio_batch_ms=args.batch_ms)
    print(json.dumps(result, ensure_ascii=False))


def summarize(rows):
    summaries = []
    for key in dict.fromkeys((r['variant'], r['threads'], r['beam'], r['idle_ms']) for r in rows):
        group = [r for r in rows if (r['variant'], r['threads'], r['beam'], r['idle_ms']) == key]
        summary = dict(zip(('variant', 'threads', 'beam', 'idle_ms'), key))
        for name in ('load_ms', 'warmup_ms', 'total_ms', 'drain_ms', 'first_english_ms',
                     'max_outstanding_source_chars', 'peak_rss_bytes', 'source_chars_per_second'):
            values = [r[name] for r in group if r.get(name) is not None]
            summary[name] = statistics.median(values) if values else None
        summary['metrics'] = {name: {stat: statistics.median(values) if
                              (values := [r['metrics'][name][stat] for r in group if r['metrics'][name][stat] is not None]) else None
                              for stat in ('median', 'p95')} for name in group[0]['metrics']}
        if group[0].get('audio'):
            summary['audio'] = {name: statistics.median(values) if
                               (values := [r['audio'][name] for r in group if r['audio'][name] is not None]) else None
                               for name in ('first_text_ms', 'max_audio_backlog_ms', 'total_ms')}
        summary['failures'] = sum(r['failure'] is not None for r in group)
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inference', 'text', 'audio'), default='text')
    parser.add_argument('--fixture', help='JSONL timestamped fragments ending with null EOF')
    parser.add_argument('--audio')
    parser.add_argument('--audio-source', choices=('recording', 'synthetic-speech'), default='recording')
    parser.add_argument('--model', help='Existing local Voxtral model directory for audio mode')
    parser.add_argument('--context-size', type=int, default=512)
    parser.add_argument('--batch-ms', type=int, choices=(80, 160, 320), default=80)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', default='cb44b5a')
    parser.add_argument('--tune', action='store_true', help='Also test 750 ms pause, threads 1/4, beams 2/1')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--variant', default='candidate', help=argparse.SUPPRESS)
    parser.add_argument('--threads', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--beam', type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument('--idle-ms', type=float, default=1500, help=argparse.SUPPRESS)
    parser.add_argument('--baseline-path', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        read_fixture(args.fixture)
        if args.repeats < 1 or args.context_size < 39:
            raise ValueError('Require repeats >= 1 and context-size >= 39')
        if args.mode == 'audio' and (not args.audio or not Path(args.audio).is_file() or not args.model or not Path(args.model).is_dir()):
            raise ValueError('Audio mode requires a local --audio file and --model directory')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.worker:
        worker(args)
        return
    source = subprocess.check_output(['git', 'show', f'{args.baseline}:voxmlx/translation.py'], cwd=ROOT)
    baseline_sha = subprocess.check_output(['git', 'rev-parse', args.baseline], cwd=ROOT, text=True).strip()
    source_hash = hashlib.sha256()
    for p in sorted((ROOT / 'voxmlx').glob('*.py')):
        source_hash.update(p.name.encode() + p.read_bytes())
    rows = []
    variants = [('baseline', 2, 4, 1500), ('candidate', 2, 4, 1500)]
    if args.mode == 'audio':
        variants.insert(0, ('asr-only', 2, 4, 1500))
    tuning = [('candidate', 1, 4, 1500), ('candidate', 4, 4, 1500),
              ('candidate', 2, 2, 1500), ('candidate', 2, 1, 1500)]
    if args.mode != 'inference':
        tuning.insert(0, ('candidate', 2, 4, 750))
    with tempfile.TemporaryDirectory(prefix='voxmlx-translation-') as directory:
        baseline = Path(directory) / 'translation.py'
        baseline.write_bytes(source)
        # Finish matched-default comparisons before the optional tuning stage.
        for stage in ([variants, tuning] if args.tune else [variants]):
            for repeat in range(args.repeats):
                for variant, threads, beam, idle in (stage if repeat % 2 == 0 else stage[::-1]):
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--mode', args.mode,
                               '--variant', variant, '--threads', str(threads), '--beam', str(beam),
                               '--idle-ms', str(idle), '--baseline-path', str(baseline),
                               '--context-size', str(args.context_size), '--batch-ms', str(args.batch_ms),
                               '--audio-source', args.audio_source]
                    for name in ('fixture', 'audio', 'model'):
                        if value := getattr(args, name):
                            command += ['--' + name, str(Path(value).resolve())]
                    output = subprocess.check_output(command, cwd=ROOT, text=True)
                    row = json.loads(output.strip().splitlines()[-1])
                    row.update(repeat=repeat + 1, baseline=baseline_sha, candidate_sha256=source_hash.hexdigest())
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps({'medians': summarize(rows), 'mode': args.mode, 'repeats': args.repeats,
                      'baseline': baseline_sha, 'candidate_sha256': source_hash.hexdigest()}), flush=True)
    if any(r['failure'] is not None for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
