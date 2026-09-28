"""Compare this checkout to a git revision, without installing either checkout.

Component mode uses fixed work (ignores EOS) and synchronized GPU timings.
The default 20 seconds spans encoder cache rollover. Paced mode exercises the actual microphone loop using a
recording/synthetic source. It reports instrumented latency, not WER.
"""
import argparse
from contextlib import redirect_stdout
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time


def worker(args):
    sys.path.insert(0, args.checkout)
    import mlx.core as mx
    import numpy as np
    from voxmlx import load_model, _build_prompt_tokens
    from voxmlx.audio import load_audio, log_mel_spectrogram_step, SAMPLES_PER_TOKEN
    from voxmlx.cache import RotatingKVCache

    model, tokenizer, config = load_model(args.model)
    if args.audio:
        audio = load_audio(args.audio)
        audio = audio[:int(args.seconds * 16000)]
    else:
        audio = np.random.default_rng(0).normal(0, .05, int(args.seconds * 16000)).astype(np.float32)
    if len(audio) < SAMPLES_PER_TOKEN:
        raise ValueError('Provide at least 80 ms of audio')
    audio = np.pad(audio, (0, (-len(audio)) % SAMPLES_PER_TOKEN))
    prompt, delay = _build_prompt_tokens(tokenizer)
    cond = model.time_embedding(mx.array([delay], dtype=mx.float32))
    text = model.language_model.embed(mx.array([prompt]))[0]
    mx.eval(cond, text)

    def components(samples):
        totals = dict(mel_ms=0., encoder_ms=0., decoder_ms=0.)
        tail = c1 = c2 = enc_cache = ds = None
        cache = [RotatingKVCache(args.context_size) for _ in model.language_model.layers]
        buffered = None
        token = None
        tokens = 0
        step = args.batch_ms * 16
        chunks = [np.concatenate([np.zeros(32 * SAMPLES_PER_TOKEN, np.float32), samples[:1280]])]
        chunks.extend(samples[i:i + step] for i in range(1280, len(samples), step))
        start = time.perf_counter()
        for chunk in chunks:
            tick = time.perf_counter()
            mel, tail = log_mel_spectrogram_step(chunk, tail)
            mx.eval(mel)
            totals['mel_ms'] += (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            embeds, c1, c2, enc_cache, ds = model.encode_step(mel, c1, c2, enc_cache, ds)
            mx.eval(embeds)
            totals['encoder_ms'] += (time.perf_counter() - tick) * 1000
            if embeds is None:
                continue
            buffered = embeds if buffered is None else mx.concatenate([buffered, embeds])
            if token is None:
                if buffered.shape[0] < len(prompt):
                    continue
                tick = time.perf_counter()
                logits = model.decode((text + buffered[:len(prompt)])[None], cond, 'causal', cache)
                token = mx.argmax(logits[0, -1])
                mx.eval(token)
                totals['decoder_ms'] += (time.perf_counter() - tick) * 1000
                buffered = buffered[len(prompt):]
            for embed in buffered:
                tick = time.perf_counter()
                text_embed = model.language_model.embed(token.reshape(1, 1))[0, 0]
                logits = model.decode((embed + text_embed)[None, None], cond, cache=cache)
                token = mx.argmax(logits[0, -1])
                mx.eval(token)
                totals['decoder_ms'] += (time.perf_counter() - tick) * 1000
                tokens += 1
            buffered = None
        totals.update(total_ms=(time.perf_counter() - start) * 1000, decode_steps=tokens)
        return totals

    # Warm all stages and the chosen batch shape before starting the measurements.
    components(audio[:min(len(audio), 16000)])
    mx.reset_peak_memory()
    if args.paced:
        result = paced_replay(args, model, tokenizer, config, audio)
    else:
        result = components(audio)
        result['encoder_ms_per_audio_second'] = result['encoder_ms'] / (len(audio) / 16000)
        result['processing_rtf'] = result['total_ms'] / (len(audio) / 16)
    # Assess the old DFT and new FFT separately against the same NumPy oracle.
    probe = audio[:16000]
    combined = np.pad(probe, (200, 0))
    frames = np.lib.stride_tricks.sliding_window_view(combined, 400)[::160]
    spectrum = np.fft.rfft(frames * np.hanning(401)[:-1])
    from voxmlx.audio import mel_filter_bank
    mel = np.abs(spectrum) ** 2 @ mel_filter_bank().T
    reference = ((np.maximum(np.log10(np.maximum(mel, 1e-10)), -6.5) + 4) / 4).T
    actual, _ = log_mel_spectrogram_step(probe, None)
    result['mel_reference_max_abs_error'] = float(np.max(np.abs(np.array(actual) - reference)))
    result.update(batch_ms=args.batch_ms, audio_seconds=len(audio) / 16000,
                  peak_memory_gb=mx.get_peak_memory() / 1e9,
                  mlx=importlib.metadata.version('mlx'),
                  device=mx.device_info()['device_name'],
                  memory_gb=mx.device_info()['memory_size'] / 2**30,
                  tf32=os.environ.get('MLX_ENABLE_TF32', 'default'),
                  quantization=config.get('quantization'),
                  source='recording' if args.audio else 'synthetic noise')
    print(json.dumps(result))


def paced_replay(args, model, tokenizer, config, audio, *, on_text=None):
    """Inject a paced microphone; EOF interrupts only at an idle/wait boundary."""
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch
    import mlx.core as mx
    import voxmlx.stream as stream

    finished = threading.Event()
    stop = threading.Event()
    conditions = []
    captured = processed = 0
    pending_chunk = 0
    started = first_text = None
    max_backlog = 0.
    original_mel = stream.log_mel_spectrogram_step
    original_encode = model.encode_step
    transcript = []
    failures = []

    def mel(chunk, tail):
        nonlocal pending_chunk
        pending_chunk = len(chunk) - (32 * 1280 if tail is None else 0)
        return original_mel(chunk, tail)

    def encode(*values):
        nonlocal processed
        result = original_encode(*values)
        mx.eval(result[0])
        if not stop.is_set():
            processed += pending_chunk
        return result

    def emit(text):
        nonlocal first_text
        if text.strip() and first_text is None:
            first_text = time.perf_counter() - started
        transcript.append(text)
        if on_text is not None:
            on_text(text)

    class Condition(threading.Condition):
        def __init__(self):
            super().__init__()
            conditions.append(self)
        def wait_for(self, predicate, timeout=None):
            if not predicate() and finished.is_set():
                raise KeyboardInterrupt
            ready = super().wait_for(lambda: predicate() or finished.is_set(), timeout)
            if not predicate() and finished.is_set():
                raise KeyboardInterrupt
            return ready and predicate()

    class Microphone:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            self.thread = threading.Thread(target=self.feed, name='benchmark-audio')
        def feed(self):
            nonlocal captured, max_backlog
            try:
                for pos in range(0, len(audio), 1280):
                    # A microphone delivers each block after its samples arrive.
                    deadline = started + (pos + 1280) / 16000
                    if stop.wait(max(0, deadline - time.perf_counter())):
                        return
                    chunk = audio[pos:pos + 1280]
                    captured += len(chunk)
                    max_backlog = max(max_backlog, (captured - processed) / 16000)
                    self.callback(chunk[:, None], len(chunk), None, None)
            except BaseException as exc:
                failures.append(str(exc))
            finally:
                finished.set()
                for condition in conditions:
                    with condition:
                        condition.notify_all()
        def start(self):
            nonlocal started
            started = time.perf_counter()
            self.thread.start()
        def stop(self):
            stop.set()
            self.thread.join()
        def close(self):
            pass

    def baseline_sleep(seconds):
        if finished.is_set():
            raise KeyboardInterrupt
        time.sleep(seconds)

    kwargs = dict(context_size=args.context_size, on_text=emit)
    if 'audio_batch_ms' in inspect.signature(stream.stream_transcribe).parameters:
        kwargs['audio_batch_ms'] = args.batch_ms
    with patch.object(stream, 'load_model', return_value=(model, tokenizer, config)), \
         patch.object(stream.sd, 'InputStream', Microphone), \
         patch.object(stream, 'threading', SimpleNamespace(Condition=Condition, Lock=threading.Lock)), \
         patch.object(stream, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=baseline_sleep)), \
         patch.object(stream, 'log_mel_spectrogram_step', mel), \
         patch.object(model, 'encode_step', encode), redirect_stdout(io.StringIO()):
        stream.stream_transcribe(**kwargs)
    if failures:
        raise RuntimeError(failures)
    return dict(total_ms=(time.perf_counter() - started) * 1000,
                first_text_ms=None if first_text is None else first_text * 1000,
                max_audio_backlog_ms=max_backlog * 1000,
                transcript=''.join(transcript))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Existing local model directory; no downloads')
    parser.add_argument('--audio', help='Optional local recording; otherwise deterministic synthetic noise')
    parser.add_argument('--seconds', type=float, default=20, help='Audio duration / recording limit')
    parser.add_argument('--context-size', type=int, default=512)
    parser.add_argument('--batches', nargs='+', type=int, choices=(80, 160, 320), default=[80, 160, 320])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', default='cb44b5a', help='Git revision to compare to')
    parser.add_argument('--paced', action='store_true', help='Run the actual loop with microphone-paced input')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--checkout', help=argparse.SUPPRESS)
    parser.add_argument('--batch-ms', type=int, default=80, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve())
    if not Path(args.model).is_dir():
        parser.error('--model must be an existing local model directory')
    if args.audio:
        args.audio = str(Path(args.audio).resolve())
    if args.seconds < .08 or args.repeats < 1 or args.context_size < 39:
        parser.error('Require seconds >= .08, repeats >= 1, context-size >= 39')
    if args.worker:
        worker(args)
        return

    root = Path(__file__).resolve().parents[1]
    rows = []
    with tempfile.TemporaryDirectory(prefix='voxmlx-baseline-') as directory:
        # Export tracked source only; no checkout, dependency install, or branch mutation.
        archive = subprocess.check_output(['git', 'archive', args.baseline, 'voxmlx'], cwd=root)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(directory, filter='data')
        variants = [('baseline', directory, 80)] if args.paced else []
        variants += [('candidate', str(root), b) for b in args.batches]
        if not args.paced:
            variants = [(label, path, b) for b in args.batches
                        for label, path in [('baseline', directory), ('candidate', str(root))]]
        for repeat in range(args.repeats):
            for label, checkout, batch in (variants if repeat % 2 == 0 else variants[::-1]):
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--checkout', checkout, '--model', args.model, '--seconds', str(args.seconds),
                           '--context-size', str(args.context_size), '--batch-ms', str(batch)]
                if args.audio:
                    command += ['--audio', args.audio]
                if args.paced:
                    command += ['--paced']
                output = subprocess.check_output(command, cwd=checkout, text=True)
                row = json.loads(output.strip().splitlines()[-1])
                row.update(variant=label, repeat=repeat + 1)
                rows.append(row)
                print(json.dumps(row), flush=True)
    summaries = []
    for label, batch in dict.fromkeys((row['variant'], row['batch_ms']) for row in rows):
        group = [r for r in rows if r['variant'] == label and r['batch_ms'] == batch]
        summary = dict(variant=label, batch_ms=batch)
        for key in group[0]:
            numeric = any(isinstance(r[key], (int, float)) for r in group)
            if key not in ('repeat', 'batch_ms') and (numeric or key == 'first_text_ms'):
                values = [r[key] for r in group if r[key] is not None]
                summary[key] = statistics.median(values) if values else None
        summaries.append(summary)
    print(json.dumps({'medians': summaries, 'mode': 'paced' if args.paced else 'components',
                      'baseline': args.baseline, 'repeats': args.repeats}), flush=True)


if __name__ == '__main__':
    main()
