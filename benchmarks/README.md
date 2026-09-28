# Streaming performance checks

Run from the repository root with its Python environment. Supply an existing
local model directory; the benchmark does not download weights or install code.
It exports baseline source to a temporary directory and alternates execution
order across repetitions, using a fresh process and the same dependencies for
each variant. The default baseline is `cb44b5a`, before these changes.

```bash
.venv/bin/python benchmarks/streaming.py --model /path/to/local/model \
    --seconds 20 --repeats 3 > components.jsonl
.venv/bin/python benchmarks/streaming.py --model /path/to/local/model \
    --audio /path/to/recording.wav --seconds 20 --repeats 3 --paced > paced.jsonl
```

The final JSON line contains medians; preceding lines contain individual runs,
hardware, MLX version, quantization, and precision settings. `--batches 80 160`
limits the candidate batch sizes. `--context-size` defaults to 512 for these
measurements, independently of the application's direct CLI default of 8192.

## What the measurements mean

- **Component mode:** replay fixed 80/160/320 ms chunks through preprocessing,
  incremental encoding, and greedy decoding. Each stage synchronizes GPU work;
  total time includes Python overhead and prefill. EOS is ignored to keep the
  amount of work identical. Model loading and a one-second warmup are excluded.
  A 20-second input crosses the encoder's 750-frame window. This mode measures
  computation, not microphone waiting or the application's async overlap.
- **Paced mode:** feed 80 ms microphone blocks at real-time deadlines into the
  actual streaming loop. Baseline uses its original scheduling; candidate uses
  the chosen minimum batch threshold and opportunistic catch-up. EOF triggers
  the ordinary shutdown flush at a wait boundary. Encoder evaluation is
  synchronized for accounting. This reports instrumented application latency,
  including startup and final flushing, rather than isolated kernel speed.
- **Maximum audio backlog:** largest captured-minus-encoded sample count at a
  microphone callback, expressed in milliseconds. It includes the incoming block
  and deliberate batch buffering; it is not end-to-end text delay.
- **First text:** time from microphone start until the first non-whitespace
  output, or `null` if none. Synthetic noise is useful for throughput and cache
  behavior, but cannot establish recognition accuracy or useful speech latency.
- **FFT error:** maximum absolute normalized-mel difference from a NumPy FFT
  reference on up to one second of input, reported for both DFT and FFT versions.
  Neither implementation is required to match the other's floating-point error.

Peak memory includes resident weights and warmup buffers still retained by MLX;
it excludes the process's non-MLX allocations. These checks use the existing
model and context settings; they do not benchmark alternate quantizations or
translation. Retain recordings and transcripts separately when assessing WER.

## Correctness checks

```bash
.venv/bin/python -m unittest discover -s tests -v
MLX_ENABLE_TF32=0 .venv/bin/python -m unittest discover -s tests -v
```

MLX can use reduced-precision float32 matrix operations on M5 hardware. Tests
exercise the normal runtime with explicit tolerances and use tighter tolerances
when `MLX_ENABLE_TF32=0` selects full float32. This environment variable is for
validation; application defaults are not changed. See the
[MLX precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

The encoder oracle checks an exact causal sliding window, including startup
chunks larger than the window, single-query cache rotation, repeated rollover,
and 80/160/320 ms partitions through convolutions, downsampling, and the adapter.
It also checks that file-mode `model.encode` matches streaming `encode_step`
beyond the window; file mode encodes in window-sized chunks through the same cache.
Streaming tests use a fake microphone to check notification-before-wait,
notification-during-wait, bounded idle waits, batch thresholds, backlog draining,
EOS resets, partial-capture flushing, and that the newest prediction is shown
before the loop waits for more audio.

## Measured results (2026-09-22)

Apple M5, 24 GiB, macOS 27.0, Python 3.12, MLX 0.32.2, cached 6-bit model,
512-token decoder context, default precision, 480 ms transcription delay.
[Machine-readable results](results-m5.json) include the model revision and medians.

Three alternating repetitions of 20 seconds of seeded noise, including startup
and encoder cache rollover:

| Version / batch | Encoder ms / audio second | Total processing seconds | Processing / audio duration |
|---|---:|---:|---:|
| Baseline / 80 ms | 203.2 | 10.33 | 0.517 |
| Candidate / 80 ms | 198.0 | 10.22 | 0.511 |
| Baseline / 160 ms | 122.5 | 8.58 | 0.429 |
| Candidate / 160 ms | 131.3 | 8.92 | 0.446 |
| Baseline / 320 ms | 86.4 | 7.82 | 0.391 |
| Candidate / 320 ms | 88.4 | 7.94 | 0.397 |

Within the candidate, 160/320 ms batching reduced encoder cost by approximately
34%/55% relative to 80 ms, and total synchronized processing time by 13%/22%.
Against the baseline at the **same** batch size, the candidate is approximately
1% faster at 80 ms, 4% slower at 160 ms, and 2% slower at 320 ms in this run.
Correct window masking adds work, and timing noise remains; these are not
claims of a universal speedup. The default stays at 80 ms.

Two alternating paced repetitions of an 8.08-second generated French clip:

| Version / minimum batch | First text (ms) | Maximum audio backlog (ms) | Total including flush (s) |
|---|---:|---:|---:|
| Baseline / 80 ms | 1561 | 80 | 8.66 |
| Candidate / 80 ms | 1551 | 120 | 8.78 |
| Candidate / 160 ms | 1580 | 160 | 8.70 |
| Candidate / 320 ms | 1751 | 320 | 8.76 |

All runs transcribed the same words; the 320 ms runs changed punctuation and
capitalization. This is a synthetic speech smoke test, not a recognition corpus
or evidence of unchanged accuracy. First-text timing is essentially unchanged
at the default, while larger batches trade latency for lower encoder work.
Backlog results do not show a consistent default-mode improvement on this clip.

### Immediate token display

The loop previously printed each token when the *next* audio chunk arrived,
although the GPU had finished it within one decode step. It now prints the
newest prediction before waiting for audio. Three alternating paced runs per
version on the same clip at 80 ms, recording every `on_text` call through
`paced_replay`:

| Version | First text (ms) | Total including flush (s) |
|---|---:|---:|
| Before | 1550 | 8.68 |
| Immediate display | 1486 | 8.69 |

All 23 tokens shown before the audio ended appeared 63–72 ms earlier (median
66.5 ms). Transcripts were identical across all six runs, and processing work is
unchanged. After an end-of-stream token, the following chunk is no longer
discarded, because the reset now happens before that chunk is encoded.

Reproduce the local speech source without using microphone recordings:

```bash
say -v Thomas -o /tmp/voxmlx-french.aiff \
    'Bonjour. Nous testons la transcription en temps réel. Le modèle fonctionne sur cet ordinateur, sans connexion à Internet.'
.venv/bin/python benchmarks/streaming.py --model /path/to/local/model \
    --audio /tmp/voxmlx-french.aiff --seconds 20 --repeats 2 --paced
```

On the noise probe, maximum normalized-mel error against NumPy fell from
0.00638 (old DFT) to 0.000196 (FFT). On the French probe it fell from 0.0428 to
0.000190. This measures preprocessing numerics, not recognition accuracy.

## Compilation follow-up

Compilation was evaluated separately on the first encoder/decoder feed-forward
blocks using actual weights and seeded random inputs. Each variant used five
warmups and 50 synchronized calls in three alternating repetitions. Outputs
matched exactly on these probes. Encoder sizes were 4/8/16 frames, and decoder
size was one token.

The first probe showed a noisy 13% gain at eight encoder frames; repeating it
reduced that gain to 0.3%. The repeat measured 0.5%, 0.3%, and 2.2% encoder gains
and a 0.4% decoder slowdown. No configuration demonstrated a repeatable 5% gain,
so compilation remains disabled. Both probe results are preserved in the JSON
report. Future changes must meet that threshold and pass numerical checks.
Whole-decoder compilation, conditioning caches, and quantization changes remain
outside this implementation.


## Translation benchmarks

English streams by local agreement, the closest machine analogue of a
simultaneous interpreter. After each completed French word, the worker translates
the open sentence again, forcing the output to continue the English already shown
(CTranslate2 `target_prefix`), and shows the words on which the last two
translations agree. Shown English is never revised. A sentence completes at
sentence/newline punctuation, after the 1,500 ms idle pause, at a 240-character
word boundary, or at shutdown; its final translation continues the shown English.
A busy worker does not retranslate queued fragments one at a time. Arrival
timestamps survive sentence splits and translation backlog. The five-second
maximum phrase timer was removed because English now streams during long
sentences. `--translation-idle-ms 750` remains an optional latency/context
tradeoff. The translation model, two CPU threads, and beam four are unchanged;
calls go directly to CTranslate2 with Argos's settings, without Argos's MiniSBD
sentence splitter or paragraph cache.

Run from the repository root, with translation dependencies installed and the
French-to-English Argos model already cached. Normal `./transcribe.sh --translate-en`
downloads it on first use. Benchmarks never download it and report a missing
model as an error. They use the launcher's cache locations, respecting explicit
`XDG_*` environment overrides.

```bash
# Fixed complete phrases, one model call each.
.venv/bin/python benchmarks/translation.py --mode inference --tune > inference.jsonl

# Actual translation worker, paced fragments, pauses, long speech, final drain.
.venv/bin/python benchmarks/translation.py --mode text --tune > text.jsonl

# Local recording through current Voxtral, with and without translation.
.venv/bin/python benchmarks/translation.py --mode audio \
  --model /absolute/path/to/local/voxtral-model \
  --audio /absolute/path/to/french.wav --tune > audio.jsonl
```

All modes default to three repetitions and baseline `cb44b5a`. Workers run in
fresh processes, alternate variant order, and finish matched-default comparisons
before tuning. Audio uses **current Voxtral for every variant**, swapping only
the baseline translation scheduler. ASR-only workers do not load or warm a
translation model. The existing streaming benchmark remains available separately.
Audio defaults are 80 ms batches, 512-token context, and the existing 480 ms
transcription delay. Supply `--audio-source synthetic-speech` for synthesized
input; a recording is required, never random noise in this mode.

`--tune` independently tests one/four CPU threads (default two), beam sizes two/one
(default four), and a 750 ms idle pause in replay modes. It does not change live
application defaults. Inference measures each whole model call, including
tokenization.

An optional `--fixture fragments.jsonl` accepts chronological fragment arrivals
in milliseconds, ending with one explicit EOF record:

```jsonl
{"at_ms": 0, "text": "Bonjour. "}
{"at_ms": 500, "text": "Comment allez-vous"}
{"at_ms": 2500, "text": null}
```

The built-in fixture also includes continuous unpunctuated speech lasting more
than five seconds. Inference uses four complete reference phrases by default;
with a custom fixture it translates each nonempty newline-delimited paragraph
of the concatenated text. Keep fixtures local if they contain private speech.

Output is captured in memory during measurement; terminal rendering and
transcript file writes are excluded.

Each stdout line is JSON: individual runs followed by median summaries. Startup
errors and diagnostic messages go to stderr. Any translation failure is retained
in the results and makes the parent command exit unsuccessfully. Reports include
outputs, input hashes, baseline revision, a hash of candidate Python sources,
model package identity, effective compute type, and software/hardware versions.
No benchmark mutates the checkout or installs a different version of the package.

Metric definitions:

- **Buffering:** oldest character arrival to sentence readiness. Readiness comes
  from sentence/length boundaries, the idle deadline, or EOF.
- **Ready wait:** readiness to the start of the completing translation call,
  including a busy worker.
- **Inference:** completing translation call's start to return (or failure).
  Partial-sentence calls are counted in `translation_calls` and in each event's
  `inference_seconds`.
- **Sentence first English:** oldest character arrival to the first English shown
  for that sentence (candidate only; the phrase scheduler shows none early).
- **Output latency:** oldest character arrival to completion of its English line.
  Candidate timing events contain monotonic timestamps and phrase IDs. The callback
  runs on the translation worker; it should return quickly.
- **First English:** replay start to the first nonempty English text shown.
  Audio reports first transcription text and microphone backlog separately.
- **Outstanding source:** maximum queued/in-flight source characters, sampled on
  text submission, excluding source already emitted. This is not an audio duration.
- **Drain:** EOF to worker completion. For audio, EOF here is completion of the
  ASR shutdown flush, so ASR's flush is part of the audio processing total.
- **Load / warmup:** import/discovery and CTranslate2 model creation, followed by
  a separately timed representative warmup phrase. Audio has separate
  ASR load/warmup timings; its warmup exercises one second of paced input plus flush.
- **Peak RSS:** whole-process high-water mark, including load/warmup. Audio also
  records peak MLX allocation after resetting the MLX peak following warmup.

Phrase metrics include medians and nearest-rank p95 values. The final summary is
the median of each run's statistic; a p95 from a tiny fixture is effectively its
slowest phrase. Baseline buffering/readiness/output-latency fields are null because
the old scheduler did not retain source timestamps. Inference mode has no phrase
buffering, queue, or display metrics. Do not interpret unavailable values as zero.

Python callers can pass `on_timing=callback` to `LiveTranslation` to receive one
dictionary per nonempty completed sentence. It contains `phrase_id`, `source_chars`,
`source`, `english`, `error`, `first_arrival`, `last_arrival`, `ready_at`, `reason`,
`first_english`, `calls`, `inference_seconds`, `inference_start` and `inference_end`
(of the completing call), and `output_end`. Times are monotonic seconds;
source lengths include consumed whitespace. Timing callback failures are reported,
remaining source is still drained, and the context manager reports the failure.
`load_french_english(allow_download=False, warmup=False)` supports offline benchmark
setup; normal application calls retain download and warmup behavior.

### Translation measurements (Apple M5)

These phrase-scheduler results predate local agreement; see
[Local agreement measurements](#local-agreement-measurements-apple-m5) below.

Measured on 2026-09-22: Apple M5, 24 GiB, macOS 27.0 (26A428), Python 3.12.11,
MLX 0.32.2, Argos Translate 1.11.0, CTranslate2 4.8.2, MiniSBD 0.9.5. The installed
French-to-English package is version 1.9. `compute_type=auto` selected
`int8_float32`; no quantization setting was changed. Each configuration has three
fresh-process repetitions. Full summaries and output comparisons are in
[translation-results-m5.json](translation-results-m5.json).

Matched defaults, using the same current Voxtral model and code for every audio
variant:

| Audio variant | First transcription text | First English | Maximum audio backlog |
|---|---:|---:|---:|
| Transcription alone | 1.550 s | — | 80 ms |
| Baseline translation scheduler | 1.551 s | 6.927 s | 80 ms |
| Corrected translation scheduler | 1.553 s | 6.649 s | 80 ms |

Optional audio experiments (same clip and three repetitions):

| Candidate setting | First English | Maximum audio backlog |
|---|---:|---:|
| Default | 6.649 s | 80 ms |
| 750 ms idle pause | 6.654 s | 80 ms |
| One CPU thread | 6.657 s | 80 ms |
| Four CPU threads | 6.662 s | 80 ms |
| Beam two | 6.643 s | 80 ms |
| Beam one | 6.633 s | 160 ms |

The shorter idle pause did not help this continuous-speech clip; the maximum
phrase timer still determined first output. Beam one improved first English by
only about 16 ms here and its median maximum audio backlog increased. These
small samples do not establish causality for the backlog change. Median process
peak RSS was about 3.92 GB for ASR alone and 4.30 GB with candidate translation.
All **63 measured runs** completed without translation failures; replay outputs
preserved source words in order. The full **40-test suite** also passed.

The corrected scheduler produced first English about **278 ms earlier** on this
clip. This is a phrase-deadline improvement, not an inference speedup. The clip's
transcription contains a long comma-separated phrase, so the five-second maximum
is the dominant delay. The few-millisecond transcription differences do not
establish a meaningful change in ASR performance.

On timestamped text replay, first English stayed approximately 18.5 ms with both
default schedulers. The continuous-speech example reached output about 0.4 seconds
earlier with the corrected maximum deadline. A 750 ms idle pause made the isolated
unfinished phrase ready 750 ms earlier and increased the total number of translated
phrases from five to six. Correcting deadlines also changes where the continuous
phrase splits; resulting translations are included in the report.

Fixed-phrase inference medians (the median of each run's median over four phrases):

| CPU threads | Beam size | Translation-call time |
|---:|---:|---:|
| 2 | 4, baseline control | 33.56 ms |
| 2 | 4, candidate default | 33.26 ms |
| 1 | 4 | 35.06 ms |
| 4 | 4 | 34.56 ms |
| 2 | 2 | 30.02 ms |
| 2 | 1 | 24.16 ms |

There is no demonstrated benefit to changing the default thread count here.
Beam one reduced this component median by about 27%, but beam changes can alter
translations and have not been quality-validated. Application defaults remain
two threads, beam four, and a 1,500 ms pause. No end-to-end percentage is inferred
from these component measurements.

The audio input was **synthetic speech**, not a human recording or an accuracy
corpus. It was generated with the macOS Thomas voice:

```bash
say -v Thomas -o /private/tmp/voxmlx-french.aiff \
  'Bonjour. Nous testons la transcription en temps réel. Le modèle fonctionne sur cet ordinateur, sans connexion à Internet.'
```

The input is approximately 8.06 seconds (8.08 after token alignment). The local
Voxtral model was `mlx-community/Voxtral-Mini-4B-Realtime-6bit`, snapshot
`02eb0caeb9dafb554c17a72b93dbf40cd3736c31`. Numerical timings, source-word preservation,
and observed output differences do not establish unchanged recognition or
translation accuracy. Broader recorded-audio and translation-quality evaluation
is still needed before choosing faster defaults.

### Local agreement measurements (Apple M5)

Same machine, software, model, and speech clip as above, measured 2026-09-22.
The baseline is the previous phrase scheduler (sentence, idle, and length
boundaries plus the five-second maximum), loaded from the pre-change source by
passing `--baseline-path` to benchmark workers. Both variants call the same direct
model wrapper, so only scheduling differs. Medians of three alternating
fresh-process repetitions:

| Audio variant | First transcription text | First English | Maximum audio backlog | Translation calls | Translation CPU | Drain after EOF |
|---|---:|---:|---:|---:|---:|---:|
| Transcription alone | 1.488 s | — | 80 ms | — | — | — |
| Previous phrase scheduler | 1.485 s | 6.588 s | 80 ms | 2 | 0.14 s | 48 ms |
| Local agreement | 1.482 s | 2.563 s | 80 ms | 18 | 1.79 s | 158 ms |

English first appeared about 4.0 seconds earlier. In a real-time replay of the
recorded tokens, it then trailed the French text by roughly 0.5–1 second.
Transcription timing and audio backlog were unchanged. Translation used about
13 times more CPU time, about 20% of the 8.85-second run on two threads. Peak
RSS was 4.05 GB with either scheduler.

The previous scheduler split the clip's comma-joined transcription at the
five-second limit: "…the model works on" / "This computer, without internet
connection." Local agreement translated one sentence: "Hello, we're testing the
transcript in real time, the model works on this computer, without connection to
the internet." Early commitments are not revised; the shown "transcript" is where
a full-sentence translation says "transcription".

On the text fixture, first English stayed at 16 ms because the first sentence is
complete on arrival. The continuous unpunctuated sentence showed English
1.24 seconds after its first word instead of waiting for the five-second maximum.
With no punctuation and a 0.7-second gap, it merged with "Merci pour votre
attention". The previous scheduler split it mid-phrase, and its English for
"limite de durée Merci pour votre attention" omitted "limite de durée".
Translation CPU was 1.44 s versus 0.25 s. All 18 runs completed without failures,
and the 45-test suite passed. One synthetic clip and one fixture are not a
translation-quality evaluation.
