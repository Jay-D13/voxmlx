# voxmlx

Realtime speech-to-text with
[Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
in [MLX](https://github.com/ml-explore/mlx).

## Quick start: live translation with quality mode

With `uv` installed, just run this from the project folder:

```bash
./transcribe.sh --translate-en --quality
```

This starts live French-to-English translation with the quality preset,
shows both languages in your terminal, and saves them in `transcripts/`.
Dependencies and missing models are downloaded automatically on first use.
Allow microphone access if prompted. Press **Ctrl+C** to stop.

## Install

```bash
pip install voxmlx
```

## Usage

### `voxmlx`

Transcribe audio from a file or stream from the microphone in real-time.

**Stream from microphone:**

```bash
voxmlx
```

**Run locally with uv and a small live context, saving the transcript:**

```bash
./transcribe.sh
```

The script uses `uv`, displays text live in your terminal, and saves the same
text to a timestamped file in `transcripts/`. Extra options are passed to
voxmlx, for example `./transcribe.sh --context-size 1024`.

**Quality preset for an M5 Max with 48 GB:**

```bash
./transcribe.sh --quality
```

This selects the [8-bit voxmlx model](https://huggingface.co/ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx),
a 1,024-token context, and a 2,400 ms transcription delay. It aims for accuracy
at the cost of later text output; it is not a speed preset. The first run
downloads the additional model weights (about 4.7 GB). Actual accuracy and
speed on an M5 Max have not been benchmarked here.

Explicit options override the preset, for example
`./transcribe.sh --quality --delay-ms 960`. Use `./transcribe.sh --help` for
script usage without loading a model.

**Live French-to-English translation:**

```bash
./transcribe.sh --translate-en
# Or combine it with the quality preset:
./transcribe.sh --quality --translate-en
```

The script installs the optional translation dependencies through `uv`. Before
opening the microphone, it downloads and prepares the French-to-English
[Argos Translate](https://github.com/argosopentech/argos-translate) model if
needed. Downloads are required on first use; speech and text are processed
locally, with translation on the CPU. Cached models work offline afterward.

The terminal and saved transcript show paired `FR:` and `EN:` lines. Translation
updates at sentence boundaries, after a short pause, or in chunks during long
unpunctuated speech. It adds phrase buffering and translation time to Voxtral's
delay. Ctrl+C drains pending translations before exiting. French source text
is kept if a translation fails. This option assumes French input.

For direct CLI usage, install with `uv sync --extra translation --no-editable`
and run `uv run --extra translation --no-editable voxmlx --translate-en`.

Allow microphone access when macOS prompts. Press Ctrl+C to stop and flush
the last words. The context window limits decoder memory, not transcript length;
512 tokens cover roughly 41 seconds of streaming positions. The audio encoder
cache uses the model's configured sliding window.

**Transcribe a file:**

```bash
voxmlx --audio audio.flac
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--audio` | Path to audio file (omit to stream from mic) | None |
| `--model` | Model path or HuggingFace model ID | `mlx-community/Voxtral-Mini-4B-Realtime-6bit` |
| `--temp` | Sampling temperature (`0` = greedy) | `0.0` |
| `--context-size` | Live decoder context; must fit `33 + delay_ms / 80` tokens | `8192` |
| `--delay-ms` | Live delay: multiples of 80 from 80–1200, or 2400 | `480` |
| `--translate-en` | Local French-to-English translation; display/save both languages | Off |

### `voxmlx-convert`

Convert Voxtral weights to voxmlx/MLX format with optional quantization.

**Basic conversion:**

```bash
voxmlx-convert --mlx-path voxtral-mlx
```

**4-bit quantized conversion:**

```bash
voxmlx-convert -q --mlx-path voxtral-mlx-4bit
```

**Convert and upload to HuggingFace:**

```bash
voxmlx-convert -q --mlx-path voxtral-mlx-4bit --upload-repo username/voxtral-mlx-4bit
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--hf-path` | HuggingFace model ID or local path | `mistralai/Voxtral-Mini-4B-Realtime-2602` |
| `--mlx-path` | Output directory | `mlx_model` |
| `-q`, `--quantize` | Quantize the model | Off |
| `--group-size` | Quantization group size | `64` |
| `--bits` | Bits per weight | `4` |
| `--dtype` | Cast weights (`float16`, `bfloat16`, `float32`) | None |
| `--upload-repo` | HuggingFace repo to upload converted model | None |

### Python API

```python
from voxmlx import transcribe

text = transcribe("audio.flac")
print(text)
```
