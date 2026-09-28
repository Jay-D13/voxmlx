import argparse
import threading
import time

import mlx.core as mx
import numpy as np
import sounddevice as sd
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from . import _build_prompt_tokens, load_model
from .audio import SAMPLES_PER_TOKEN, log_mel_spectrogram_step
from .cache import RotatingKVCache

N_LEFT_PAD_TOKENS = 32
N_FLUSH_PAD_TOKENS = 11  # Additional padding beyond the transcription delay.


def stream_transcribe(
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
    context_size: int = 8192,
    delay_ms: int = 480,
    on_text=None,
    audio_batch_ms: int = 80,
):
    if delay_ms not in (*range(80, 1201, 80), 2400):
        raise ValueError("delay_ms must be a multiple of 80 from 80 to 1200, or 2400")
    if audio_batch_ms not in (80, 160, 320):
        raise ValueError("audio_batch_ms must be 80, 160, or 320")
    batch_samples = audio_batch_ms // 80 * SAMPLES_PER_TOKEN
    n_delay_tokens = delay_ms // 80
    min_context = 1 + N_LEFT_PAD_TOKENS + n_delay_tokens
    if context_size < min_context:
        raise ValueError(f"context_size must be at least {min_context} tokens for the streaming prompt")
    emit = on_text if on_text is not None else lambda text: print(text, end="", flush=True)
    model, sp, config = load_model(model_path)

    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp, num_delay_tokens=n_delay_tokens)
    prefix_len = len(prompt_tokens)
    eos_token_id = sp.eos_id

    t_cond = model.time_embedding(mx.array([n_delay_tokens], dtype=mx.float32))
    mx.eval(t_cond)

    prompt_ids = mx.array([prompt_tokens])
    text_embeds = model.language_model.embed(prompt_ids)[0]  # [prefix_len, 3072]
    mx.eval(text_embeds)

    n_layers = len(model.language_model.layers)
    sliding_window = context_size

    def sample(logits):
        if temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / temperature).squeeze()

    def emit_pending():
        """Emit y unless already shown. Returns True on EOS, resetting cache and y."""
        nonlocal cache, y, y_emitted
        if y_emitted:
            return False
        token_id = y.item()
        if token_id == eos_token_id:
            emit("\n")
            cache = None
            y = None
            return True
        emit(sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE))
        y_emitted = True
        return False

    def decode_steps(embeds, n_to_decode):
        """Decode n_to_decode positions from embeds[0..n_to_decode-1].

        Returns (n_consumed, hit_eos). On EOS, cache and y are reset.
        """
        nonlocal y, y_emitted

        for i in range(n_to_decode):
            token_embed = model.language_model.embed(y.reshape(1, 1))[0, 0]
            step_embed = (embeds[i] + token_embed)[None, None, :]
            logits = model.decode(step_embed, t_cond, mask=None, cache=cache)
            next_y = sample(logits)
            mx.async_eval(next_y)

            # Emit the previous token while the GPU computes the next one.
            if emit_pending():
                return i, True

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            y, y_emitted = next_y, False

        # Show the newest prediction now, not when the next audio chunk arrives.
        return n_to_decode, emit_pending()

    # Audio buffer and lock
    condition = threading.Condition()
    audio_buf = np.zeros(0, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal audio_buf
        with condition:
            audio_buf = np.append(audio_buf, indata[:, 0])
            condition.notify()

    # Decoder state
    cache = None
    y = None
    y_emitted = False

    # Incremental encoder state
    audio_tail = None       # mel STFT overlap (240 samples)
    conv1_tail = None       # conv1 kernel overlap (2 frames)
    conv2_tail = None       # conv2 kernel overlap (1 frame)
    encoder_cache = None    # KV cache for encoder transformer layers
    ds_buf = None           # partial downsample group

    # Bounded buffers and counters
    pending_audio = np.zeros(0, dtype=np.float32)  # unprocessed audio remainder
    audio_embeds = None     # only undecoded embeddings
    n_audio_samples_fed = 0 # total real audio samples fed (for safe decode limit)
    n_total_decoded = 0     # total positions consumed (prefill + decode)
    first_cycle = True
    prefilled = False

    def reset_all_state():
        nonlocal audio_tail, conv1_tail, conv2_tail, encoder_cache, ds_buf
        nonlocal audio_embeds, n_audio_samples_fed
        nonlocal n_total_decoded, first_cycle, prefilled
        audio_tail = None
        conv1_tail = None
        conv2_tail = None
        encoder_cache = None
        ds_buf = None
        audio_embeds = None
        n_audio_samples_fed = 0
        n_total_decoded = 0
        first_cycle = True
        prefilled = False

    def encode_chunk(chunk):
        nonlocal first_cycle, n_audio_samples_fed, audio_tail, conv1_tail
        nonlocal conv2_tail, encoder_cache, ds_buf, audio_embeds
        n_audio_samples_fed += len(chunk)
        if first_cycle:
            chunk = np.concatenate([
                np.zeros(N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32),
                chunk,
            ])
            first_cycle = False
        mel, audio_tail = log_mel_spectrogram_step(chunk, audio_tail)
        new_embeds, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
            mel, conv1_tail, conv2_tail, encoder_cache, ds_buf
        )
        if new_embeds is not None:
            mx.eval(new_embeds)
            audio_embeds = (new_embeds if audio_embeds is None
                            else mx.concatenate([audio_embeds, new_embeds]))

    def decode_available(flushing=False):
        nonlocal cache, y, y_emitted, prefilled, audio_embeds, n_total_decoded
        if audio_embeds is None:
            return False
        if not prefilled:
            if audio_embeds.shape[0] < prefix_len:
                return False
            cache = [RotatingKVCache(sliding_window) for _ in range(n_layers)]
            prefix_embeds = (text_embeds + audio_embeds[:prefix_len])[None, :, :]
            logits = model.decode(prefix_embeds, t_cond, "causal", cache)
            mx.eval(logits, *[x for c in cache for x in (c.keys, c.values)])
            y, y_emitted = sample(logits), False
            mx.async_eval(y)
            audio_embeds = audio_embeds[prefix_len:]
            n_total_decoded = prefix_len
            prefilled = True

        n_decodable = audio_embeds.shape[0]
        if not flushing:
            safe_total = N_LEFT_PAD_TOKENS + n_audio_samples_fed // SAMPLES_PER_TOKEN
            n_decodable = min(n_decodable, safe_total - n_total_decoded)
        # With nothing new to decode, this still shows the prefill's prediction.
        n_consumed, hit_eos = decode_steps(audio_embeds, max(n_decodable, 0))
        n_total_decoded += n_consumed
        audio_embeds = (audio_embeds[n_consumed:]
                        if audio_embeds.shape[0] > n_consumed else None)
        if hit_eos:
            # Pending raw audio has not been encoded yet; retain it across EOS.
            reset_all_state()
        return hit_eos

    print("Listening... (Ctrl+C to stop)\n", flush=True)
    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="float32",
        blocksize=SAMPLES_PER_TOKEN, callback=callback,
    )
    stream.start()
    try:
        start_time = time.monotonic()
        warned_no_audio = False
        while True:
            minimum = SAMPLES_PER_TOKEN if first_cycle else batch_samples
            with condition:
                # Check under the callback's lock, including data already pending:
                # notifications before this wait cannot be lost.
                ready = condition.wait_for(
                    lambda: len(pending_audio) + len(audio_buf) >= minimum,
                    timeout=0.25,
                )
                if ready:
                    pending_audio = np.concatenate([pending_audio, audio_buf])
                    audio_buf = np.zeros(0, dtype=np.float32)
            if not ready:
                if first_cycle and not warned_no_audio and time.monotonic() - start_time > 2.0:
                    warned_no_audio = True
                    print(
                        "Warning: No audio received. Check that your terminal app "
                        "has microphone permission in System Settings > Privacy & "
                        "Security > Microphone.", flush=True,
                    )
                continue

            # Start after 80 ms; catch up in bounded batches without extra waiting.
            n_tokens = 1 if first_cycle else min(len(pending_audio) // SAMPLES_PER_TOKEN, 4)
            n_feed = n_tokens * SAMPLES_PER_TOKEN
            chunk, pending_audio = pending_audio[:n_feed], pending_audio[n_feed:]
            encode_chunk(chunk)
            decode_available()

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        with condition:
            pending_audio = np.concatenate([pending_audio, audio_buf])
            audio_buf = np.zeros(0, dtype=np.float32)

        # Flush even if interrupted before prefill or below the batch threshold.
        if n_audio_samples_fed or len(pending_audio):
            real_remaining = len(pending_audio)
            alignment = (-real_remaining) % SAMPLES_PER_TOKEN
            pending_audio = np.pad(pending_audio, (0, alignment +
                (n_delay_tokens + N_FLUSH_PAD_TOKENS) * SAMPLES_PER_TOKEN))
            while len(pending_audio):
                n_feed = SAMPLES_PER_TOKEN if first_cycle else 4 * SAMPLES_PER_TOKEN
                chunk, pending_audio = pending_audio[:n_feed], pending_audio[n_feed:]
                real_remaining = max(0, real_remaining - len(chunk))
                encode_chunk(chunk)
                if decode_available(flushing=True) and not real_remaining:
                    break

        if y is not None and not y_emitted:
            token_id = y.item()
            if token_id != eos_token_id:
                emit(sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE))
        emit("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Live streaming speech-to-text with Voxtral"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Voxtral-Mini-4B-Realtime-6bit",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--context-size", type=int, default=8192,
        help="Decoder context in tokens (must fit streaming prompt; try 512)",
    )
    parser.add_argument(
        "--delay-ms", type=int, default=480,
        help="Transcription delay: multiples of 80 from 80 to 1200, or 2400",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument("--audio-batch-ms", type=int, choices=(80, 160, 320), default=80,
                        help="Minimum live audio batch (larger batches add buffering latency)")
    args = parser.parse_args()

    stream_transcribe(
        model_path=args.model,
        temperature=args.temp,
        context_size=args.context_size,
        delay_ms=args.delay_ms,
        audio_batch_ms=args.audio_batch_ms,
    )
