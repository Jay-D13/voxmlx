__version__ = "0.0.2"

import argparse
import math
from contextlib import ExitStack
from pathlib import Path

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

from .generate import generate
from .weights import download_model, load_model as _load_weights


def _load_tokenizer(model_path: Path) -> Tekkenizer:
    tekken_path = model_path / "tekken.json"
    return Tekkenizer.from_file(str(tekken_path))


def _build_prompt_tokens(
    sp: Tekkenizer,
    n_left_pad_tokens: int = 32,
    num_delay_tokens: int = 6,
) -> tuple[list[int], int]:
    streaming_pad = sp.get_special_token("[STREAMING_PAD]")
    prefix_len = n_left_pad_tokens + num_delay_tokens
    tokens = [sp.bos_id] + [streaming_pad] * prefix_len
    return tokens, num_delay_tokens


def _delay_tokens(delay_ms: int) -> int:
    if delay_ms not in (*range(80, 1201, 80), 2400):
        raise ValueError("delay_ms must be a multiple of 80 from 80 to 1200, or 2400")
    return delay_ms // 80


def load_model(model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"):
    if not Path(model_path).exists():
        model_path = download_model(model_path)
    else:
        model_path = Path(model_path)

    model, config = _load_weights(model_path)
    sp = _load_tokenizer(model_path)
    return model, sp, config


def transcribe(
    audio_path: str,
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
    delay_ms: int = 2400,
) -> str:
    # Latency doesn't matter for files, so default to the longest, most accurate delay.
    n_delay_tokens = _delay_tokens(delay_ms)
    model, sp, config = load_model(model_path)

    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp, num_delay_tokens=n_delay_tokens)

    output_tokens = generate(
        model,
        audio_path,
        prompt_tokens,
        n_delay_tokens=n_delay_tokens,
        temperature=temperature,
        eos_token_id=sp.eos_id,
    )

    return sp.decode(output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)


def main():
    parser = argparse.ArgumentParser(description="Voxtral Mini Realtime speech-to-text")
    parser.add_argument("--audio", default=None, help="Path to audio file (omit to stream from mic)")
    parser.add_argument("--model", default="mlx-community/Voxtral-Mini-4B-Realtime-6bit", help="Model path or HF model ID")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--context-size", type=int, default=8192, help="Live decoder context in tokens (must fit streaming prompt; try 512)")
    parser.add_argument("--delay-ms", type=int, help="Transcription delay: multiples of 80 from 80 to 1200, or 2400 (default: 480 live, 2400 for --audio)")
    parser.add_argument("--translate-en", action="store_true", help="Translate French speech into English locally; show English live")
    parser.add_argument("--transcript", type=Path, help="Append the transcript to this file (French and English pairs with --translate-en)")
    parser.add_argument("--audio-batch-ms", type=int, choices=(80, 160, 320), default=80, help="Minimum live audio batch in ms (larger batches add buffering latency)")
    parser.add_argument("--translation-idle-ms", type=float, default=1500, help="Pause before completing an unpunctuated sentence (default: 1500 ms)")
    args = parser.parse_args()
    if not math.isfinite(args.translation_idle_ms) or args.translation_idle_ms <= 0:
        parser.error("--translation-idle-ms must be positive and finite")
    # Without --delay-ms, each mode keeps its own default.
    delay = {} if args.delay_ms is None else {"delay_ms": args.delay_ms}

    with ExitStack() as stack:
        record = None
        if args.transcript is not None:
            record = stack.enter_context(args.transcript.open("a", encoding="utf-8"))
        if args.translate_en:
            from .translation import LiveTranslation, load_french_english

            translation = LiveTranslation(load_french_english(), idle_seconds=args.translation_idle_ms / 1000,
                                          record=record)
            on_text = stack.enter_context(translation).write
        else:
            def on_text(text):
                print(text, end="", flush=True)
                if record is not None:
                    record.write(text)
                    record.flush()

        if args.audio is not None:
            text = transcribe(
                args.audio,
                model_path=args.model,
                temperature=args.temp,
                **delay,
            )
            on_text(text if args.translate_en else text + "\n")
        else:
            from .stream import stream_transcribe

            stream_transcribe(
                model_path=args.model,
                temperature=args.temp,
                context_size=args.context_size,
                audio_batch_ms=args.audio_batch_ms,
                **delay,
                on_text=on_text,
            )
