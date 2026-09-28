"""Local French-to-English translation, independent of the MLX decoding loop.

English streams like a simultaneous interpreter's. After each completed French
word, the open sentence is translated again, continuing from the English already
shown; new words appear once two consecutive translations agree (local
agreement). Shown words are never revised. A sentence is completed when it ends,
pauses, grows too long, or the stream stops.
"""

import hashlib
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import sys
import tempfile
from threading import Thread
import time
import urllib.request
import zipfile

SENTENCE_END = re.compile(r'[.!?…]["»”]*\s|\n')
MAX_SENTENCE_CHARS = 240
_WORDS = re.compile(r"\s*\S+")

# Argos Translate's French-to-English package, pinned so translations are reproducible.
MODEL_NAME = "translate-fr_en-1_9"
MODEL_URL = f"https://argos-net.com/v1/{MODEL_NAME}.argosmodel"
MODEL_SHA256 = "3b3052fee6bb1e8e8e632a26a723eb2a2c7710dfe73ba61ffd9b83e85d4f14c1"


def model_dir():
    """Model location: $VOXMLX_CACHE_DIR, else ~/.cache/voxmlx (respecting $XDG_CACHE_HOME)."""
    cache = os.environ.get("VOXMLX_CACHE_DIR") or (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "voxmlx")
    return Path(cache) / MODEL_NAME


class _Tokenizer:
    """SentencePiece, decoded exactly as Argos does so translations are unchanged."""

    def __init__(self, path):
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(path))

    def encode(self, text):
        return self.processor.encode(text, out_type=str)

    def decode(self, pieces):
        return self.processor.decode_pieces(pieces).replace("▁", " ").replace("_", " ")


class FrenchEnglish:
    """Argos French-to-English model that can continue English already shown."""

    def __init__(self, path, *, threads=2, beam_size=4):
        import ctranslate2

        # Source and target share this SentencePiece model, so shown English re-encodes exactly.
        self.tokenizer = _Tokenizer(path / "sentencepiece.model")
        self.beam_size = beam_size
        # Argos's defaults, except two CPU threads.
        self.translator = ctranslate2.Translator(
            str(path / "model"), device="cpu", inter_threads=1, intra_threads=threads,
            compute_type="auto",
        )

    def __call__(self, text, prefix=""):
        """Translate text into English that starts with prefix."""
        if not text.strip():
            return ""
        result = self.translator.translate_batch(
            [self.tokenizer.encode(text)],
            target_prefix=[self.tokenizer.encode(prefix)] if prefix else None,
            # Argos's settings for a single translation.
            beam_size=self.beam_size, length_penalty=0.2, replace_unknowns=True,
        )
        return self.tokenizer.decode(result[0].hypotheses[0]).lstrip()


def _download(target):
    """Download, verify, and unpack the model; target appears only once complete."""
    print("Downloading the local French → English translation model...", file=sys.stderr, flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive, digest = Path(tmp) / "model.zip", hashlib.sha256()
        # argos-net.com rejects urllib's default User-Agent.
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "voxmlx"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as file:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                file.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError(f"Checksum mismatch for {MODEL_URL}; the model was not installed")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        (Path(tmp) / MODEL_NAME).rename(target)


def load_french_english(*, allow_download=True, warmup=True, threads=2, beam_size=4):
    try:
        import ctranslate2  # Checked before any download.
    except ImportError as exc:
        raise RuntimeError("Install translation support with uv sync --extra translation --no-editable") from exc

    path = model_dir()
    if not (path / "model").is_dir():
        if not allow_download:
            raise RuntimeError(f"Cached French-to-English model is required in {path}; run --translate-en once online")
        _download(path)

    translate = FrenchEnglish(path, threads=threads, beam_size=beam_size)
    # Warm up before microphone capture.
    if warmup:
        translate("Bonjour.")
        print("Local French → English translation ready.", file=sys.stderr, flush=True)
    return translate


class _Pending:
    """French text with per-character arrival times, retained across sentence splits."""
    def __init__(self, idle):
        self.text = ""
        self.arrivals = []
        self.idle = idle

    def append(self, text, arrived):
        self.text += text
        self.arrivals.extend([arrived] * len(text))

    def deadline(self):
        return self.arrivals[-1] + self.idle if self.text else None

    def complete(self):
        """French up to the last whole word; the word still arriving may change."""
        return self.text[:self.text.rfind(" ") + 1].strip()

    def ready(self, now):
        """Return (cut, reason, ready_at) once the open sentence should be completed."""
        if not self.text:
            return None
        if match := SENTENCE_END.search(self.text):
            return match.end(), "sentence", self.arrivals[match.end() - 1]
        # Cut at the last whole word: shown English may already cover every earlier word.
        cut = self.text.rfind(" ")
        if len(self.text) >= MAX_SENTENCE_CHARS and cut > 0:
            return cut + 1, "length", self.arrivals[-1]
        idle = self.arrivals[-1] + self.idle
        if idle <= now:
            return len(self.text), "idle", idle
        return None

    def take(self, cut):
        text, arrivals = self.text[:cut], self.arrivals[:cut]
        self.text, self.arrivals = self.text[cut:], self.arrivals[cut:]
        return text, arrivals


class LiveTranslation:
    """Stream English to stdout; optionally save French and English pairs to record."""

    def __init__(self, translate, idle_seconds=1.5, *, record=None, on_timing=None):
        if not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("Translation pause must be positive and finite")
        self.translate = translate
        self.idle_seconds = idle_seconds
        self.record = record
        self.on_timing = on_timing
        # Preserve all text in memory if translation falls behind;
        # use a disk-backed queue if sustained translation backlog becomes a problem.
        self.queue = Queue()
        self.error = None
        self.thread = Thread(target=self._run, name="french-to-english")

    def __enter__(self):
        self.thread.start()
        return self

    def write(self, text):
        if text:
            self.queue.put((text, time.monotonic()))

    def __exit__(self, exc_type, exc, traceback):
        self.queue.put((None, time.monotonic()))
        self.thread.join()
        if self.error is not None and exc_type is None:
            raise RuntimeError("Translation failed; the French text was shown in its place") from self.error

    def _start_sentence(self):
        self.shown = ""       # English on screen for the open sentence
        self.previous = None  # latest partial translation, awaiting agreement
        self.translated = ""  # French it translated
        self.timing = dict(calls=0, inference_seconds=0.0, first_english=None, error=None)

    def _show(self, text, english=True):
        print(text, end="", flush=True)
        if english and self.timing["first_english"] is None:
            self.timing["first_english"] = time.monotonic()

    def _translate(self, source):
        start = time.monotonic()
        try:
            return self.translate(source, self.shown)
        except Exception as exc:
            self.error = exc
            self.timing["error"] = str(exc)
            print(f"Translation error: {exc}", file=sys.stderr, flush=True)
            return None
        finally:
            end = time.monotonic()
            self.timing["calls"] += 1
            self.timing["inference_seconds"] += end - start
            self.timing.update(inference_start=start, inference_end=end)

    def _agree(self, source):
        """Show the English on which the last two translations of the open sentence agree."""
        if not source or source == self.translated or self.timing["error"]:
            return
        self.translated = source
        english = self._translate(source)
        if english is None:
            return
        # A translated partial sentence gains closing punctuation; that is not agreement.
        english = re.sub(r"[.!?…]+$", "", english.rstrip())
        if self.previous is not None:
            agreed = ""
            for ours, theirs in zip(_WORDS.findall(self.previous), _WORDS.findall(english)):
                if ours != theirs:
                    break
                agreed += ours
            if len(agreed) > len(self.shown) and agreed.startswith(self.shown):
                self._show(agreed[len(self.shown):])
                self.shown = agreed
        self.previous = english

    def _complete(self, pending, cut, reason, ready_at):
        text, arrivals = pending.take(cut)
        source = text.strip()
        if source:
            english = None if self.timing["error"] else self._translate(source)
            if english is None:
                self._show(("\n" if self.shown else "") + f"[Translation unavailable] {source}\n", english=False)
            elif english.startswith(self.shown):
                self._show(english[len(self.shown):] + "\n")
            else:  # Not expected with a forced prefix; repeat the sentence rather than garble it.
                self._show(f"\n{english}\n")
            if self.record is not None:
                saved = english if english is not None else "[Translation unavailable; French text preserved]"
                try:
                    self.record.write(f"FR: {source}\nEN: {saved}\n\n")
                    self.record.flush()
                except OSError as exc:
                    self.error = exc
                    print(f"Transcript write error: {exc}", file=sys.stderr, flush=True)
            self.phrase_id += 1
            self.timing.update(phrase_id=self.phrase_id, source_chars=len(text), source=source,
                               english=english, first_arrival=arrivals[0], last_arrival=arrivals[-1],
                               ready_at=ready_at, reason=reason, output_end=time.monotonic())
            if self.on_timing is not None:
                try:
                    self.on_timing(self.timing)
                except Exception as exc:
                    # Instrumentation must not terminate the worker and lose source text.
                    self.error = exc
                    print(f"Translation timing callback error: {exc}", file=sys.stderr, flush=True)
        self._start_sentence()

    def _run(self):
        pending = _Pending(self.idle_seconds)
        self.phrase_id = 0
        self._start_sentence()

        def flush_ready(now):
            while ready := pending.ready(now):
                self._complete(pending, *ready)

        while True:
            deadline = pending.deadline()
            timeout = None if deadline is None else max(0, deadline - time.monotonic())
            try:
                text, arrived = self.queue.get(timeout=timeout)
            except Empty:
                flush_ready(time.monotonic())
                continue
            # Replay source time before wall time: queued fragments that arrived
            # before a deadline belong to that sentence even when inference is slow.
            flush_ready(arrived)
            if text is None:
                if pending.text:
                    self._complete(pending, len(pending.text), "shutdown", arrived)
                return
            pending.append(text, arrived)
            flush_ready(arrived)
            # Retranslate only when caught up, so a slow model skips intermediate words.
            if self.queue.empty():
                self._agree(pending.complete())
