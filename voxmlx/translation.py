"""Local French-to-English translation, independent of the MLX decoding loop."""

import os
from queue import Empty, Queue
import re
import sys
from threading import Thread
import time


def load_french_english():
    # Keep this feature local even if Argos has a remote provider configured.
    os.environ["ARGOS_MODEL_PROVIDER"] = "OPENNMT"
    os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
    # Stanza refreshes its remote index on startup, even with cached weights.
    os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
    os.environ.setdefault("ARGOS_INTRA_THREADS", "2")
    try:
        from argostranslate import package, translate
    except ImportError as exc:
        raise RuntimeError("Install translation support with uv sync --extra translation --no-editable") from exc

    if not any(p.from_code == "fr" and p.to_code == "en"
               for p in package.get_installed_packages()):
        print("Downloading the local French → English translation model...", file=sys.stderr, flush=True)
        package.update_package_index()
        model = next((p for p in package.get_available_packages()
                      if p.from_code == "fr" and p.to_code == "en"), None)
        if model is None:
            raise RuntimeError("No French-to-English model available in the Argos package index")
        package.install_from_path(model.download())

    translator = translate.get_translation_from_codes("fr", "en")
    # Warm up before microphone capture; also fetch any sentence-splitting assets.
    translator.translate("Bonjour.")
    print("Local French → English translation ready.", file=sys.stderr, flush=True)
    return translator.translate


class LiveTranslation:
    def __init__(self, translate, idle_seconds=1.5, max_seconds=5.0):
        self.translate = translate
        self.idle_seconds = idle_seconds
        self.max_seconds = max_seconds
        # ponytail: preserve all text in memory if translation falls behind;
        # use a disk-backed queue if sustained translation backlog becomes a problem.
        self.queue = Queue()
        self.error = None
        self.thread = Thread(target=self._run, name="french-to-english")

    def __enter__(self):
        self.thread.start()
        return self

    def write(self, text):
        if text:
            self.queue.put(text)

    def __exit__(self, exc_type, exc, traceback):
        self.queue.put(None)
        self.thread.join()
        if self.error is not None and exc_type is None:
            raise RuntimeError("Translation failed; the French text was preserved above") from self.error

    def _emit(self, text):
        text = text.strip()
        if not text:
            return
        print(f"FR: {text}", flush=True)
        try:
            print(f"EN: {self.translate(text)}\n", flush=True)
        except Exception as exc:
            self.error = exc
            print("EN: [Translation unavailable; French text preserved]\n", flush=True)
            print(f"Translation error: {exc}", file=sys.stderr, flush=True)

    def _run(self):
        pending = ""
        started = time.monotonic()
        while True:
            try:
                text = self.queue.get(timeout=self.idle_seconds)
            except Empty:
                self._emit(pending)
                pending = ""
                continue
            if text is None:
                self._emit(pending)
                return
            if not pending:
                started = time.monotonic()
            pending += text
            # Sentence boundaries, including the final newline from Voxtral.
            while match := re.search(r'[.!?…]["»”]*\s|\n', pending):
                self._emit(pending[:match.end()])
                pending = pending[match.end():]
                started = time.monotonic()
            # Keep long, unpunctuated speech live, splitting at a word boundary.
            if len(pending) >= 240 or time.monotonic() - started >= self.max_seconds:
                cut = pending.rfind(" ")
                if cut > 0:
                    self._emit(pending[:cut])
                    pending = pending[cut + 1:]
                    started = time.monotonic()
