from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from voxmlx import main
from voxmlx.translation import FrenchEnglish, LiveTranslation, _download

WORDS = {"Bonjour.": "Hello.", "Bonjour": "Hello", "Comment": "How", "allez-vous": "are you"}


def word_by_word(text, prefix=""):
    """Fake model translating word by word, so it always continues its own prefix."""
    english = " ".join(WORDS.get(word, word.upper()) for word in text.split())
    assert english.startswith(prefix), (english, prefix)
    return english


class TranslationTests(unittest.TestCase):
    def run_cli(self, argv, fragments):
        output = StringIO()

        def stream(**kwargs):
            for fragment in fragments:
                kwargs["on_text"](fragment)

        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "room.txt"
            with redirect_stdout(output), patch("sys.argv", ["voxmlx", *argv, "--transcript", str(transcript)]):
                with patch("voxmlx.translation.load_french_english", return_value=word_by_word):
                    with patch("voxmlx.stream.stream_transcribe", side_effect=stream):
                        main()
            return output.getvalue(), transcript.read_text()

    def test_cli_shows_english_and_saves_both_languages(self):
        shown, saved = self.run_cli(["--translate-en"], ("Bon", "jour. ", "Comment allez", "-vous"))
        self.assertEqual(shown, "Hello.\nHow are you\n")
        self.assertEqual(saved, "FR: Bonjour.\nEN: Hello.\n\nFR: Comment allez-vous\nEN: How are you\n\n")

    def test_cli_without_translation_shows_and_saves_transcription(self):
        shown, saved = self.run_cli([], ("Bonjour", " le monde", "\n"))
        self.assertEqual(shown, "Bonjour le monde\n")
        self.assertEqual(saved, shown)

    def test_idle_sentence_completes_without_waiting_for_stop(self):
        translated = Event()
        output = StringIO()

        def translate(text, prefix=""):
            translated.set()
            return "Hello"

        with redirect_stdout(output), LiveTranslation(translate, idle_seconds=0.02) as live:
            live.write("Bonjour")
            self.assertTrue(translated.wait(2))
        self.assertEqual(output.getvalue(), "Hello\n")

    def test_long_sentence_is_split_without_losing_words(self):
        events = []
        source = "bonjour " * 80
        with redirect_stdout(StringIO()), LiveTranslation(word_by_word, on_timing=events.append) as live:
            for i in range(0, len(source), 7):
                live.write(source[i:i + 7])
        self.assertGreater(len(events), 1)
        self.assertEqual(" ".join(e["source"] for e in events).split(), source.split())

    def test_translation_failure_shows_and_saves_french(self):
        output, record = StringIO(), StringIO()

        def fail(text, prefix=""):
            raise RuntimeError("test failure")

        with redirect_stdout(output), redirect_stderr(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "French text was shown"):
                with LiveTranslation(fail, record=record) as live:
                    live.write("Bonjour.\nAu revoir")
        self.assertEqual(output.getvalue(),
                         "[Translation unavailable] Bonjour.\n[Translation unavailable] Au revoir\n")
        self.assertIn("FR: Au revoir\n", record.getvalue())
        self.assertEqual(record.getvalue().count("EN: [Translation unavailable; French text preserved]"), 2)

    def test_file_translation_shows_english(self):
        output = StringIO()
        with redirect_stdout(output), patch("sys.argv", ["voxmlx", "--audio", "sample.wav", "--translate-en"]):
            with patch("voxmlx.translation.load_french_english", return_value=word_by_word):
                with patch("voxmlx.transcribe", return_value="Bonjour"):
                    main()
        self.assertEqual(output.getvalue(), "Hello\n")

    def test_model_continues_the_shown_english(self):
        calls = []
        tokenizer = SimpleNamespace(encode=lambda text: text.split(), decode=lambda pieces: " " + " ".join(pieces))
        backend = SimpleNamespace(translate_batch=lambda source, **options: calls.append((source, options)) or
                                  [SimpleNamespace(hypotheses=[["Hello,", "we"]])])
        model = object.__new__(FrenchEnglish)
        model.tokenizer, model.translator, model.beam_size = tokenizer, backend, 4
        self.assertEqual(model("Bonjour, nous", "Hello,"), "Hello, we")
        self.assertEqual(calls[0][0], [["Bonjour,", "nous"]])
        self.assertEqual(calls[0][1]["target_prefix"], [["Hello,"]])
        model("Bonjour")
        self.assertIsNone(calls[1][1]["target_prefix"])
        self.assertEqual(model("  "), "")
        self.assertEqual(len(calls), 2)

    def test_download_installs_only_a_verified_model(self):
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("translate-fr_en-1_9/model/model.bin", b"weights")
            zf.writestr("translate-fr_en-1_9/sentencepiece.model", b"pieces")
        requests = []

        def urlopen(request, timeout):
            requests.append(request)
            return BytesIO(archive.getvalue())

        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()), \
             patch("voxmlx.translation.urllib.request.urlopen", side_effect=urlopen):
            target = Path(directory) / "translate-fr_en-1_9"
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                _download(target)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with patch("voxmlx.translation.MODEL_SHA256", hashlib.sha256(archive.getvalue()).hexdigest()):
                _download(target)
            self.assertEqual((target / "model" / "model.bin").read_bytes(), b"weights")
            self.assertEqual([p.name for p in Path(directory).iterdir()], [target.name])
        # argos-net.com rejects urllib's default User-Agent.
        self.assertEqual(requests[0].get_header("User-agent"), "voxmlx")


if __name__ == "__main__":
    unittest.main()
