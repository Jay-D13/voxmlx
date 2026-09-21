from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from threading import Event
import unittest
from unittest.mock import patch

from voxmlx import main
from voxmlx.translation import LiveTranslation


class TranslationTests(unittest.TestCase):
    def test_cli_stream_translates_fragments_and_flushes_final_phrase(self):
        output = StringIO()

        def stream(**kwargs):
            for fragment in ("Bon", "jour. ", "Comment allez", "-vous"):
                kwargs["on_text"](fragment)

        def translate(text):
            return {"Bonjour.": "Hello.", "Comment allez-vous": "How are you"}[text]

        with redirect_stdout(output), patch("sys.argv", ["voxmlx", "--translate-en"]):
            with patch("voxmlx.translation.load_french_english", return_value=translate):
                with patch("voxmlx.stream.stream_transcribe", side_effect=stream):
                    main()
        self.assertEqual(output.getvalue(), "FR: Bonjour.\nEN: Hello.\n\nFR: Comment allez-vous\nEN: How are you\n\n")

    def test_idle_phrase_translates_without_waiting_for_stop(self):
        translated = Event()
        output = StringIO()

        def translate(text):
            translated.set()
            return "Hello"

        with redirect_stdout(output), LiveTranslation(translate, idle_seconds=0.02) as live:
            live.write("Bonjour")
            self.assertTrue(translated.wait(2))
        self.assertIn("EN: Hello", output.getvalue())

    def test_long_phrase_is_split_without_losing_words(self):
        chunks = []
        source = "bonjour " * 80
        with redirect_stdout(StringIO()), LiveTranslation(lambda text: chunks.append(text) or "Hello") as live:
            for i in range(0, len(source), 7):
                live.write(source[i:i + 7])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).split(), source.split())

    def test_translation_failure_preserves_french_and_reports_failure(self):
        output = StringIO()

        def fail(text):
            raise RuntimeError("test failure")

        with redirect_stdout(output), redirect_stderr(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "French text was preserved"):
                with LiveTranslation(fail) as live:
                    live.write("Bonjour.\nAu revoir")
        self.assertIn("FR: Bonjour.", output.getvalue())
        self.assertIn("FR: Au revoir", output.getvalue())
        self.assertEqual(output.getvalue().count("Translation unavailable"), 2)

    def test_file_translation_uses_same_bilingual_output(self):
        output = StringIO()
        with redirect_stdout(output), patch("sys.argv", ["voxmlx", "--audio", "sample.wav", "--translate-en"]):
            with patch("voxmlx.translation.load_french_english", return_value=lambda text: "Hello"):
                with patch("voxmlx.transcribe", return_value="Bonjour"):
                    main()
        self.assertEqual(output.getvalue(), "FR: Bonjour\nEN: Hello\n\n")


if __name__ == "__main__":
    unittest.main()
