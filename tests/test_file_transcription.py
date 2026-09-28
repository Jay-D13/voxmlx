import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from voxmlx import main, transcribe
from voxmlx.audio import SAMPLES_PER_TOKEN

generate_module = importlib.import_module("voxmlx.generate")
TOKENIZER = SimpleNamespace(bos_id=1, eos_id=2, get_special_token=lambda _: 7,
                            decode=lambda tokens, special_token_policy=None: "Bonjour")


class Stop(Exception):
    pass


class FileTranscriptionTests(unittest.TestCase):
    def transcribe_delays(self, run):
        """Run with the model stubbed out; return the delay tokens each generate call received."""
        with patch("voxmlx.load_model", return_value=(None, TOKENIZER, {})), \
             patch("voxmlx.generate", return_value=[]) as generate:
            run()
        for call in generate.call_args_list:
            # BOS, 32 left-pad tokens, then the delay tokens.
            self.assertEqual(len(call.args[2]), 1 + 32 + call.kwargs["n_delay_tokens"])
        return [call.kwargs["n_delay_tokens"] for call in generate.call_args_list]

    def test_files_default_to_the_longest_delay(self):
        self.assertEqual(self.transcribe_delays(lambda: transcribe("talk.wav")), [30])
        self.assertEqual(self.transcribe_delays(lambda: transcribe("talk.wav", delay_ms=480)), [6])

    def test_invalid_file_delay_fails_before_loading_model(self):
        with patch("voxmlx.load_model") as load:
            for delay in (0, 100, 1280):
                with self.subTest(delay=delay), self.assertRaises(ValueError):
                    transcribe("talk.wav", delay_ms=delay)
        load.assert_not_called()

    def test_right_pad_outlasts_the_delay(self):
        padded = []

        def mel(audio):
            padded.append(len(audio) // SAMPLES_PER_TOKEN)
            raise Stop

        with patch.object(generate_module, "load_audio", return_value=np.zeros(10 * SAMPLES_PER_TOKEN)), \
             patch.object(generate_module, "log_mel_spectrogram", side_effect=mel):
            for delay in (6, 30):
                with self.assertRaises(Stop):
                    generate_module.generate(None, "talk.wav", [], n_delay_tokens=delay)
        # 32 left-pad tokens, 10 audio tokens, then the delay plus 11 flush tokens.
        self.assertEqual(padded, [32 + 10 + 17, 32 + 10 + 41])

    def test_cli_uses_each_modes_default_delay(self):
        def cli(*argv):
            with patch("sys.argv", ["voxmlx", *argv]), patch("builtins.print"):
                main()

        self.assertEqual(self.transcribe_delays(lambda: cli("--audio", "talk.wav")), [30])
        self.assertEqual(self.transcribe_delays(lambda: cli("--audio", "talk.wav", "--delay-ms", "960")), [12])
        with patch("voxmlx.stream.stream_transcribe") as stream:
            cli()
        # Live mode keeps stream_transcribe's own 480 ms default.
        self.assertNotIn("delay_ms", stream.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
