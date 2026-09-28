import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TranscribeScriptTests(unittest.TestCase):
    def test_presets_save_and_display_text_without_running_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(Path(__file__).resolve().parents[1] / "transcribe.sh", root)
            uv = root / "uv"
            uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS_FILE"\nprintf "Test transcript.\\n"\n')
            uv.chmod(0o755)
            env = {**os.environ, "PATH": f"{root}:" + os.environ["PATH"],
                   "ARGS_FILE": str(root / "args")}
            for args, expected in (
                ([], ["--context-size", "512"]),
                (["--audio-batch-ms", "160"], ["--context-size", "512", "--audio-batch-ms", "160"]),
                (["--quality"], ["--model", "ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx",
                                 "--context-size", "1024", "--delay-ms", "2400"]),
                (["--quality", "--delay-ms", "960"],
                 ["--model", "ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx",
                  "--context-size", "1024", "--delay-ms", "2400", "--delay-ms", "960"]),
                (["--translate-en", "--translation-idle-ms", "750"],
                 ["--context-size", "512", "--translate-en", "--translation-idle-ms", "750"]),
                (["--translate-en"], ["--context-size", "512", "--translate-en"]),
                (["--translate-en", "--quality"],
                 ["--model", "ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx",
                  "--context-size", "1024", "--delay-ms", "2400", "--translate-en"]),
                (["--quality", "--translate-en"],
                 ["--model", "ellamind/Voxtral-Mini-4B-Realtime-8bit-mlx",
                  "--context-size", "1024", "--delay-ms", "2400", "--translate-en"]),
            ):
                with self.subTest(args=args):
                    result = subprocess.run(["bash", str(root / "transcribe.sh"), *args],
                                            env=env, capture_output=True, text=True, check=True)
                    actual = (root / "args").read_text().splitlines()
                    self.assertIn("Test transcript.\n", result.stdout)
                    if "--translate-en" in args:
                        # English goes to the terminal; voxmlx saves both languages itself.
                        self.assertEqual(actual[:-2], ["run", "--python", "3.12", "--no-editable",
                                                       "--extra", "translation", "voxmlx", *expected])
                        self.assertEqual(actual[-2], "--transcript")
                        self.assertEqual(Path(actual[-1]).parent.resolve(), (root / "transcripts").resolve())
                    else:
                        self.assertEqual(actual, ["run", "--python", "3.12", "--no-editable", "voxmlx", *expected])
                        self.assertTrue(any("Test transcript.\n" in p.read_text()
                                            for p in (root / "transcripts").glob("*.txt")))
            (root / "args").unlink()
            subprocess.run(["bash", str(root / "transcribe.sh"), "--help"],
                           env=env, capture_output=True, check=True)
            self.assertFalse((root / "args").exists())


if __name__ == "__main__":
    unittest.main()
