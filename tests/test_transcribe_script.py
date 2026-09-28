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
            uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS_FILE"\n'
                          'printf "%s\\n" "${XDG_CACHE_HOME-unset}" "${XDG_DATA_HOME-unset}" "$VOXMLX_CACHE_DIR" > "$ENV_FILE"\n'
                          'printf "Test transcript.\\n"\n')
            uv.chmod(0o755)
            inherited = {k: v for k, v in os.environ.items()
                         if k not in ("XDG_CACHE_HOME", "XDG_DATA_HOME", "VOXMLX_CACHE_DIR")}
            env = {**inherited, "PATH": f"{root}:" + os.environ["PATH"],
                   "ARGS_FILE": str(root / "args"), "ENV_FILE": str(root / "env")}
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
                    xdg_cache, xdg_data, cache = (root / "env").read_text().splitlines()
                    # Only voxmlx's cache moves into the project; uv keeps its own cache and Pythons.
                    self.assertEqual((xdg_cache, xdg_data), ("unset", "unset"))
                    self.assertEqual(Path(cache).resolve(), (root / ".cache/voxmlx").resolve())
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
