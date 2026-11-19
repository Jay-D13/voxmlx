from fnmatch import fnmatch
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from voxmlx.weights import download_model

# File listings of mistralai/Voxtral-Mini-4B-Realtime-2602 and of a sharded converted MLX repo.
ORIGINAL_REPO = ["README.md", "config.json", "consolidated.safetensors", "model.safetensors",
                 "params.json", "tekken.json"]
CONVERTED_REPO = ["README.md", "config.json", "model-00001-of-00002.safetensors",
                  "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tekken.json"]


def download(repo_files):
    """Run download_model against a fake Hub repo and return the files it fetched."""
    with tempfile.TemporaryDirectory() as directory:
        def fake_snapshot_download(model_id, allow_patterns):
            for name in repo_files:
                if any(fnmatch(name, pattern) for pattern in allow_patterns):
                    (Path(directory) / name).write_text("")
            return directory

        with mock.patch("voxmlx.weights.snapshot_download", fake_snapshot_download):
            path = download_model("org/repo")
        return sorted(p.name for p in path.iterdir())


class DownloadModelTests(unittest.TestCase):
    def test_original_repo_skips_transformers_weights(self):
        self.assertEqual(download(ORIGINAL_REPO), ["consolidated.safetensors", "params.json", "tekken.json"])

    def test_converted_repo_fetches_its_weights(self):
        self.assertEqual(download(CONVERTED_REPO), sorted(CONVERTED_REPO[1:]))


if __name__ == "__main__":
    unittest.main()
