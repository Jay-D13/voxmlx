from pathlib import Path
import tempfile
import unittest
from unittest import mock

from voxmlx import convert


class ConvertTests(unittest.TestCase):
    def test_local_directory_is_used_directly(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output, \
                mock.patch.object(convert, "download_model") as download, \
                mock.patch.object(convert, "load_model", side_effect=RuntimeError("stop")) as load:
            with self.assertRaises(RuntimeError):
                convert.convert(hf_path=source, mlx_path=str(Path(output) / "out"))
        download.assert_not_called()
        load.assert_called_once_with(Path(source))

    def test_upload_uses_upload_folder_without_linking_local_source(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output, \
                mock.patch("huggingface_hub.HfApi", autospec=True) as api_class:
            convert._upload_to_hub(output, "user/voxtral-mlx", hf_path=source)
            readme = (Path(output) / "README.md").read_text()
        api_class.return_value.upload_folder.assert_called_once_with(
            folder_path=output, repo_id="user/voxtral-mlx", repo_type="model")
        self.assertNotIn(source, readme)
        self.assertNotIn("base_model", readme)


if __name__ == "__main__":
    unittest.main()
