from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import face_to_ck3_gui


class FaceToCK3GUITests(unittest.TestCase):
    def test_resource_root_uses_pyinstaller_bundle_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            sys, "_MEIPASS", directory, create=True
        ):
            self.assertEqual(
                face_to_ck3_gui.resource_root(), face_to_ck3_gui.Path(directory)
            )

    def test_wsl_fontconfig_is_selected_when_windows_fonts_exist(self) -> None:
        with mock.patch.object(face_to_ck3_gui.sys, "platform", "linux"), mock.patch.object(
            face_to_ck3_gui.Path, "is_dir", return_value=True
        ), mock.patch.object(face_to_ck3_gui.Path, "is_file", return_value=True), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            face_to_ck3_gui.prepare_wsl_fonts()
            self.assertEqual(
                os.environ["FONTCONFIG_FILE"], str(face_to_ck3_gui.WSL_FONTCONFIG)
            )

    def test_wsl_clipboard_uses_windows_clip_without_touching_tk(self) -> None:
        root = mock.Mock()
        completed = mock.Mock(returncode=0, stderr=b"")
        with mock.patch.object(face_to_ck3_gui, "is_wsl", return_value=True), mock.patch.object(
            face_to_ck3_gui, "WINDOWS_CLIP", face_to_ck3_gui.Path("/bin/true")
        ), mock.patch.object(
            face_to_ck3_gui.Path, "is_file", return_value=True
        ), mock.patch.object(
            face_to_ck3_gui.subprocess, "run", return_value=completed
        ) as run:
            backend = face_to_ck3_gui.copy_text_to_clipboard("ruler={\n}\n", root)
        self.assertEqual(backend, "windows_clip.exe")
        root.clipboard_clear.assert_not_called()
        root.clipboard_append.assert_not_called()
        root.update.assert_not_called()
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["input"], b"ruler={\n}\n")
        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)

    def test_native_clipboard_uses_idle_update_only(self) -> None:
        root = mock.Mock()
        with mock.patch.object(face_to_ck3_gui, "is_wsl", return_value=False):
            backend = face_to_ck3_gui.copy_text_to_clipboard("dna", root)
        self.assertEqual(backend, "tk")
        root.clipboard_clear.assert_called_once_with()
        root.clipboard_append.assert_called_once_with("dna")
        root.update_idletasks.assert_called_once_with()
        root.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
