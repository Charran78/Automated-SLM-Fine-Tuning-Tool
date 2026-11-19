from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import remote_adapter


MODEL_ID = "org/example-model"
REVISION = "a" * 40


def artifact_files(
    *,
    model_id: str = MODEL_ID,
    resolved_revision: str | None = REVISION,
    requested_revision: str | None = None,
    adapter_model_id: str | None = None,
) -> dict[str, bytes]:
    manifest = {
        "model": model_id,
        "model_resolved_revision": resolved_revision,
        "model_requested_revision": requested_revision,
    }
    config = {
        "base_model_name_or_path": adapter_model_id or model_id,
        "revision": REVISION,
    }
    return {
        "manifest.json": json.dumps(manifest).encode(),
        "adapter/adapter_config.json": json.dumps(config).encode(),
        "adapter/adapter_model.safetensors": b"weights",
        "adapter/tokenizer.json": b"{}",
        "adapter/tokenizer_config.json": b"{}",
    }


def write_zip(
    path: Path,
    files: dict[str, bytes] | None = None,
    *,
    extra_entries: tuple[tuple[str, bytes, int], ...] = (),
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in (files or artifact_files()).items():
            archive.writestr(name, content)
        for name, content, mode in extra_entries:
            entry = zipfile.ZipInfo(name)
            entry.external_attr = mode << 16
            archive.writestr(entry, content)


class RemoteArtifactInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.zip_path = Path(self.temp_dir.name) / "artifact.zip"

    def test_inspects_valid_archive_and_hash_without_modifying_it(self) -> None:
        write_zip(self.zip_path)
        original_hash = hashlib.sha256(self.zip_path.read_bytes()).hexdigest()

        metadata = remote_adapter.inspect_remote_artifact(self.zip_path)

        self.assertEqual(metadata.model_id, MODEL_ID)
        self.assertEqual(metadata.recorded_revision, REVISION)
        self.assertEqual(metadata.adapter_revision, REVISION)
        self.assertEqual(metadata.archive_sha256, original_hash)
        self.assertEqual(metadata.adapter_root, "adapter")
        self.assertEqual(
            hashlib.sha256(self.zip_path.read_bytes()).hexdigest(), original_hash
        )

    def test_rejects_corrupted_crc(self) -> None:
        write_zip(self.zip_path)
        contents = bytearray(self.zip_path.read_bytes())
        contents[-20] ^= 0xFF
        self.zip_path.write_bytes(contents)

        with self.assertRaises(remote_adapter.RemoteArtifactError):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_missing_required_files(self) -> None:
        files = artifact_files()
        del files["adapter/adapter_model.safetensors"]
        write_zip(self.zip_path, files)

        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "peso"):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_path_traversal(self) -> None:
        write_zip(
            self.zip_path,
            extra_entries=(("../outside.txt", b"x", stat.S_IFREG | 0o600),),
        )

        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "traversal"):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_symlink(self) -> None:
        write_zip(
            self.zip_path,
            extra_entries=(("adapter/link", b"target", stat.S_IFLNK | 0o777),),
        )

        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "enlace"):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_duplicate_members(self) -> None:
        write_zip(
            self.zip_path,
            extra_entries=(
                ("adapter/tokenizer.json", b"second", stat.S_IFREG | 0o600),
            ),
        )

        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "duplic"):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_archive_limits(self) -> None:
        write_zip(self.zip_path)

        with patch.object(remote_adapter, "MAX_ARCHIVE_MEMBERS", 2):
            with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "límite"):
                remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_manifest_and_adapter_model_mismatch(self) -> None:
        write_zip(self.zip_path, artifact_files(adapter_model_id="other/model"))

        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "coinciden"):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_rejects_manifest_and_adapter_revision_mismatch(self) -> None:
        files = artifact_files()
        config = json.loads(files["adapter/adapter_config.json"])
        config["revision"] = "b" * 40
        files["adapter/adapter_config.json"] = json.dumps(config).encode()
        write_zip(self.zip_path, files)

        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "revisión SHA"):
            remote_adapter.inspect_remote_artifact(self.zip_path)

    def test_resolves_current_revision_with_huggingface_api(self) -> None:
        api_class = Mock(
            return_value=SimpleNamespace(
                model_info=Mock(return_value=SimpleNamespace(sha=REVISION.upper()))
            )
        )
        with patch.dict(
            sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=api_class)}
        ):
            resolved = remote_adapter.resolve_current_revision(MODEL_ID)

        self.assertEqual(resolved, REVISION)
        api_class.return_value.model_info.assert_called_once_with(MODEL_ID)


class RemoteArtifactConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.zip_path = self.root / "artifact.zip"
        write_zip(self.zip_path)
        self.output_dir = self.root / "converted"
        self.pipeline = Mock()
        self.pipeline.export_to_gguf.side_effect = self._export_files
        self.pipeline.register_in_ollama = Mock()
        self.model_loader = Mock(return_value=Mock(name="base_model"))
        self.adapter_loader = Mock(return_value=Mock(name="peft_model"))
        self.tokenizer_loader = Mock(return_value=Mock(name="tokenizer"))
        self.pipeline_factory = Mock(return_value=self.pipeline)
        self.dependencies = SimpleNamespace(
            torch=SimpleNamespace(float32="float32"),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self.model_loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=self.tokenizer_loader),
            PeftModel=SimpleNamespace(from_pretrained=self.adapter_loader),
        )

    def _export_files(self, *, export_dir: str, quant_type: str) -> str:
        destination = Path(export_dir)
        destination.mkdir()
        (destination / "model-q4_k_m.gguf").write_bytes(b"gguf")
        modelfile = destination / "Modelfile"
        modelfile.write_text("FROM model-q4_k_m.gguf\n", encoding="utf-8")
        return str(modelfile)

    def convert(self, **overrides: object) -> remote_adapter.ConversionResult:
        arguments: dict[str, object] = {
            "archive_path": self.zip_path,
            "output_dir": self.output_dir,
            "model_revision": None,
            "acknowledge_unrecorded_revision": False,
            "quant_type": "q4_k_m",
            "ollama_model_name": "example-ft",
            "register_ollama": False,
        }
        arguments.update(overrides)
        with (
            patch.object(
                remote_adapter,
                "_load_pipeline_class",
                return_value=self.pipeline_factory,
            ),
            patch.object(
                remote_adapter,
                "_load_conversion_dependencies",
                return_value=self.dependencies,
            ),
        ):
            return remote_adapter.convert_remote_artifact(**arguments)

    def test_conversion_isolated_pinned_and_preserves_archive(self) -> None:
        expected_hash = hashlib.sha256(self.zip_path.read_bytes()).hexdigest()

        result = self.convert()

        self.assertEqual(result.output_dir, self.output_dir)
        self.assertEqual(result.archive_sha256, expected_hash)
        self.assertEqual(
            hashlib.sha256(self.zip_path.read_bytes()).hexdigest(), expected_hash
        )
        self.pipeline_factory.assert_called_once()
        self.assertFalse(self.pipeline.use_unsloth)
        self.model_loader.assert_called_once_with(
            MODEL_ID, revision=REVISION, torch_dtype="float32", device_map="cpu"
        )
        self.adapter_loader.assert_called_once()
        self.tokenizer_loader.assert_called_once()
        manifest = json.loads(
            (self.output_dir / "conversion_manifest.json").read_text()
        )
        self.assertEqual(manifest["model_revision"], REVISION)
        self.assertEqual(manifest["archive_sha256"], expected_hash)

    def test_missing_recorded_revision_requires_acknowledged_explicit_sha(self) -> None:
        write_zip(self.zip_path, artifact_files(resolved_revision=None))

        with self.assertRaisesRegex(
            remote_adapter.RemoteArtifactError, "explícitamente"
        ):
            self.convert(model_revision=REVISION)
        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "40"):
            self.convert(model_revision="main", acknowledge_unrecorded_revision=True)

    def test_requested_revision_is_not_a_resolved_revision(self) -> None:
        write_zip(
            self.zip_path,
            artifact_files(resolved_revision=None, requested_revision=REVISION),
        )

        with self.assertRaisesRegex(
            remote_adapter.RemoteArtifactError, "explícitamente"
        ):
            self.convert(model_revision=REVISION)

    def test_unrecorded_revision_converts_after_explicit_acknowledgement(self) -> None:
        write_zip(self.zip_path, artifact_files(resolved_revision=None))

        result = self.convert(
            model_revision=REVISION,
            acknowledge_unrecorded_revision=True,
        )

        self.assertEqual(result.model_revision, REVISION)

    def test_registered_revision_cannot_be_replaced(self) -> None:
        with self.assertRaisesRegex(remote_adapter.RemoteArtifactError, "registrada"):
            self.convert(model_revision="b" * 40)

    def test_registered_revision_does_not_need_acknowledgement(self) -> None:
        result = self.convert(model_revision=REVISION)

        self.assertEqual(result.model_revision, REVISION)

    def test_existing_output_directory_is_never_reused(self) -> None:
        self.output_dir.mkdir()

        with self.assertRaisesRegex(remote_adapter.ConversionError, "ya existe"):
            self.convert()

        self.pipeline.export_to_gguf.assert_not_called()

    def test_ollama_conflict_is_rejected_without_replacing_model(self) -> None:
        with (
            patch.object(remote_adapter.shutil, "which", return_value="ollama.exe"),
            patch.object(
                remote_adapter.subprocess,
                "run",
                return_value=SimpleNamespace(
                    stdout="NAME ID SIZE MODIFIED\nexample-ft:latest x 1GB now\n"
                ),
            ) as run,
        ):
            with self.assertRaisesRegex(remote_adapter.ConversionError, "ya existe"):
                self.convert(register_ollama=True)

        self.assertFalse(self.output_dir.exists())
        self.pipeline.export_to_gguf.assert_not_called()
        self.pipeline.register_in_ollama.assert_not_called()
        run.assert_called_once()

    def test_post_export_ollama_error_preserves_and_reports_artifacts(self) -> None:
        self.pipeline.register_in_ollama.side_effect = RuntimeError("ollama failed")
        with (
            patch.object(remote_adapter.shutil, "which", return_value="ollama.exe"),
            patch.object(
                remote_adapter.subprocess,
                "run",
                return_value=SimpleNamespace(stdout="NAME ID SIZE MODIFIED\n"),
            ),
        ):
            with self.assertRaisesRegex(
                remote_adapter.ConversionError,
                re.escape(str(self.output_dir)),
            ):
                self.convert(register_ollama=True)

        self.assertTrue((self.output_dir / "Modelfile").exists())
        self.assertTrue(any(self.output_dir.glob("*.gguf")))

    def test_successful_ollama_registration_is_explicit(self) -> None:
        with (
            patch.object(remote_adapter.shutil, "which", return_value="ollama.exe"),
            patch.object(
                remote_adapter.subprocess,
                "run",
                return_value=SimpleNamespace(stdout="NAME ID SIZE MODIFIED\n"),
            ),
        ):
            result = self.convert(register_ollama=True)

        self.pipeline.register_in_ollama.assert_called_once()
        self.assertEqual(result.ollama_model_name, "example-ft")


if __name__ == "__main__":
    unittest.main()
