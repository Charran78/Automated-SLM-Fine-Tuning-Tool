import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kaggle_runner import KaggleDataset, KaggleError, KaggleRunner


class KaggleRunnerTests(unittest.TestCase):
    @staticmethod
    def _generated_compute_dtype_selector():
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        selector = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "select_compute_dtype"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[selector], type_ignores=[]),
                "<generated-compute-dtype-selector>",
                "exec",
            ),
            namespace,
        )
        return namespace["select_compute_dtype"]

    @staticmethod
    def _generated_find_dataset_files():
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset-slug"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        finder = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "find_dataset_files"
        )
        namespace = {"Path": Path}
        exec(
            compile(ast.Module(body=[finder], type_ignores=[]), "<dataset-file-finder>", "exec"),
            namespace,
        )
        return namespace["find_dataset_files"]

    def test_missing_cli_has_an_actionable_error(self):
        with patch("kaggle_runner.shutil.which", return_value=None):
            with self.assertRaisesRegex(KaggleError, "No se encontró Kaggle CLI"):
                KaggleRunner().status("owner/notebook")

    def test_generated_finder_loads_files_from_expected_mount_path(self):
        find_dataset_files = self._generated_find_dataset_files()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir)
            data_file = input_root / "dataset-slug" / "nested" / "train.jsonl"
            data_file.parent.mkdir(parents=True)
            data_file.write_text('{"text":"example"}\n', encoding="utf-8")

            self.assertEqual(
                find_dataset_files("owner/dataset-slug", input_root),
                [data_file],
            )

    def test_generated_finder_falls_back_to_differently_named_mount(self):
        find_dataset_files = self._generated_find_dataset_files()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir)
            (input_root / "dataset-slug").mkdir()
            data_file = input_root / "actual-mounted-name" / "nested" / "train.jsonl"
            data_file.parent.mkdir(parents=True)
            data_file.write_text('{"text":"example"}\n', encoding="utf-8")

            self.assertEqual(
                find_dataset_files("owner/dataset-slug", input_root),
                [data_file],
            )

    def test_generated_finder_rejects_ambiguous_dataset_mounts(self):
        find_dataset_files = self._generated_find_dataset_files()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir)
            files = [
                input_root / mount / "train.jsonl"
                for mount in ("other-dataset-a", "other-dataset-b")
            ]
            for data_file in files:
                data_file.parent.mkdir(parents=True)
                data_file.write_text('{"text":"example"}\n', encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as context:
                find_dataset_files("owner/dataset-slug", input_root)

            self.assertIn("Dataset candidates: other-dataset-a, other-dataset-b", str(context.exception))

    def test_generated_finder_reports_root_and_mounts_when_no_dataset_is_mounted(self):
        find_dataset_files = self._generated_find_dataset_files()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            input_root.mkdir()

            with self.assertRaises(FileNotFoundError) as context:
                find_dataset_files("owner/dataset-slug", input_root)

            message = str(context.exception)
            self.assertIn(str(input_root), message)
            self.assertIn("Mounted paths: none", message)
            self.assertIn("owner/dataset-slug", message)
            self.assertIn("Attach the dataset", message)

    def test_generated_compute_dtype_uses_fp16_on_t4_and_bf16_on_supported_gpus(self):
        select_compute_dtype = self._generated_compute_dtype_selector()
        torch_module = SimpleNamespace(
            bfloat16="bf16",
            float16="fp16",
            cuda=SimpleNamespace(get_device_capability=lambda: (7, 5)),
        )

        self.assertEqual(select_compute_dtype(torch_module), "fp16")

        for capability in ((8, 0), (8, 6), (9, 0)):
            torch_module.cuda.get_device_capability = lambda capability=capability: capability
            with self.subTest(capability=capability):
                self.assertEqual(select_compute_dtype(torch_module), "bf16")

    def test_auth_and_cli_failures_are_reported_clearly(self):
        auth_run = Mock(
            return_value=subprocess.CompletedProcess(
                ["kaggle"], 1, "", "401 Unauthorized"
            )
        )
        with self.assertRaisesRegex(KaggleError, "kaggle auth login"):
            KaggleRunner(cli_path="kaggle", run_command=auth_run).status("owner/run")

        failed_run = Mock(
            return_value=subprocess.CompletedProcess(
                ["kaggle"], 1, "", "Kernel not found"
            )
        )
        with self.assertRaisesRegex(KaggleError, "Kernel not found"):
            KaggleRunner(cli_path="kaggle", run_command=failed_run).status("owner/run")

    def test_status_and_output_use_official_cli_commands(self):
        run_command = Mock(
            side_effect=[
                subprocess.CompletedProcess(["kaggle"], 0, "KernelWorkerStatus.RUNNING", ""),
                subprocess.CompletedProcess(["kaggle"], 0, "Downloaded", ""),
            ]
        )
        runner = KaggleRunner(cli_path="kaggle", run_command=run_command)
        self.assertEqual(runner.status("owner/run"), "KernelWorkerStatus.RUNNING")
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = runner.download_output("owner/run", Path(temp_dir) / "artifacts")
            self.assertTrue(destination.is_dir())
        self.assertEqual(run_command.call_args_list[0].args[0], ["kaggle", "kernels", "status", "owner/run"])
        self.assertEqual(run_command.call_args_list[1].args[0][1:4], [
            "kernels", "output", "owner/run"
        ])
        self.assertIn("-o", run_command.call_args_list[1].args[0])

    def test_local_upload_requires_consent_and_stays_private(self):
        run_command = Mock(
            side_effect=lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "ok", "")
        )
        runner = KaggleRunner(cli_path="kaggle", run_command=run_command)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "private.jsonl"
            data_file.write_text('{"messages":[]}\n', encoding="utf-8")
            expected_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()

            with self.assertRaisesRegex(KaggleError, "Confirma explícitamente"):
                runner.submit(
                    kernel_id="owner/private-run",
                    model="org/model",
                    dataset=KaggleDataset("local", str(data_file)),
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-4,
                )
            self.assertEqual(run_command.call_count, 0)

            observed: dict[str, object] = {}

            def inspect_command(command, **_kwargs):
                if command[1:3] == ["datasets", "create"]:
                    dataset_metadata = json.loads(
                        (Path(command[4]) / "dataset-metadata.json").read_text(encoding="utf-8")
                    )
                    observed["dataset_metadata"] = dataset_metadata
                    observed["dataset_command"] = command
                if command[1:3] == ["kernels", "push"]:
                    kernel_dir = Path(command[4])
                    metadata = json.loads(
                        (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
                    )
                    observed["kernel_metadata"] = metadata
                    observed["script"] = (kernel_dir / "train.py").read_text(encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "ok", "")

            run_command.side_effect = inspect_command
            result = runner.submit(
                kernel_id="owner/private-run",
                model="org/model",
                dataset=KaggleDataset(
                    "local", str(data_file), upload_consent=True
                ),
                epochs=2,
                batch_size=1,
                learning_rate=1e-4,
            )
            self.assertEqual(result.kernel_id, "owner/private-run")
            self.assertRegex(
                result.dataset_id or "",
                r"^owner/slm-ft-[0-9a-f]{10}-[0-9a-f]{4}$",
            )

        self.assertNotIn("--public", observed["dataset_command"])
        self.assertEqual(observed["dataset_metadata"]["licenses"], [{"name": "other"}])
        kernel_metadata = observed["kernel_metadata"]
        self.assertEqual(kernel_metadata["is_private"], "true")
        self.assertEqual(kernel_metadata["enable_internet"], "true")
        self.assertEqual(kernel_metadata["dataset_sources"], [result.dataset_id])
        self.assertIn("CONFIG", observed["script"])
        self.assertIn(expected_sha256, observed["script"])

    def test_kaggle_reference_attaches_dataset_without_uploading_files(self):
        observed: dict[str, object] = {}

        def inspect_command(command, **_kwargs):
            if command[1:3] == ["kernels", "push"]:
                metadata_path = Path(command[4]) / "kernel-metadata.json"
                observed["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
            return subprocess.CompletedProcess(command, 0, "ok", "")

        runner = KaggleRunner(cli_path="kaggle", run_command=inspect_command)
        result = runner.submit(
            kernel_id="owner/run",
            model="org/model",
            dataset=KaggleDataset("kaggle", "data-owner/dataset-name"),
            epochs=1,
            batch_size=1,
            learning_rate=2e-4,
        )
        self.assertEqual(result.dataset_id, "data-owner/dataset-name")
        self.assertEqual(observed["metadata"]["dataset_sources"], ["data-owner/dataset-name"])
        self.assertEqual(len(observed["metadata"]["dataset_sources"]), 1)

    def test_huggingface_dataset_requires_and_embeds_a_pinned_revision(self):
        observed: dict[str, str] = {}
        revision = "a" * 40

        def inspect_command(command, **_kwargs):
            if command[1:3] == ["kernels", "push"]:
                script = (Path(command[4]) / "train.py").read_text(encoding="utf-8")
                observed["script"] = script
                observed["metadata"] = json.loads(
                    (Path(command[4]) / "kernel-metadata.json").read_text(encoding="utf-8")
                )
            return subprocess.CompletedProcess(command, 0, "ok", "")

        runner = KaggleRunner(cli_path="kaggle", run_command=inspect_command)
        with self.assertRaisesRegex(KaggleError, "SHA de revisión"):
            runner.submit(
                kernel_id="owner/run",
                model="org/model",
                dataset=KaggleDataset("huggingface", "org/dataset", "main"),
                epochs=1,
                batch_size=1,
                learning_rate=2e-4,
            )
        runner.submit(
            kernel_id="owner/run",
            model="org/model",
            dataset=KaggleDataset("huggingface", "org/dataset", revision),
            epochs=1,
            batch_size=1,
            learning_rate=2e-4,
        )
        self.assertIn(revision, observed["script"])
        self.assertEqual(observed["metadata"]["dataset_sources"], [])
        self.assertEqual(observed["metadata"]["enable_internet"], "true")

    def test_generated_script_compiles_and_uses_measured_4bit_qlora(self):
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": "b" * 40,
            "dataset": {
                "kind": "huggingface",
                "value": "org/dataset",
                "revision": "a" * 40,
            },
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })

        compile(script, "<kaggle-training-script>", "exec")
        script_lines = script.splitlines()
        gpu_setup_line = next(
            index
            for index, line in enumerate(script_lines)
            if line.strip() == 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"'
        )
        torch_import_line = next(
            index
            for index, line in enumerate(script_lines)
            if line.strip() == "import torch"
        )
        self.assertLess(gpu_setup_line, torch_import_line)
        self.assertIn("find_dataset_files", script)
        self.assertIn("load_in_4bit=True", script)
        self.assertIn("prepare_model_for_kbit_training", script)
        self.assertIn("baseline_metrics", script)
        self.assertIn("final_metrics", script)
        self.assertIn("manifest.json", script)
        self.assertIn("model_resolved_revision", script)
        self.assertIn("dataset_content_sha256", script)
        self.assertIn("environment_packages", script)
        for requirement in (
            "transformers==5.20.2",
            "datasets==4.8.5",
            "trl==1.18.1",
            "peft==0.21.2",
            "accelerate==1.15.0",
            "bitsandbytes==0.50.2",
        ):
            self.assertIn(requirement, script)

    def test_generated_model_revision_resolver_returns_a_full_hub_sha(self):
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        resolver = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "resolve_model_revision"
        )
        namespace = {"re": re}
        api = Mock(
            return_value=SimpleNamespace(
                model_info=Mock(return_value=SimpleNamespace(sha="a" * 40))
            )
        )
        with patch.dict(
            sys.modules, {"huggingface_hub": SimpleNamespace(HfApi=api)}
        ):
            exec(
                compile(
                    ast.Module(body=[resolver], type_ignores=[]),
                    "<model-revision-resolver>",
                    "exec",
                ),
                namespace,
            )
            resolved = namespace["resolve_model_revision"]("org/model", None)

        self.assertEqual(resolved, "a" * 40)
        api.return_value.model_info.assert_called_once_with(
            "org/model", revision="main"
        )

    def test_generated_local_file_hash_matches_upload_sha256(self):
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        helper = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "hash_dataset_files"
        )
        namespace = {"hashlib": hashlib}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                "<dataset-file-hash>",
                "exec",
            ),
            namespace,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "train.jsonl"
            data_file.write_bytes(b'{"text":"hello"}\n')
            expected = hashlib.sha256(data_file.read_bytes()).hexdigest()
            actual = namespace["hash_dataset_files"]([data_file])

        self.assertEqual(actual, expected)

    def test_generated_training_args_support_transformers_4_and_5_warmup_apis(self):
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        helper = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_training_args"
        )

        def legacy_sft_config(*, warmup_ratio=0.0, **kwargs):
            return {"warmup_ratio": warmup_ratio, **kwargs}

        def current_sft_config(*, warmup_steps=0, **kwargs):
            return {"warmup_steps": warmup_steps, **kwargs}

        for sft_config, warmup_parameter in (
            (legacy_sft_config, "warmup_ratio"),
            (current_sft_config, "warmup_steps"),
        ):
            namespace = {
                "inspect": inspect,
                "SFTConfig": sft_config,
                "torch_module": SimpleNamespace(bfloat16="bf16", float16="fp16"),
            }
            helper_module = ast.Module(body=[helper], type_ignores=[])
            exec(
                compile(helper_module, "<generated-training-args>", "exec"),
                namespace,
            )
            training_args = namespace["create_training_args"](
                {"seed": 42}, "fp16", namespace["torch_module"]
            )
            self.assertEqual(training_args[warmup_parameter], 0.1)
            self.assertEqual(training_args["seed"], 42)

    def test_generated_training_args_disable_scaler_on_t4_and_keep_bf16_path(self):
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        helper = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_training_args"
        )
        torch_module = SimpleNamespace(bfloat16="bf16", float16="fp16")
        namespace = {
            "inspect": inspect,
            "SFTConfig": lambda **kwargs: kwargs,
            "torch_module": torch_module,
        }
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), "<generated-t4-args>", "exec"),
            namespace,
        )

        training_args = namespace["create_training_args"](
            {"seed": 42}, torch_module.float16, torch_module
        )

        self.assertFalse(training_args["fp16"])
        self.assertFalse(training_args["bf16"])

        bf16_training_args = namespace["create_training_args"](
            {"seed": 42}, torch_module.bfloat16, torch_module
        )
        self.assertFalse(bf16_training_args["fp16"])
        self.assertTrue(bf16_training_args["bf16"])

    def test_generated_script_converts_bfloat16_model_tensors_before_training(self):
        script = KaggleRunner._make_training_script({
            "model": "org/model",
            "model_revision": None,
            "dataset": {"kind": "kaggle", "value": "owner/dataset"},
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        })
        script_ast = ast.parse(script)
        normalizer = next(
            node
            for node in script_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "normalize_fp16_training_tensors"
        )

        class FakeTensor:
            def __init__(self, dtype, requires_grad=False):
                self.dtype = dtype
                self.requires_grad = requires_grad

            @property
            def data(self):
                return self

            @data.setter
            def data(self, value):
                self.dtype = value.dtype

            def to(self, *, dtype):
                return FakeTensor(dtype)

        class FakeModel:
            def __init__(self):
                self.parameters_to_check = [
                    FakeTensor("bf16", requires_grad=True),
                    FakeTensor("fp16", requires_grad=True),
                    FakeTensor("bf16"),
                    FakeTensor("fp32"),
                ]
                self.buffers_to_check = [FakeTensor("bf16"), FakeTensor("uint8")]

            def parameters(self):
                return iter(self.parameters_to_check)

            def buffers(self):
                return iter(self.buffers_to_check)

        namespace = {}
        exec(
            compile(
                ast.Module(body=[normalizer], type_ignores=[]),
                "<generated-tensor-normalizer>",
                "exec",
            ),
            namespace,
        )
        model = FakeModel()
        namespace["normalize_fp16_training_tensors"](
            model,
            SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32"),
            "fp16",
        )

        self.assertEqual(
            [tensor.dtype for tensor in model.parameters_to_check],
            ["fp32", "fp32", "fp16", "fp32"],
        )
        self.assertEqual(
            [tensor.dtype for tensor in model.buffers_to_check],
            ["fp16", "uint8"],
        )
        self.assertFalse(
            any(
                tensor.dtype == "bf16"
                for tensor in (*model.parameters_to_check, *model.buffers_to_check)
            )
        )

        bf16_model = FakeModel()
        namespace["normalize_fp16_training_tensors"](
            bf16_model,
            SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32"),
            "bf16",
        )
        self.assertEqual(
            [tensor.dtype for tensor in bf16_model.parameters_to_check],
            ["bf16", "fp16", "bf16", "fp32"],
        )

    def test_model_revision_must_be_a_full_commit_sha(self):
        run_command = Mock()
        with self.assertRaisesRegex(KaggleError, "revisión del modelo"):
            KaggleRunner(cli_path="kaggle", run_command=run_command).submit(
                kernel_id="owner/run",
                model="org/model",
                model_revision="main",
                dataset=KaggleDataset("kaggle", "data-owner/dataset-name"),
                epochs=1,
                batch_size=1,
                learning_rate=2e-4,
            )
        self.assertEqual(run_command.call_count, 0)


if __name__ == "__main__":
    unittest.main()
