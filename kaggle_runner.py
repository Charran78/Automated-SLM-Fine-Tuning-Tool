"""Optional Kaggle CLI integration for remote fine-tuning."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


DatasetKind = Literal["local", "kaggle", "huggingface"]
_KAGGLE_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
_KAGGLE_TRAINING_PACKAGES = (
    "transformers==5.20.2",
    "datasets==4.8.5",
    "trl==1.18.1",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "bitsandbytes==0.50.2",
)


class KaggleError(RuntimeError):
    """Actionable error from Kaggle CLI setup or execution."""


@dataclass(frozen=True)
class KaggleDataset:
    kind: DatasetKind
    value: str
    revision: str | None = None
    upload_consent: bool = False


@dataclass(frozen=True)
class KaggleRun:
    kernel_id: str
    dataset_id: str | None


class KaggleRunner:
    """Calls the optional official Kaggle CLI without storing credentials."""

    def __init__(
        self,
        cli_path: str | None = None,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.cli_path = cli_path
        self._run_command = run_command

    def _executable(self) -> str:
        executable = self.cli_path or shutil.which("kaggle")
        if executable is None:
            raise KaggleError(
                "No se encontró Kaggle CLI. Instálalo opcionalmente con "
                "`python -m pip install kaggle` y reinicia la aplicación."
            )
        return executable

    def _run(self, *args: str, timeout: int = 120) -> str:
        command = [self._executable(), *args]
        try:
            result = self._run_command(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise KaggleError("Kaggle CLI agotó el tiempo de espera.") from error
        except OSError as error:
            raise KaggleError(f"No se pudo ejecutar Kaggle CLI: {error}") from error

        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        if result.returncode:
            details = output or f"el comando terminó con código {result.returncode}"
            if any(
                word in details.lower()
                for word in ("unauthorized", "authentication", "credentials", "token", "401", "403")
            ):
                raise KaggleError(
                    "Kaggle rechazó la autenticación. Configúrala localmente con "
                    "`kaggle auth login` o con el mecanismo oficial de Kaggle; "
                    "no guardes credenciales en el repositorio.\n" + details
                )
            raise KaggleError(f"Falló Kaggle CLI: {details}")
        return output

    @staticmethod
    def _validate_kernel_id(kernel_id: str) -> str:
        kernel_id = kernel_id.strip()
        if not _KAGGLE_REF.fullmatch(kernel_id):
            raise KaggleError("El ID del notebook debe tener formato usuario/slug.")
        return kernel_id

    @staticmethod
    def _prepare_local_dataset(
        dataset: KaggleDataset,
        workspace: Path,
        owner: str,
    ) -> tuple[str, str]:
        if not dataset.upload_consent:
            raise KaggleError(
                "Confirma explícitamente que autorizas subir este archivo a Kaggle como dataset privado."
            )
        source = Path(dataset.value).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in {".json", ".jsonl"}:
            raise KaggleError("El dataset local debe ser un archivo JSON o JSONL existente.")

        digest_builder = hashlib.sha256()
        with source.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()[:10]
        slug = f"slm-ft-{digest}-{uuid.uuid4().hex[:4]}"
        dataset_dir = workspace / "dataset"
        dataset_dir.mkdir()
        shutil.copy2(source, dataset_dir / source.name)
        metadata = {
            "title": f"SLM fine-tuning {slug}",
            "id": f"{owner}/{slug}",
            "description": f"SHA256 del archivo de origen: {digest_builder.hexdigest()}",
            "licenses": [{"name": "other"}],
        }
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata["id"], digest_builder.hexdigest()

    @staticmethod
    def _make_training_script(config: dict[str, object]) -> str:
        template = r'''import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Keep Trainer from seeing multiple GPUs and wrapping the model in DataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# These versions match the successful smoke test; keep Kaggle's image-provided torch.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    __TRAINING_PACKAGES__,
])

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

CONFIG = json.loads(__CONFIG_JSON__)
ARTIFACTS = Path("/kaggle/working/artifacts")
CHECKPOINTS = Path("/kaggle/working/checkpoints")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle no asignó una GPU. Revisa la cuota y la configuración del notebook.")

def find_dataset_files(
    dataset_id: str,
    input_root: Path = Path("/kaggle/input"),
) -> list[Path]:
    expected_root = input_root / dataset_id.rsplit("/", 1)[-1]
    if expected_root.is_dir():
        expected_files = sorted(
            path
            for path in expected_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
        if expected_files:
            return expected_files

    if input_root.is_dir():
        mounted_files = sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
        mounted_paths = sorted(str(path) for path in input_root.iterdir())
        files_by_mount = {}
        for path in mounted_files:
            mount_name = path.relative_to(input_root).parts[0]
            files_by_mount.setdefault(mount_name, []).append(path)
        if len(files_by_mount) == 1:
            return next(iter(files_by_mount.values()))
    else:
        mounted_paths = []
        files_by_mount = {}

    raise FileNotFoundError(
        f"No JSON/JSONL files found for Kaggle dataset '{dataset_id}'. "
        f"Input root: {input_root} (exists: {input_root.is_dir()}). "
        f"Mounted paths: {', '.join(mounted_paths) if mounted_paths else 'none'}. "
        f"Dataset candidates: {', '.join(sorted(files_by_mount)) if files_by_mount else 'none'}. "
        "Attach the dataset to this notebook and verify it contains .json or .jsonl files."
    )

def load_training_data():
    source = CONFIG["dataset"]
    if source["kind"] == "huggingface":
        dataset = load_dataset(
            source["value"], revision=source["revision"], split="train"
        )
        return dataset, None

    dataset_id = source["value"]
    files = find_dataset_files(dataset_id)
    source_files_sha256 = hash_dataset_files(files)
    expected_sha256 = source.get("sha256")
    if expected_sha256 and source_files_sha256 != expected_sha256:
        raise ValueError(
            "El SHA-256 de los archivos Kaggle no coincide con el archivo "
            "local autorizado antes de subirlo."
        )
    dataset = load_dataset(
        "json", data_files=[str(file) for file in files], split="train"
    )
    return dataset, source_files_sha256

def hash_dataset_files(files):
    digest = hashlib.sha256()
    ordered_files = sorted(files)
    for file_path in ordered_files:
        if len(ordered_files) > 1:
            digest.update(file_path.name.encode("utf-8"))
            digest.update(b"\0")
        with file_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

def hash_dataset_rows(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        canonical_row = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest.update(canonical_row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

def resolve_model_revision(model_id, requested_revision):
    from huggingface_hub import HfApi

    requested = requested_revision or "main"
    try:
        resolved = HfApi().model_info(model_id, revision=requested).sha
    except Exception as error:
        raise RuntimeError(
            f"No se pudo resolver el commit de {model_id}@{requested}: {error}"
        ) from error
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved):
        raise RuntimeError(
            f"Hugging Face no devolvió un SHA completo para {model_id}@{requested}."
        )
    return resolved.lower()

model_name = CONFIG["model"]
model_revision = CONFIG.get("model_revision") or None
model_resolved_revision = resolve_model_revision(model_name, model_revision)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, revision=model_resolved_revision
)
if tokenizer.chat_template is None:
    raise ValueError(
        "The selected tokenizer has no chat_template; this remote workflow "
        "currently requires a chat-formatted causal language model."
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset, dataset_files_sha256 = load_training_data()
dataset_content_sha256 = hash_dataset_rows(dataset)
if "messages" in dataset.column_names:
    def format_messages(examples):
        return {"text": [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in examples["messages"]
        ]}
    dataset = dataset.map(format_messages, batched=True)
elif "text" not in dataset.column_names:
    raise ValueError("The dataset must contain a 'messages' or 'text' column.")

dataset_fingerprint = getattr(dataset, "_fingerprint", None)
if len(dataset) > 10:
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
else:
    train_dataset, eval_dataset = dataset, None

def select_compute_dtype(torch_module):
    capability = torch_module.cuda.get_device_capability()
    return (
        torch_module.bfloat16
        if capability >= (8, 0)
        else torch_module.float16
    )

compute_dtype = select_compute_dtype(torch)
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=model_resolved_revision,
    quantization_config=quantization,
    torch_dtype=compute_dtype,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
))

def normalize_fp16_training_tensors(model, torch_module, compute_dtype):
    if compute_dtype != torch_module.float16:
        return
    # T4 uses FP16 only for quantized compute, not autocast/GradScaler training.
    for parameter in model.parameters():
        if parameter.requires_grad:
            target_dtype = torch_module.float32
        elif parameter.dtype == torch_module.bfloat16:
            target_dtype = compute_dtype
        else:
            continue
        if parameter.dtype != target_dtype:
            parameter.data = parameter.data.to(dtype=target_dtype)
    for buffer in model.buffers():
        if buffer.dtype == torch_module.bfloat16:
            buffer.data = buffer.data.to(dtype=compute_dtype)

normalize_fp16_training_tensors(model, torch, compute_dtype)

def create_training_args(training_config, compute_dtype, torch_module):
    # Transformers 5 accepts the same warmup fraction through warmup_steps.
    warmup_parameter = (
        "warmup_ratio"
        if "warmup_ratio" in inspect.signature(SFTConfig).parameters
        else "warmup_steps"
    )
    return SFTConfig(**{
        **training_config,
        "bf16": compute_dtype == torch_module.bfloat16,
        # Avoid GradScaler on pre-Ampere GPUs; QLoRA still computes in FP16.
        "fp16": False,
        warmup_parameter: 0.1,
    })

training_args = create_training_args({
    "output_dir": str(CHECKPOINTS),
    "per_device_train_batch_size": CONFIG["batch_size"],
    "per_device_eval_batch_size": CONFIG["batch_size"],
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "num_train_epochs": CONFIG["epochs"],
    "learning_rate": CONFIG["learning_rate"],
    "lr_scheduler_type": "cosine",
    "logging_steps": 5,
    "save_strategy": "epoch",
    "eval_strategy": "epoch" if eval_dataset is not None else "no",
    "optim": "adamw_8bit",
    "dataset_text_field": "text",
    "max_length": 2048,
    "report_to": "none",
    "seed": 42,
}, compute_dtype, torch)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

start_time = time.time()
baseline_metrics = trainer.evaluate() if eval_dataset is not None else {}
train_result = trainer.train()
final_metrics = trainer.evaluate() if eval_dataset is not None else {}
trainer.save_model(str(ARTIFACTS / "adapter"))
tokenizer.save_pretrained(str(ARTIFACTS / "adapter"))

package_versions = {}
for package in (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
):
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = "unknown"

environment_packages = {}
for distribution in importlib.metadata.distributions():
    package_name = distribution.metadata.get("Name")
    if package_name:
        environment_packages[package_name] = distribution.version

gpu_info = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    gpu_info.append({
        "name": properties.name,
        "memory_bytes": properties.total_memory,
    })

manifest = {
    "model": model_name,
    "model_requested_revision": model_revision,
    "model_resolved_revision": model_resolved_revision,
    "dataset": CONFIG["dataset"],
    "dataset_fingerprint": dataset_fingerprint,
    "dataset_files_sha256": dataset_files_sha256,
    "dataset_content_sha256": dataset_content_sha256,
    "dataset_rows": len(dataset),
    "train_rows": len(train_dataset),
    "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
    "seed": 42,
    "epochs": CONFIG["epochs"],
    "batch_size": CONFIG["batch_size"],
    "learning_rate": CONFIG["learning_rate"],
    "method": "QLoRA 4-bit NF4 + PEFT",
    "baseline_metrics": baseline_metrics,
    "train_metrics": train_result.metrics,
    "final_metrics": final_metrics,
    "elapsed_seconds": round(time.time() - start_time, 2),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "packages": package_versions,
    "environment_packages": environment_packages,
    "gpus": gpu_info,
}
manifest_path = ARTIFACTS / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)

artifact_path = Path("/kaggle/working/model_artifact.zip")
with zipfile.ZipFile(artifact_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for file in (ARTIFACTS / "adapter").rglob("*"):
        if file.is_file():
            archive.write(file, file.relative_to(ARTIFACTS))
    archive.write(manifest_path, manifest_path.name)

shutil.rmtree(CHECKPOINTS, ignore_errors=True)
print(f"SLM_FINE_TUNING_COMPLETE: {artifact_path}")
'''
        config_literal = repr(json.dumps(config, ensure_ascii=False))
        requirements_literal = ",\n    ".join(
            repr(package) for package in _KAGGLE_TRAINING_PACKAGES
        )
        return (
            template.replace("__CONFIG_JSON__", config_literal)
            .replace("__TRAINING_PACKAGES__", requirements_literal)
        )

    def submit(
        self,
        *,
        kernel_id: str,
        model: str,
        model_revision: str | None = None,
        dataset: KaggleDataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> KaggleRun:
        kernel_id = self._validate_kernel_id(kernel_id)
        if not model.strip():
            raise KaggleError("Indica un modelo de Hugging Face.")
        if model_revision and not _HF_REVISION.fullmatch(model_revision.strip()):
            raise KaggleError(
                "La revisión del modelo debe ser un SHA de commit de 40 caracteres hexadecimales."
            )
        if dataset.kind == "kaggle" and not _KAGGLE_REF.fullmatch(dataset.value.strip()):
            raise KaggleError("La referencia Kaggle debe tener formato usuario/slug.")
        if dataset.kind == "huggingface":
            if not dataset.value.strip() or not dataset.revision or not _HF_REVISION.fullmatch(dataset.revision):
                raise KaggleError(
                    "Para Hugging Face indica el ID del dataset y un SHA de revisión de 40 caracteres hexadecimales."
                )
        if dataset.kind not in {"local", "kaggle", "huggingface"}:
            raise KaggleError("Fuente de dataset no reconocida.")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise KaggleError("Los hiperparámetros deben ser mayores que cero.")

        dataset_id: str | None = None
        dataset_source: dict[str, str]
        local_dataset_sha256: str | None = None
        with tempfile.TemporaryDirectory(prefix="slm-kaggle-") as temporary_dir:
            workspace = Path(temporary_dir)
            if dataset.kind == "local":
                dataset_id, local_dataset_sha256 = self._prepare_local_dataset(
                    dataset, workspace, kernel_id.split("/", 1)[0]
                )
                self._run("datasets", "create", "-p", str(workspace / "dataset"), timeout=1800)
                dataset_source = {
                    "kind": "kaggle",
                    "value": dataset_id,
                    "sha256": local_dataset_sha256,
                }
                dataset_sources = [dataset_id]
            elif dataset.kind == "kaggle":
                dataset_id = dataset.value.strip()
                dataset_source = {"kind": "kaggle", "value": dataset_id}
                dataset_sources = [dataset_id]
            else:
                dataset_source = {
                    "kind": "huggingface",
                    "value": dataset.value.strip(),
                    "revision": dataset.revision or "",
                }
                dataset_sources = []

            config = {
                "model": model.strip(),
                "model_revision": model_revision.strip() if model_revision else None,
                "dataset": dataset_source,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
            code_path = workspace / "train.py"
            code_path.write_text(self._make_training_script(config), encoding="utf-8")
            metadata = {
                "id": kernel_id,
                "title": kernel_id.rsplit("/", 1)[-1].replace("-", " ").title(),
                "code_file": "train.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "dataset_sources": dataset_sources,
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (workspace / "kernel-metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._run("kernels", "push", "-p", str(workspace), timeout=1800)
        return KaggleRun(kernel_id=kernel_id, dataset_id=dataset_id)

    def status(self, kernel_id: str) -> str:
        kernel_id = self._validate_kernel_id(kernel_id)
        return self._run("kernels", "status", kernel_id, timeout=120)

    def download_output(self, kernel_id: str, output_dir: str | os.PathLike[str]) -> Path:
        kernel_id = self._validate_kernel_id(kernel_id)
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._run("kernels", "output", kernel_id, "-p", str(destination), "-o", timeout=1800)
        return destination
