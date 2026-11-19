from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, NoReturn


MAX_ARCHIVE_BYTES = 5 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 20 * 1024**3
MAX_MEMBER_BYTES = 10 * 1024**3
MAX_ARCHIVE_MEMBERS = 20_000
MAX_MEMBER_COMPRESSION_RATIO = 1_000
MAX_METADATA_BYTES = 4 * 1024**2

_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
_OLLAMA_NAME = re.compile(r"^[a-z0-9][a-z0-9._/-]*(?::[a-zA-Z0-9._-]+)?$")
_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
_TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)


class RemoteArtifactError(ValueError):
    """Artifact ZIP no válido o solicitud de importación insegura."""


class ConversionError(RuntimeError):
    """Error durante la conversión; indica dónde quedaron los archivos."""


@dataclass(frozen=True)
class RemoteArtifactMetadata:
    model_id: str
    recorded_revision: str | None
    adapter_revision: str | None
    archive_sha256: str
    adapter_root: str


@dataclass(frozen=True)
class ConversionResult:
    model_id: str
    model_revision: str
    archive_sha256: str
    output_dir: Path
    gguf_files: tuple[Path, ...]
    modelfile_path: Path
    ollama_model_name: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_member_name(name: str) -> tuple[str, bool]:
    if not name or "\\" in name or "\x00" in name:
        raise RemoteArtifactError(f"Nombre de ruta ZIP no válido: {name!r}.")
    if name.startswith("/") or re.match(r"^[a-zA-Z]:", name):
        raise RemoteArtifactError(f"Ruta absoluta no permitida: {name!r}.")
    is_directory = name.endswith("/")
    raw_parts = name.split("/")
    if is_directory:
        raw_parts.pop()
    if any(_is_unsafe_path_part(part) for part in raw_parts):
        raise RemoteArtifactError(f"Ruta insegura o traversal en ZIP: {name!r}.")
    normalized = PurePosixPath(*raw_parts).as_posix()
    if (
        normalized != "manifest.json"
        and normalized != "adapter"
        and not normalized.startswith("adapter/")
    ):
        raise RemoteArtifactError(f"Ruta fuera de manifest.json y adapter/: {name!r}.")
    return normalized, is_directory


def _is_unsafe_path_part(part: str) -> bool:
    reserved_windows_name = re.match(
        r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", part, re.I
    )
    return (
        part in {"", ".", ".."}
        or ":" in part
        or part.endswith((" ", "."))
        or reserved_windows_name is not None
    )


def _validate_member_type(member: zipfile.ZipInfo, is_directory: bool) -> None:
    unix_mode = member.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise RemoteArtifactError(
            f"El ZIP contiene un enlace simbólico: {member.filename!r}."
        )
    file_type = stat.S_IFMT(unix_mode)
    if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
        raise RemoteArtifactError(
            f"Tipo de archivo no permitido en ZIP: {member.filename!r}."
        )
    if is_directory != member.is_dir():
        raise RemoteArtifactError(
            f"Tipo de miembro inconsistente en ZIP: {member.filename!r}."
        )


def _validate_zip_member(
    member: zipfile.ZipInfo,
    normalized_names: set[str],
) -> str:
    normalized, is_directory = _validate_member_name(member.filename)
    if normalized.casefold() in normalized_names:
        raise RemoteArtifactError(
            f"El ZIP contiene miembros duplicados: {member.filename!r}."
        )
    normalized_names.add(normalized.casefold())
    _validate_member_type(member, is_directory)
    return normalized


def _validate_zip_members(
    members: list[zipfile.ZipInfo],
) -> dict[str, zipfile.ZipInfo]:
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise RemoteArtifactError("El ZIP está vacío o supera el límite de miembros.")
    member_by_name: dict[str, zipfile.ZipInfo] = {}
    normalized_names: set[str] = set()
    total_size = 0
    for member in members:
        normalized = _validate_zip_member(member, normalized_names)
        if member.file_size > MAX_MEMBER_BYTES:
            raise RemoteArtifactError("El ZIP supera el límite de tamaño por archivo.")
        if (
            not member.is_dir()
            and member.file_size
            and (
                member.compress_size == 0
                or member.file_size / member.compress_size
                > MAX_MEMBER_COMPRESSION_RATIO
            )
        ):
            raise RemoteArtifactError(
                f"El ZIP contiene un miembro con ratio de compresión excesivo: "
                f"{member.filename!r}."
            )
        total_size += member.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise RemoteArtifactError("El ZIP supera el límite descomprimido.")
        member_by_name[normalized] = member
    return member_by_name


def _validate_archive_crc(archive: zipfile.ZipFile) -> None:
    try:
        corrupt_member = archive.testzip()
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise RemoteArtifactError(
            f"No se pudo validar el CRC del ZIP: {error}"
        ) from error
    if corrupt_member is not None:
        raise RemoteArtifactError(f"CRC inválido en el miembro ZIP {corrupt_member!r}.")


def _validate_zip(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    member_by_name = _validate_zip_members(archive.infolist())
    _validate_archive_crc(archive)
    return member_by_name


def _read_json_member(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    name: str,
) -> dict[str, Any]:
    member = members.get(name)
    if member is None or member.is_dir():
        raise RemoteArtifactError(f"Falta el archivo requerido {name!r}.")
    if member.file_size > MAX_METADATA_BYTES:
        raise RemoteArtifactError(
            f"Los metadatos {name!r} superan el tamaño permitido."
        )
    try:
        value = json.loads(archive.read(member).decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
        zipfile.BadZipFile,
    ) as error:
        raise RemoteArtifactError(f"JSON inválido en {name!r}: {error}") from error
    if not isinstance(value, dict):
        raise RemoteArtifactError(f"{name!r} debe contener un objeto JSON.")
    return value


def _recorded_revision(manifest: dict[str, Any]) -> str | None:
    resolved = manifest.get("model_resolved_revision")
    if resolved is None or resolved == "":
        return None
    if not isinstance(resolved, str) or not _SHA40.fullmatch(resolved):
        raise RemoteArtifactError(
            "La revisión resuelta del manifiesto debe ser un SHA de 40 caracteres."
        )
    return resolved.lower()


def _validate_model_ids(
    manifest: dict[str, Any],
    adapter_config: dict[str, Any],
) -> str:
    model_id = manifest.get("model")
    adapter_model_id = adapter_config.get("base_model_name_or_path")
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or not isinstance(adapter_model_id, str)
        or adapter_model_id.strip() != model_id.strip()
    ):
        raise RemoteArtifactError(
            "Los model ID del manifiesto y adapter_config.json no coinciden."
        )
    return model_id.strip()


def _validate_adapter_structure(
    members: dict[str, zipfile.ZipInfo],
) -> None:
    adapter_files = {
        name.removeprefix("adapter/").rsplit("/", 1)[-1]: member
        for name, member in members.items()
        if name.startswith("adapter/") and not member.is_dir()
    }
    has_weights = any(
        member.file_size > 0
        and (
            filename in _WEIGHT_NAMES
            or re.fullmatch(r"adapter_model-\d+-of-\d+\.(?:safetensors|bin)", filename)
        )
        for filename, member in adapter_files.items()
    )
    if not has_weights:
        raise RemoteArtifactError("Falta el archivo de pesos del adaptador PEFT.")
    if not any(
        filename in _TOKENIZER_NAMES and member.file_size > 0
        for filename, member in adapter_files.items()
    ):
        raise RemoteArtifactError("Faltan los archivos del tokenizer del adaptador.")


def _inspect_open_archive(
    archive: zipfile.ZipFile,
    archive_hash: str,
) -> RemoteArtifactMetadata:
    members = _validate_zip(archive)
    manifest = _read_json_member(archive, members, "manifest.json")
    config = _read_json_member(archive, members, "adapter/adapter_config.json")
    model_id = _validate_model_ids(manifest, config)
    _validate_adapter_structure(members)
    adapter_revision = config.get("revision")
    if adapter_revision is not None and not isinstance(adapter_revision, str):
        raise RemoteArtifactError(
            "La revisión del adaptador debe ser una cadena o null."
        )
    recorded_revision = _recorded_revision(manifest)
    if (
        recorded_revision
        and isinstance(adapter_revision, str)
        and _SHA40.fullmatch(adapter_revision)
        and adapter_revision.lower() != recorded_revision
    ):
        raise RemoteArtifactError(
            "La revisión SHA del adaptador no coincide con el manifiesto."
        )
    return RemoteArtifactMetadata(
        model_id=model_id,
        recorded_revision=recorded_revision,
        adapter_revision=adapter_revision,
        archive_sha256=archive_hash,
        adapter_root="adapter",
    )


def inspect_remote_artifact(path: str | Path) -> RemoteArtifactMetadata:
    """Valida sin extraer un ZIP PEFT y devuelve los metadatos de procedencia."""
    archive_path = Path(path)
    try:
        if not archive_path.is_file():
            raise RemoteArtifactError("La ruta del artefacto debe ser un archivo ZIP.")
        archive_size = archive_path.stat().st_size
        if archive_size > MAX_ARCHIVE_BYTES:
            raise RemoteArtifactError("El ZIP supera el límite de tamaño comprimido.")
        archive_hash = _sha256_file(archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            return _inspect_open_archive(archive, archive_hash)
    except RemoteArtifactError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise RemoteArtifactError(f"No se pudo leer el ZIP: {error}") from error


def resolve_current_revision(model_id: str) -> str:
    """Resuelve el commit actual de Hugging Face a un SHA completo."""
    if not model_id.strip():
        raise ValueError("Indica un model ID de Hugging Face.")
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().model_info(model_id.strip()).sha
    except Exception as error:
        raise RemoteArtifactError(
            f"No se pudo resolver la revisión actual de {model_id!r}: {error}"
        ) from error
    if not isinstance(resolved, str) or not _SHA40.fullmatch(resolved):
        raise RemoteArtifactError(
            f"Hugging Face no devolvió un SHA de 40 caracteres para {model_id!r}."
        )
    return resolved.lower()


def _load_pipeline_class() -> type[Any]:
    from slm_automator import SLMFineTunerPipeline

    return SLMFineTunerPipeline


def _load_conversion_dependencies() -> SimpleNamespace:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return SimpleNamespace(
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        PeftModel=PeftModel,
    )


def _extract_validated_archive(
    archive_path: Path,
    target_dir: Path,
    expected_hash: str,
) -> Path:
    if _sha256_file(archive_path) != expected_hash:
        raise RemoteArtifactError("El ZIP cambió desde la inspección inicial.")
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = _validate_zip(archive)
        for name, member in members.items():
            target = target_dir.joinpath(*PurePosixPath(name).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if _sha256_file(archive_path) != expected_hash:
        raise RemoteArtifactError("El ZIP cambió durante la extracción temporal.")
    return target_dir / "adapter"


def _resolve_import_revision(
    recorded_revision: str | None,
    supplied_revision: str | None,
    acknowledged: bool,
) -> str:
    if recorded_revision:
        if supplied_revision and supplied_revision.lower() != recorded_revision:
            raise RemoteArtifactError(
                "No se puede sustituir la revisión registrada en el manifiesto."
            )
        return recorded_revision
    if not supplied_revision or not _SHA40.fullmatch(supplied_revision):
        raise RemoteArtifactError(
            "Falta revisión registrada: indica un SHA explícito de 40 caracteres."
        )
    if not acknowledged:
        raise RemoteArtifactError(
            "Debes aceptar explícitamente el uso de una revisión no registrada."
        )
    return supplied_revision.lower()


def _ollama_existing_names() -> set[str]:
    result = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        columns = line.split()
        if columns:
            names.add(columns[0].lower())
    return names


def _validate_ollama_name(name: str) -> None:
    if not _OLLAMA_NAME.fullmatch(name):
        raise RemoteArtifactError(
            "El nombre Ollama debe usar formato model[:tag] sin espacios."
        )


def _prepare_ollama(name: str) -> None:
    _validate_ollama_name(name)
    if shutil.which("ollama") is None:
        raise ConversionError("No se encontró el binario ollama en PATH.")
    try:
        existing_names = _ollama_existing_names()
    except (OSError, subprocess.SubprocessError) as error:
        raise ConversionError(f"No se pudo consultar Ollama: {error}") from error
    normalized = name.lower()
    default_tag_name = normalized if ":" in normalized else f"{normalized}:latest"
    if normalized in existing_names or default_tag_name in existing_names:
        raise ConversionError(f"El modelo Ollama {name!r} ya existe.")


def _load_import_pipeline(
    metadata: RemoteArtifactMetadata,
    revision: str,
    adapter_root: Path,
    destination: Path,
) -> Any:
    pipeline = _load_pipeline_class()(
        model_name=metadata.model_id,
        output_dir=str(destination),
    )
    pipeline.use_unsloth = False
    dependencies = _load_conversion_dependencies()
    base_model = dependencies.AutoModelForCausalLM.from_pretrained(
        metadata.model_id,
        revision=revision,
        torch_dtype=dependencies.torch.float32,
        device_map="cpu",
    )
    pipeline.model = dependencies.PeftModel.from_pretrained(
        base_model, str(adapter_root)
    )
    pipeline.tokenizer = dependencies.AutoTokenizer.from_pretrained(str(adapter_root))
    return pipeline


def _export_artifacts(
    pipeline: Any,
    destination: Path,
    quant_type: str,
    source_path: Path,
    expected_hash: str,
) -> tuple[Path, tuple[Path, ...]]:
    modelfile = Path(
        pipeline.export_to_gguf(
            export_dir=str(destination),
            quant_type=quant_type,
        )
    )
    gguf_files = tuple(sorted(destination.rglob("*.gguf")))
    if not modelfile.is_file() or not gguf_files:
        raise RuntimeError("La exportación no produjo un Modelfile y al menos un GGUF.")
    _ensure_archive_unchanged(source_path, expected_hash)
    return modelfile, gguf_files


def _ensure_archive_unchanged(source_path: Path, expected_hash: str) -> None:
    if _sha256_file(source_path) != expected_hash:
        raise RuntimeError("El ZIP de origen cambió durante la conversión.")


def _write_conversion_manifest(
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    quant_type: str,
    modelfile: Path,
    gguf_files: tuple[Path, ...],
) -> None:
    manifest_data = {
        "model_id": metadata.model_id,
        "model_revision": revision,
        "recorded_revision": metadata.recorded_revision,
        "adapter_revision": metadata.adapter_revision,
        "archive_sha256": metadata.archive_sha256,
        "quant_type": quant_type,
        "gguf_files": [str(path) for path in gguf_files],
        "modelfile_path": str(modelfile),
    }
    (destination / "conversion_manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _complete_conversion(
    pipeline: Any,
    source_path: Path,
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    quant_type: str,
    modelfile: Path,
    gguf_files: tuple[Path, ...],
    ollama_model_name: str,
    register_ollama: bool,
) -> ConversionResult:
    _write_conversion_manifest(
        destination, metadata, revision, quant_type, modelfile, gguf_files
    )
    _register_ollama_if_requested(
        pipeline, modelfile, ollama_model_name, register_ollama
    )
    return _finalize_conversion_result(
        source_path,
        destination,
        metadata,
        revision,
        gguf_files,
        modelfile,
        ollama_model_name,
        register_ollama,
    )


def _finalize_conversion_result(
    source_path: Path,
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    gguf_files: tuple[Path, ...],
    modelfile: Path,
    ollama_model_name: str,
    register_ollama: bool,
) -> ConversionResult:
    _ensure_archive_unchanged(source_path, metadata.archive_sha256)
    return _build_conversion_result(
        destination,
        metadata,
        revision,
        gguf_files,
        modelfile,
        ollama_model_name,
        register_ollama,
    )


def _register_ollama_if_requested(
    pipeline: Any,
    modelfile: Path,
    ollama_model_name: str,
    register_ollama: bool,
) -> None:
    if register_ollama:
        _prepare_ollama(ollama_model_name)
        pipeline.register_in_ollama(
            str(modelfile),
            ollama_model_name=ollama_model_name,
        )


def _build_conversion_result(
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    gguf_files: tuple[Path, ...],
    modelfile: Path,
    ollama_model_name: str,
    register_ollama: bool,
) -> ConversionResult:
    return ConversionResult(
        model_id=metadata.model_id,
        model_revision=revision,
        archive_sha256=metadata.archive_sha256,
        output_dir=destination,
        gguf_files=gguf_files,
        modelfile_path=modelfile,
        ollama_model_name=ollama_model_name if register_ollama else None,
    )


def _export_from_archive(
    source_path: Path,
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    quant_type: str,
) -> tuple[Any, Path, tuple[Path, ...]]:
    with tempfile.TemporaryDirectory(prefix="remote-adapter-") as temporary_dir:
        adapter_root = _extract_validated_archive(
            source_path,
            Path(temporary_dir),
            metadata.archive_sha256,
        )
        pipeline = _load_import_pipeline(metadata, revision, adapter_root, destination)
        modelfile, gguf_files = _export_artifacts(
            pipeline,
            destination,
            quant_type,
            source_path,
            metadata.archive_sha256,
        )
    return pipeline, modelfile, gguf_files


def _perform_conversion(
    source_path: Path,
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    quant_type: str,
    ollama_model_name: str,
    register_ollama: bool,
) -> ConversionResult:
    pipeline, modelfile, gguf_files = _export_from_archive(
        source_path, destination, metadata, revision, quant_type
    )
    return _complete_conversion(
        pipeline,
        source_path,
        destination,
        metadata,
        revision,
        quant_type,
        modelfile,
        gguf_files,
        ollama_model_name,
        register_ollama,
    )


def _prepare_conversion(
    archive_path: str | Path,
    output_dir: str | Path,
    model_revision: str | None,
    acknowledge_unrecorded_revision: bool,
    quant_type: str,
    ollama_model_name: str,
    register_ollama: bool = True,
) -> tuple[Path, Path, RemoteArtifactMetadata, str]:
    source_path = Path(archive_path)
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise ConversionError(f"El directorio de salida ya existe: {destination}")
    if not quant_type.strip():
        raise ValueError("Indica un tipo de cuantización.")

    metadata = inspect_remote_artifact(source_path)
    revision = _resolve_import_revision(
        metadata.recorded_revision,
        model_revision,
        acknowledge_unrecorded_revision,
    )
    if register_ollama:
        _prepare_ollama(ollama_model_name)
    return source_path, destination, metadata, revision


def _raise_conversion_error(error: Exception, destination: Path) -> NoReturn:
    if destination.exists():
        raise ConversionError(
            f"No se pudo completar la conversión: {error}. "
            f"La salida parcial se conserva en: {destination}"
        ) from error
    if isinstance(error, (RemoteArtifactError, ConversionError)):
        raise error
    raise ConversionError(f"No se pudo completar la conversión: {error}") from error


def convert_remote_artifact(
    archive_path: str | Path,
    output_dir: str | Path,
    model_revision: str | None,
    acknowledge_unrecorded_revision: bool,
    quant_type: str,
    ollama_model_name: str,
    register_ollama: bool = True,
) -> ConversionResult:
    """Importa un adaptador sin entrenar ni modificar el pipeline global."""
    prepared = _prepare_conversion(
        archive_path,
        output_dir,
        model_revision,
        acknowledge_unrecorded_revision,
        quant_type,
        ollama_model_name,
        register_ollama,
    )
    return _convert_with_error_reporting(
        *prepared,
        quant_type,
        ollama_model_name,
        register_ollama,
    )


def _convert_with_error_reporting(
    source_path: Path,
    destination: Path,
    metadata: RemoteArtifactMetadata,
    revision: str,
    quant_type: str,
    ollama_model_name: str,
    register_ollama: bool,
) -> ConversionResult:
    try:
        return _perform_conversion(
            source_path,
            destination,
            metadata,
            revision,
            quant_type,
            ollama_model_name,
            register_ollama,
        )
    except Exception as error:
        _raise_conversion_error(error, destination)
