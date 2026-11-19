import asyncio
import contextlib
import io
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from nicegui import background_tasks, ui
from transformers import TrainerCallback
from kaggle_runner import KaggleDataset, KaggleError, KaggleRunner
from remote_adapter import (
    RemoteArtifactMetadata,
    convert_remote_artifact,
    inspect_remote_artifact,
    resolve_current_revision,
)
from slm_automator import SLMFineTunerPipeline

# --- Coordinación global de trabajos y estado aislado por ejecución ---
_active_job_lock = threading.Lock()
_TERMINAL_EVENTS = {
    "loaded",
    "done",
    "stopped",
    "evaluated",
    "exported",
    "remote_imported",
    "error",
}


@dataclass
class JobContext:
    """Estado privado de una ejecución; nunca se comparte entre pestañas."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    progress: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    terminal: bool = False

    def send(self, message: dict) -> None:
        self.progress.put(message)
        if message.get("type") in _TERMINAL_EVENTS:
            self.terminal = True

DATASETS_DIR = "./Datasets"
DEFAULT_MODELS = [
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/gemma-3-1b-it",
    "unsloth/gemma-3-4b-it",
    "unsloth/Phi-3.5-mini-instruct",
]
_shared_pipeline = SLMFineTunerPipeline()

# --- Callback real de Hugging Face -> cola -> UI ---
class QueueProgressCallback(TrainerCallback):
    """Envía métricas y cancelación a la ejecución que consume la UI."""
    def __init__(self, context: JobContext):
        self.context = context
        self._stopping_reported = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.context.send({
                "type": "log",
                "step": state.global_step,
                "total": state.max_steps,
                "loss": logs["loss"],
            })

    def on_step_end(self, args, state, control, **kwargs):
        if self.context.stop_event.is_set():
            control.should_training_stop = True
            if not self._stopping_reported:
                self.context.send({"type": "stopping"})
                self._stopping_reported = True
        return control

# --- Utilidades ---
def list_datasets() -> list[str]:
    """Lista los archivos .jsonl de la carpeta Datasets/."""
    if not os.path.isdir(DATASETS_DIR):
        return []
    return sorted(
        os.path.join(DATASETS_DIR, f)
        for f in os.listdir(DATASETS_DIR)
        if f.endswith(".jsonl") or f.endswith(".json")
    )


def _latest_remote_artifact_path() -> str:
    """Sugiere el ZIP más reciente; el usuario puede elegir otra ruta."""
    artifacts_dir = Path.home() / "kaggle_output"
    try:
        candidates = list(artifacts_dir.glob("*/model_artifact.zip"))
        return str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else ""
    except OSError:
        return ""


def _new_sibling_output_path(archive_path: str) -> str:
    """Propone una carpeta de salida hermana que todavía no exista."""
    archive = Path(archive_path).expanduser()
    base = archive.with_name(f"{archive.stem}_gguf")
    candidate = base
    suffix = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return str(candidate)


def get_pipeline() -> SLMFineTunerPipeline:
    """Devuelve el único pipeline, compartido por la app y protegido por el lock global."""
    return _shared_pipeline

# --- Workers que se ejecutan en hilos separados ---
def _execute_job(context: JobContext, worker, *args) -> None:
    """Asegura un estado terminal y libera el lock incluso si el worker falla."""
    try:
        worker(context, *args)
    except Exception as error:  # noqa: BLE001
        context.send({"type": "error", "msg": repr(error)})
    finally:
        if not context.terminal:
            context.send({
                "type": "error",
                "msg": "El trabajo terminó sin informar un estado final.",
            })
        _active_job_lock.release()


def start_job(worker, *args) -> JobContext | None:
    """Inicia un trabajo si no hay otro trabajo pesado activo."""
    if not _active_job_lock.acquire(blocking=False):
        return None

    context = JobContext()
    try:
        background_tasks.create(asyncio.to_thread(_execute_job, context, worker, *args))
    except Exception:
        _active_job_lock.release()
        raise
    return context


def _load_worker(context: JobContext, pipeline: SLMFineTunerPipeline, model_name: str, dataset_path: str):
    """Carga modelo y dataset. Se ejecuta en un hilo para no bloquear la UI."""
    pipeline.model = None
    pipeline.tokenizer = None
    pipeline.train_dataset = None
    pipeline.eval_dataset = None
    pipeline.trainer = None
    pipeline.model_name = model_name
    try:
        context.send({"type": "status", "msg": f"Cargando modelo {model_name}..."})
        pipeline.load_model_and_tokenizer()
        context.send({"type": "status", "msg": "Modelo cargado. Preparando dataset..."})
        pipeline.prepare_dataset(dataset_path, test_size=0.1)
        context.send({"type": "status", "msg": "Dataset preparado. Listo para entrenar."})
        context.send({"type": "loaded"})
    except Exception:
        pipeline.model = None
        pipeline.tokenizer = None
        pipeline.train_dataset = None
        pipeline.eval_dataset = None
        pipeline.trainer = None
        raise


def _train_worker(context: JobContext, pipeline: SLMFineTunerPipeline, epochs: int, batch_size: int, lr: float):
    """Entrena en un hilo. Replica la lógica de pipeline.train() añadiendo el callback."""
    pipeline.output_dir = os.path.join("./output_model", context.run_id)
    pipeline.model.train()
    if pipeline.use_unsloth and hasattr(
        __import__("unsloth").FastLanguageModel, "for_training"
    ):
        from unsloth import FastLanguageModel
        pipeline.model = FastLanguageModel.for_training(pipeline.model)

    pipeline._init_trainer(epochs=epochs, batch_size=batch_size, learning_rate=lr)
    pipeline.trainer.add_callback(QueueProgressCallback(context))
    context.send({"type": "started"})
    pipeline.trainer.train()
    context.send({"type": "stopped" if context.stop_event.is_set() else "done"})


def _eval_worker(context: JobContext, pipeline: SLMFineTunerPipeline, prompts: list[str]) -> None:
    """Ejecuta evaluate_model capturando su salida de stdout para mostrarla en la UI."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pipeline.evaluate_model(test_prompts=prompts, stage_label="Evaluación UI")
    context.send({"type": "evaluated", "text": buf.getvalue()})


def _export_worker(context: JobContext, pipeline: SLMFineTunerPipeline, quant: str, ollama_name: str) -> None:
    """Exporta a GGUF y registra en Ollama capturando la salida."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_dir = os.path.join("./gguf_output", context.run_id)
        modelfile = pipeline.export_to_gguf(export_dir=export_dir, quant_type=quant)
        pipeline.register_in_ollama(modelfile, ollama_model_name=ollama_name)
    context.send({"type": "exported", "text": buf.getvalue()})


def _remote_import_worker(
    context: JobContext,
    archive_path: str,
    output_dir: str,
    model_revision: str | None,
    acknowledge_unrecorded_revision: bool,
    quant_type: str,
    ollama_model_name: str,
    register_ollama: bool,
) -> None:
    """Importa el ZIP en un pipeline aislado del estado de entrenamiento."""
    context.send({"type": "remote_started"})
    try:
        result = convert_remote_artifact(
            archive_path,
            output_dir,
            model_revision,
            acknowledge_unrecorded_revision,
            quant_type,
            ollama_model_name,
            register_ollama,
        )
    except Exception as error:  # noqa: BLE001
        destination = Path(output_dir)
        generated_files = sorted(
            str(path)
            for pattern in ("*.gguf", "Modelfile")
            for path in destination.rglob(pattern)
        ) if destination.is_dir() else []
        details = f"Error durante la importación: {error}"
        if generated_files:
            details += "\nArchivos generados (salida parcial conservada):\n"
            details += "\n".join(generated_files)
        elif destination.exists():
            details += f"\nSalida parcial conservada en: {destination}"
        context.send({"type": "error", "msg": details})
        return
    context.send({"type": "remote_imported", "result": result})


# --- UI ---
@ui.page("/")
async def main():
    await ui.context.client.connected()

    with ui.header().classes("bg-primary items-center"):
        ui.label("SLM Fine-Tuning Tool").classes("text-h5")

    remote_metadata: RemoteArtifactMetadata | None = None
    remote_inspected_path = ""

    with ui.tabs().classes("w-full") as tabs:
        tab_config = ui.tab("Configuración")
        tab_train = ui.tab("Entrenamiento")
        tab_eval = ui.tab("Evaluación")
        tab_export = ui.tab("Exportación")
        tab_kaggle = ui.tab("Kaggle (opcional)")

    with ui.tab_panels(tabs, value=tab_config).classes("w-full"):
        # ---------- Configuración ----------
        with ui.tab_panel(tab_config):
            ui.label("Modelo base").classes("text-subtitle1")
            with ui.row().classes("w-full items-center"):
                model_select = ui.select(
                    DEFAULT_MODELS,
                    value=DEFAULT_MODELS[0],
                    label="Modelos populares",
                ).classes("flex-grow")
                model_manual = ui.input("…o escribe un repo de Hugging Face").classes("flex-grow")
            ui.label(
                "El fine-tuning se hace sobre pesos en formato Transformers (safetensors), "
                "no sobre GGUF. La cuantización (Q4_K_M, Q6_K…) se elige al exportar."
            ).classes("text-caption text-grey-7")

            ui.separator()
            ui.label("Dataset").classes("text-subtitle1")
            datasets = list_datasets()
            with ui.row().classes("w-full items-center"):
                dataset_select = ui.select(
                    datasets,
                    label="Archivos en ./Datasets/",
                    value=datasets[0] if datasets else None,
                ).classes("flex-grow")
                dataset_manual = ui.input("…o escribe una ruta").classes("flex-grow")
                ui.button(icon="refresh", on_click=lambda: dataset_select.set_options(list_datasets()))

            ui.separator()
            ui.label("Hiperparámetros").classes("text-subtitle1")
            with ui.row().classes("w-full items-center gap-4"):
                epochs_input = ui.number("Épocas", value=3, min=1, max=20, step=1).classes("w-32")
                batch_input = ui.number("Batch size", value=2, min=1, max=16, step=1).classes("w-32")
                lr_select = ui.select(
                    [5e-5, 1e-4, 2e-4, 5e-4],
                    value=2e-4,
                    label="Learning rate",
                ).classes("w-48")
            ui.label(
                "Épocas: cuántas veces el modelo ve el dataset completo. "
                "Batch size: ejemplos procesados a la vez. "
                "Learning rate: tamaño del paso de optimización (2e-4 = 0.0002)."
            ).classes("text-caption text-grey-7")

            btn_load = ui.button("Cargar modelo y dataset", icon="download")
            load_status = ui.label("").classes("text-caption")

        # ---------- Entrenamiento ----------
        with ui.tab_panel(tab_train):
            progress_bar = ui.linear_progress(value=0, show_value=False).props(
                "instant-feedback rounded"
            )
            progress_label = ui.label("Listo para entrenar")
            log_area = ui.log(max_lines=200).classes(
                "w-full h-64 bg-gray-900 text-green-400 font-mono text-xs"
            )
            chart = ui.echart({
                "title": {"text": "Training Loss"},
                "xAxis": {"type": "category", "data": [], "name": "Paso"},
                "yAxis": {"type": "value", "name": "Loss"},
                "series": [{"type": "line", "data": [], "smooth": True, "name": "Loss"}],
                "tooltip": {"trigger": "axis"},
            }).classes("w-full h-72")
            with ui.row():
                btn_train = ui.button("Iniciar entrenamiento", icon="play_arrow")
                btn_stop = ui.button("Detener", icon="stop", color="negative")
                btn_stop.disable()

        # ---------- Evaluación ----------
        with ui.tab_panel(tab_eval):
            ui.label("Evaluación cualitativa").classes("text-subtitle1")
            eval_prompts = ui.textarea(
                "Prompts (uno por línea)",
                value="¿Qué es LoRA y para qué sirve?\nExplícame QLoRA en dos frases.",
            ).classes("w-full")
            btn_eval = ui.button("Evaluar modelo actual", icon="science")
            eval_output = ui.textarea("Resultados").classes("w-full h-64").props("readonly")

        # ---------- Exportación ----------
        with ui.tab_panel(tab_export):
            quant_select = ui.select(
                ["q4_k_m", "q5_k_m", "q6_k", "q8_0"],
                value="q4_k_m",
                label="Cuantización GGUF",
            )
            ui.label(
                "Q4_K_M: máximo ahorro. Q5_K_M: equilibrio. "
                "Q6_K: casi sin pérdida de calidad. Q8_0: prácticamente FP16."
            ).classes("text-caption text-grey-7")
            ollama_name = ui.input("Nombre en Ollama", value="qwen2.5-ft")
            btn_export = ui.button("Exportar a GGUF y registrar en Ollama", icon="publish")
            export_log = ui.log(max_lines=100).classes(
                "w-full h-48 bg-gray-900 text-green-400 font-mono text-xs"
            )
            ui.separator()
            ui.label("Importar adaptador remoto desde ZIP").classes("text-subtitle1")
            suggested_archive = _latest_remote_artifact_path()
            if suggested_archive:
                ui.label(
                    f"ZIP más reciente sugerido: {suggested_archive}"
                ).classes("text-caption text-grey-7")
            remote_archive_path = ui.input(
                "Ruta local al ZIP",
                value=suggested_archive,
                placeholder="Puedes indicar cualquier ZIP local compatible",
            ).classes("w-full")
            remote_output_dir = ui.input(
                "Carpeta de salida (nueva, hermana del ZIP)",
                value=_new_sibling_output_path(suggested_archive)
                if suggested_archive
                else "",
            ).classes("w-full")
            with ui.row().classes("items-center"):
                btn_inspect_remote = ui.button(
                    "Inspeccionar ZIP", icon="fact_check"
                )
                btn_resolve_revision = ui.button(
                    "Resolver HEAD actual", icon="sync"
                )
            remote_inspect_status = ui.label("").classes("text-caption")
            remote_metadata_label = ui.label("").classes("text-body2")
            remote_revision_warning = ui.label("").classes("text-orange-9")
            remote_unrecorded_warning = ui.label(
                "ATENCIÓN: el manifiesto no registra el SHA de entrenamiento. "
                "Debes resolver o indicar explícitamente un SHA de 40 caracteres "
                "y aceptar esta revisión no registrada antes de importar."
            ).classes("text-orange-9")
            remote_unrecorded_warning.set_visibility(False)
            remote_revision_input = ui.input(
                "SHA de revisión del modelo (40 caracteres hexadecimales)"
            ).classes("w-full")
            remote_acknowledge_revision = ui.checkbox(
                "Acepto importar usando una revisión que no está registrada "
                "en el manifiesto."
            )
            remote_acknowledge_revision.set_visibility(False)
            register_remote_ollama = ui.checkbox(
                "Registrar el modelo importado en Ollama",
                value=True,
            )
            btn_import_remote = ui.button(
                "Importar ZIP a GGUF", icon="archive"
            )
            remote_import_log = ui.log(max_lines=100).classes(
                "w-full h-48 bg-gray-900 text-green-400 font-mono text-xs"
            )

            def on_remote_archive_changed():
                nonlocal remote_metadata, remote_inspected_path
                remote_metadata = None
                remote_inspected_path = ""
                remote_metadata_label.text = ""
                remote_inspect_status.text = ""
                remote_revision_warning.text = ""
                remote_unrecorded_warning.set_visibility(False)
                remote_acknowledge_revision.set_visibility(False)
                remote_revision_input.value = ""
                archive_value = (remote_archive_path.value or "").strip()
                remote_output_dir.value = (
                    _new_sibling_output_path(archive_value) if archive_value else ""
                )

            remote_archive_path.on_value_change(on_remote_archive_changed)

        # ---------- Entrenamiento remoto opcional ----------
        with ui.tab_panel(tab_kaggle):
            ui.label("Entrenamiento remoto con Kaggle").classes("text-subtitle1")
            ui.label(
                "Requiere Kaggle CLI instalado y autenticado localmente. "
                "El flujo local existente no depende de Kaggle."
            ).classes("text-caption text-grey-7")
            kaggle_kernel_id = ui.input(
                "ID privado del notebook (usuario/slug)",
                placeholder="tu-usuario/slm-finetuning",
            ).classes("w-full")
            kaggle_model_revision = ui.input(
                "SHA de revisión del modelo (opcional, 40 caracteres hexadecimales)"
            ).classes("w-full")
            kaggle_dataset_kind = ui.select(
                {
                    "kaggle": "Referencia de dataset de Kaggle",
                    "huggingface": "Dataset público de Hugging Face",
                    "local": "Archivo local (subir como dataset privado)",
                },
                value="kaggle",
                label="Origen del dataset",
            ).classes("w-full")
            kaggle_dataset_ref = ui.input(
                "Referencia Kaggle (usuario/slug)"
            ).classes("w-full")
            kaggle_hf_dataset = ui.input(
                "ID público de dataset Hugging Face"
            ).classes("w-full")
            kaggle_hf_revision = ui.input(
                "SHA de revisión del dataset (40 caracteres hexadecimales)"
            ).classes("w-full")
            kaggle_local_dataset = ui.input(
                "Ruta al archivo JSON/JSONL local",
                value=(dataset_manual.value or dataset_select.value or ""),
            ).classes("w-full")
            kaggle_upload_consent = ui.checkbox(
                "Autorizo subir este archivo a Kaggle como dataset privado."
            )
            ui.label(
                "El kernel se crea privado. Hugging Face remoto requiere internet habilitado "
                "en Kaggle; la disponibilidad de esa opción por API depende de Kaggle."
            ).classes("text-caption text-orange-9")
            kaggle_submit_button = ui.button(
                "Enviar entrenamiento a Kaggle", icon="cloud_upload"
            )
            kaggle_status_button = ui.button("Consultar estado", icon="sync")
            kaggle_download_button = ui.button(
                "Descargar artefactos", icon="download"
            )
            kaggle_result = ui.textarea("Estado y resultado").classes(
                "w-full h-48"
            ).props("readonly")

            def update_kaggle_dataset_fields():
                kind = kaggle_dataset_kind.value
                kaggle_dataset_ref.set_visibility(kind == "kaggle")
                kaggle_hf_dataset.set_visibility(kind == "huggingface")
                kaggle_hf_revision.set_visibility(kind == "huggingface")
                kaggle_local_dataset.set_visibility(kind == "local")
                kaggle_upload_consent.set_visibility(kind == "local")

            update_kaggle_dataset_fields()
            kaggle_dataset_kind.on_value_change(update_kaggle_dataset_fields)

    # ---------- Callbacks ----------
    def resolve_model() -> str:
        return model_manual.value.strip() or model_select.value

    def resolve_dataset() -> str:
        return dataset_manual.value.strip() or dataset_select.value

    async def next_message(context: JobContext) -> dict:
        while True:
            try:
                return context.progress.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)

    def notify_job_busy() -> None:
        ui.notify(
            "Ya hay otro trabajo pesado en curso. Espera a que termine.",
            type="warning",
        )

    active_train_context: JobContext | None = None

    async def on_load():
        model = resolve_model()
        dataset = resolve_dataset()
        if not model or not dataset:
            ui.notify("Selecciona un modelo y un dataset", type="warning")
            return

        pipeline = get_pipeline()
        context = start_job(_load_worker, pipeline, model, dataset)
        if context is None:
            notify_job_busy()
            return

        btn_load.disable()
        load_status.text = "Cargando… (puede tardar si el modelo no está en caché)"

        async def consume_load(job: JobContext):
            try:
                while True:
                    msg = await next_message(job)
                    if msg["type"] == "status":
                        load_status.text = msg["msg"]
                    elif msg["type"] == "loaded":
                        load_status.text = "Modelo y dataset listos."
                        ui.notify("Carga completada", type="positive")
                        break
                    elif msg["type"] == "error":
                        load_status.text = f"Error: {msg['msg']}"
                        ui.notify("Error durante la carga", type="negative")
                        break
            finally:
                btn_load.enable()

        background_tasks.create(consume_load(context))

    async def on_train():
        nonlocal active_train_context

        pipeline = get_pipeline()
        if pipeline.model is None or pipeline.train_dataset is None:
            ui.notify("Primero carga el modelo y el dataset", type="warning")
            return

        context = start_job(
            _train_worker,
            pipeline,
            int(epochs_input.value),
            int(batch_input.value),
            float(lr_select.value),
        )
        if context is None:
            notify_job_busy()
            return

        active_train_context = context
        progress_bar.value = 0
        log_area.clear()
        chart.options["xAxis"]["data"].clear()
        chart.options["series"][0]["data"].clear()
        chart.update()
        btn_train.disable()
        btn_stop.enable()

        async def consume_train(job: JobContext):
            nonlocal active_train_context

            try:
                while True:
                    msg = await next_message(job)
                    event_type = msg["type"]
                    if event_type == "started":
                        log_area.push("Entrenamiento iniciado.")
                        progress_label.text = "Entrenando…"
                    elif event_type == "log":
                        step, total, loss = msg["step"], msg["total"], msg["loss"]
                        progress_bar.value = (step / total) if total else 0
                        progress_label.text = f"Paso {step}/{total} — loss={loss:.4f}"
                        log_area.push(f"Paso {step}: loss={loss:.4f}")
                        chart.options["xAxis"]["data"].append(str(step))
                        chart.options["series"][0]["data"].append(loss)
                        chart.update()
                    elif event_type == "stopping":
                        log_area.push("Detención solicitada; esperando al paso actual.")
                    elif event_type == "stopped":
                        progress_label.text = "Entrenamiento detenido por el usuario."
                        log_area.push("Entrenamiento detenido.")
                        progress_bar.value = 0
                        break
                    elif event_type == "done":
                        progress_label.text = "Entrenamiento completado."
                        log_area.push("Entrenamiento finalizado.")
                        progress_bar.value = 1.0
                        break
                    elif event_type == "error":
                        progress_label.text = f"Error: {msg['msg']}"
                        log_area.push(f"Error: {msg['msg']}")
                        break
            finally:
                active_train_context = None
                btn_train.enable()
                btn_stop.disable()

        background_tasks.create(consume_train(context))

    def on_stop():
        if active_train_context is None or active_train_context.terminal:
            return
        active_train_context.stop_event.set()
        ui.notify("Detención solicitada; se aplicará al final del paso actual.", type="info")

    async def on_eval():
        pipeline = get_pipeline()
        if pipeline.model is None:
            ui.notify("Carga primero un modelo", type="warning")
            return
        prompts = [p for p in (eval_prompts.value or "").splitlines() if p.strip()]
        context = start_job(_eval_worker, pipeline, prompts)
        if context is None:
            notify_job_busy()
            return

        btn_eval.disable()
        eval_output.value = "Evaluando… (en CPU puede tardar varios minutos)"
        try:
            while True:
                msg = await next_message(context)
                if msg["type"] == "evaluated":
                    eval_output.value = msg["text"]
                    ui.notify("Evaluación completada", type="positive")
                    break
                if msg["type"] == "error":
                    eval_output.value = f"Error: {msg['msg']}"
                    ui.notify("Error en la evaluación", type="negative")
                    break
        finally:
            btn_eval.enable()

    async def on_inspect_remote():
        nonlocal remote_metadata, remote_inspected_path
        archive_value = (remote_archive_path.value or "").strip()
        if not archive_value:
            ui.notify("Indica la ruta local del ZIP", type="warning")
            return

        archive_path = Path(archive_value).expanduser()
        btn_inspect_remote.disable()
        remote_inspect_status.text = "Inspeccionando ZIP…"
        remote_metadata_label.text = ""
        remote_revision_warning.text = ""
        remote_unrecorded_warning.set_visibility(False)
        remote_acknowledge_revision.set_visibility(False)
        try:
            metadata = await asyncio.to_thread(inspect_remote_artifact, archive_path)
        except Exception as error:  # noqa: BLE001
            remote_metadata = None
            remote_inspected_path = ""
            remote_inspect_status.text = f"No se pudo inspeccionar el ZIP: {error}"
            ui.notify("No se pudo inspeccionar el ZIP", type="negative")
        else:
            remote_metadata = metadata
            remote_inspected_path = str(archive_path.resolve())
            recorded_revision = metadata.recorded_revision or "NO REGISTRADA"
            adapter_revision = metadata.adapter_revision or "no indicada"
            remote_metadata_label.text = (
                f"Modelo: {metadata.model_id} | "
                f"SHA de entrenamiento en manifiesto: {recorded_revision} | "
                f"Revisión del adaptador: {adapter_revision} | "
                f"SHA-256 del ZIP: {metadata.archive_sha256}"
            )
            remote_inspect_status.text = "ZIP válido e inspeccionado."
            remote_revision_input.value = metadata.recorded_revision or ""
            is_unrecorded = metadata.recorded_revision is None
            remote_unrecorded_warning.set_visibility(is_unrecorded)
            remote_acknowledge_revision.set_visibility(is_unrecorded)
            remote_acknowledge_revision.value = False
            ui.notify("ZIP inspeccionado correctamente", type="positive")
        finally:
            btn_inspect_remote.enable()

    async def on_resolve_remote_revision():
        if remote_metadata is None:
            ui.notify("Inspecciona primero el ZIP", type="warning")
            return
        btn_resolve_revision.disable()
        remote_revision_warning.text = "Consultando la revisión HEAD en Hugging Face…"
        remote_revision_warning.set_visibility(True)
        try:
            revision = await asyncio.to_thread(
                resolve_current_revision, remote_metadata.model_id
            )
        except Exception as error:  # noqa: BLE001
            remote_revision_warning.text = f"No se pudo resolver HEAD: {error}"
            ui.notify("No se pudo resolver la revisión actual", type="negative")
        else:
            remote_revision_input.value = revision
            recorded_revision = remote_metadata.recorded_revision
            if recorded_revision and revision != recorded_revision:
                remote_revision_warning.text = (
                    f"HEAD actual: {revision}. El manifiesto registra {recorded_revision}; "
                    "la conversión no permite sustituir esa revisión registrada."
                )
                ui.notify(
                    "HEAD difiere de la revisión registrada en el manifiesto",
                    type="warning",
                )
            else:
                remote_revision_warning.text = f"SHA resuelto: {revision}"
                ui.notify("Revisión actual resuelta", type="positive")
        finally:
            btn_resolve_revision.enable()

    async def on_import_remote():
        if remote_metadata is None:
            ui.notify("Inspecciona primero el ZIP", type="warning")
            return
        archive_value = (remote_archive_path.value or "").strip()
        archive_path = Path(archive_value).expanduser()
        if not archive_value or str(archive_path.resolve()) != remote_inspected_path:
            ui.notify("Vuelve a inspeccionar el ZIP seleccionado", type="warning")
            return

        revision = (remote_revision_input.value or "").strip()
        acknowledged = bool(remote_acknowledge_revision.value)
        if remote_metadata.recorded_revision is None:
            if not revision:
                ui.notify("Indica un SHA de revisión de 40 caracteres", type="warning")
                return
            if not acknowledged:
                ui.notify(
                    "Debes aceptar explícitamente la revisión no registrada",
                    type="warning",
                )
                return
        elif revision and revision.lower() != remote_metadata.recorded_revision:
            ui.notify(
                "El manifiesto tiene una revisión registrada que no se puede sustituir",
                type="warning",
            )
            return
        else:
            revision = remote_metadata.recorded_revision

        output_value = (remote_output_dir.value or "").strip()
        if not output_value:
            ui.notify("Indica una carpeta de salida nueva", type="warning")
            return
        output_path = Path(output_value).expanduser()
        if output_path.exists() or output_path.is_symlink():
            ui.notify("La carpeta de salida ya existe; elige una nueva", type="warning")
            return

        ollama_model_name = (ollama_name.value or "").strip()
        context = start_job(
            _remote_import_worker,
            str(archive_path),
            str(output_path),
            revision or None,
            acknowledged,
            quant_select.value,
            ollama_model_name,
            bool(register_remote_ollama.value),
        )
        if context is None:
            notify_job_busy()
            return

        btn_import_remote.disable()
        remote_import_log.clear()
        remote_import_log.push(
            f"Importación iniciada para {remote_metadata.model_id}; "
            "no se ejecutará entrenamiento."
        )
        try:
            while True:
                msg = await next_message(context)
                if msg["type"] == "remote_started":
                    remote_import_log.push(
                        "Convirtiendo el adaptador a GGUF; puede tardar varios minutos."
                    )
                elif msg["type"] == "remote_imported":
                    result = msg["result"]
                    remote_import_log.push("Importación completada.")
                    remote_import_log.push(f"Modelo: {result.model_id}")
                    remote_import_log.push(f"Revisión: {result.model_revision}")
                    remote_import_log.push(f"Salida: {result.output_dir}")
                    for gguf_path in result.gguf_files:
                        remote_import_log.push(f"GGUF: {gguf_path}")
                    remote_import_log.push(f"Modelfile: {result.modelfile_path}")
                    if result.ollama_model_name:
                        remote_import_log.push(
                            f"Registrado en Ollama como: {result.ollama_model_name}"
                        )
                    else:
                        remote_import_log.push("Registro en Ollama omitido.")
                    ui.notify("Importación del ZIP completada", type="positive")
                    break
                elif msg["type"] == "error":
                    remote_import_log.push(f"Error: {msg['msg']}")
                    ui.notify("Error durante la importación", type="negative")
                    break
        finally:
            btn_import_remote.enable()

    async def on_export():
        pipeline = get_pipeline()
        if pipeline.model is None:
            ui.notify("Carga y entrena un modelo antes de exportar", type="warning")
            return
        context = start_job(
            _export_worker, pipeline, quant_select.value, ollama_name.value
        )
        if context is None:
            notify_job_busy()
            return

        btn_export.disable()
        export_log.clear()
        export_log.push("Exportando… (puede tardar varios minutos)")
        try:
            while True:
                msg = await next_message(context)
                if msg["type"] == "exported":
                    export_log.push(msg["text"])
                    export_log.push("Exportación y registro completados.")
                    ui.notify("Modelo exportado y registrado en Ollama", type="positive")
                    break
                if msg["type"] == "error":
                    export_log.push(f"Error: {msg['msg']}")
                    ui.notify("Error durante la exportación", type="negative")
                    break
        finally:
            btn_export.enable()

    async def on_kaggle_submit():
        kind = kaggle_dataset_kind.value
        if kind == "local":
            dataset = KaggleDataset(
                kind="local",
                value=(kaggle_local_dataset.value or "").strip(),
                upload_consent=bool(kaggle_upload_consent.value),
            )
        elif kind == "huggingface":
            dataset = KaggleDataset(
                kind="huggingface",
                value=(kaggle_hf_dataset.value or "").strip(),
                revision=(kaggle_hf_revision.value or "").strip(),
            )
        else:
            dataset = KaggleDataset(
                kind="kaggle",
                value=(kaggle_dataset_ref.value or "").strip(),
            )

        kaggle_submit_button.disable()
        kaggle_result.value = "Preparando y enviando el trabajo…"
        try:
            result = await asyncio.to_thread(
                KaggleRunner().submit,
                kernel_id=(kaggle_kernel_id.value or "").strip(),
                model=resolve_model(),
                model_revision=(kaggle_model_revision.value or "").strip() or None,
                dataset=dataset,
                epochs=int(epochs_input.value or 1),
                batch_size=int(batch_input.value or 1),
                learning_rate=float(lr_select.value or 2e-4),
            )
            kaggle_kernel_id.value = result.kernel_id
            if dataset.kind == "local":
                source_summary = f"Dataset local privado subido: {result.dataset_id}"
            elif dataset.kind == "kaggle":
                source_summary = f"Dataset Kaggle asociado: {result.dataset_id}"
            else:
                source_summary = f"Dataset HF fijado: {dataset.value}@{dataset.revision}"
            kaggle_result.value = (
                f"Enviado: {result.kernel_id}\n"
                f"{source_summary}\n"
                "Kaggle iniciará la ejecución. Usa «Consultar estado» para seguirla."
            )
        except KaggleError as error:
            kaggle_result.value = f"Error: {error}"
            ui.notify("No se pudo enviar el trabajo a Kaggle", type="negative")
        finally:
            kaggle_submit_button.enable()

    async def on_kaggle_status():
        kernel_id = (kaggle_kernel_id.value or "").strip()
        kaggle_status_button.disable()
        try:
            kaggle_result.value = await asyncio.to_thread(
                KaggleRunner().status, kernel_id
            )
        except KaggleError as error:
            kaggle_result.value = f"Error: {error}"
            ui.notify("No se pudo consultar Kaggle", type="negative")
        finally:
            kaggle_status_button.enable()

    async def on_kaggle_download():
        kernel_id = (kaggle_kernel_id.value or "").strip()
        destination = os.path.join(
            os.path.expanduser("~/kaggle_output"), kernel_id.rsplit("/", 1)[-1]
        )
        kaggle_download_button.disable()
        try:
            downloaded_to = await asyncio.to_thread(
                KaggleRunner().download_output, kernel_id, destination
            )
            kaggle_result.value = f"Artefactos descargados en: {downloaded_to}"
            ui.notify("Descarga de artefactos completada", type="positive")
        except KaggleError as error:
            kaggle_result.value = f"Error: {error}"
            ui.notify("No se pudieron descargar los artefactos", type="negative")
        finally:
            kaggle_download_button.enable()

    # ---------- Cableado de eventos ----------
    btn_load.on_click(on_load)
    btn_train.on_click(on_train)
    btn_stop.on_click(on_stop)
    btn_eval.on_click(on_eval)
    btn_export.on_click(on_export)
    btn_inspect_remote.on_click(on_inspect_remote)
    btn_resolve_revision.on_click(on_resolve_remote_revision)
    btn_import_remote.on_click(on_import_remote)
    kaggle_submit_button.on_click(on_kaggle_submit)
    kaggle_status_button.on_click(on_kaggle_status)
    kaggle_download_button.on_click(on_kaggle_download)

if __name__ == "__main__":
    ui.run(title="SLM Fine-Tuner", port=8080, native=False, reload=False)
