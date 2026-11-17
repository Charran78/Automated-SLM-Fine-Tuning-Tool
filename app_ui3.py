import asyncio
import contextlib
import io
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from nicegui import background_tasks, ui
from transformers import TrainerCallback
from slm_automator import SLMFineTunerPipeline

# --- Coordinación global de trabajos y estado aislado por ejecución ---
_active_job_lock = threading.Lock()
_TERMINAL_EVENTS = {"loaded", "done", "stopped", "evaluated", "exported", "error"}


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

# --- UI ---
@ui.page("/")
async def main():
    await ui.context.client.connected()

    with ui.header().classes("bg-primary items-center"):
        ui.label("SLM Fine-Tuning Tool").classes("text-h5")

    with ui.tabs().classes("w-full") as tabs:
        tab_config = ui.tab("Configuración")
        tab_train = ui.tab("Entrenamiento")
        tab_eval = ui.tab("Evaluación")
        tab_export = ui.tab("Exportación")

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

    # ---------- Cableado de eventos ----------
    btn_load.on_click(on_load)
    btn_train.on_click(on_train)
    btn_stop.on_click(on_stop)
    btn_eval.on_click(on_eval)
    btn_export.on_click(on_export)

if __name__ == "__main__":
    ui.run(title="SLM Fine-Tuner", port=8080, native=False, reload=False)
