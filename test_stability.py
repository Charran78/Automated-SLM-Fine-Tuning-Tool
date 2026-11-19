import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import app_ui3
from slm_automator import SLMFineTunerPipeline


class StabilityTests(unittest.TestCase):
    def test_quantizer_resolves_visual_studio_release_output(self):
        with tempfile.TemporaryDirectory() as build_dir:
            executable_name = (
                "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
            )
            executable = Path(build_dir) / "bin" / "Release" / executable_name
            executable.parent.mkdir(parents=True)
            executable.touch()

            self.assertEqual(
                SLMFineTunerPipeline._find_llama_quantize_binary(build_dir),
                str(executable),
            )

    def test_quantizer_resolution_returns_none_when_binary_is_missing(self):
        with tempfile.TemporaryDirectory() as build_dir:
            self.assertIsNone(
                SLMFineTunerPipeline._find_llama_quantize_binary(build_dir)
            )

    def test_each_job_has_an_independent_progress_queue_and_stop_event(self):
        first = app_ui3.JobContext()
        second = app_ui3.JobContext()

        first.send({"type": "log", "step": 1})
        first.stop_event.set()

        self.assertEqual(first.progress.get_nowait(), {"type": "log", "step": 1})
        self.assertTrue(first.stop_event.is_set())
        self.assertTrue(second.progress.empty())
        self.assertFalse(second.stop_event.is_set())

    def test_all_tabs_use_the_single_pipeline_guarded_by_the_global_lock(self):
        self.assertIs(app_ui3.get_pipeline(), app_ui3.get_pipeline())

    def test_busy_global_lock_rejects_additional_work(self):
        self.assertTrue(app_ui3._active_job_lock.acquire(blocking=False))
        try:
            self.assertIsNone(app_ui3.start_job(lambda _context: None))
        finally:
            app_ui3._active_job_lock.release()

    def test_worker_errors_are_terminal_and_release_global_lock(self):
        context = app_ui3.JobContext()

        def fail(_context):
            raise RuntimeError("load failed")

        self.assertTrue(app_ui3._active_job_lock.acquire(blocking=False))
        app_ui3._execute_job(context, fail)

        message = context.progress.get_nowait()
        self.assertEqual(message["type"], "error")
        self.assertIn("load failed", message["msg"])
        self.assertTrue(context.terminal)
        self.assertFalse(app_ui3._active_job_lock.locked())

    def test_load_failure_clears_partial_pipeline_state(self):
        context = app_ui3.JobContext()
        pipeline = SimpleNamespace(
            model="partial model",
            tokenizer="partial tokenizer",
            train_dataset="partial train data",
            eval_dataset="partial eval data",
            trainer="partial trainer",
            load_model_and_tokenizer=Mock(side_effect=RuntimeError("load failed")),
        )

        self.assertTrue(app_ui3._active_job_lock.acquire(blocking=False))
        app_ui3._execute_job(context, app_ui3._load_worker, pipeline, "model", "data.jsonl")

        self.assertIsNone(pipeline.model)
        self.assertIsNone(pipeline.tokenizer)
        self.assertIsNone(pipeline.train_dataset)
        self.assertIsNone(pipeline.eval_dataset)
        self.assertIsNone(pipeline.trainer)
        self.assertTrue(context.terminal)
        self.assertEqual(context.progress.get_nowait()["type"], "status")
        self.assertEqual(context.progress.get_nowait()["type"], "error")
        self.assertFalse(app_ui3._active_job_lock.locked())

    def test_cpu_fallback_resolves_non_qwen_unsloth_models(self):
        resolve = SLMFineTunerPipeline._cpu_fallback_model_name

        self.assertEqual(
            resolve("unsloth/Llama-3.2-1B-Instruct"),
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(
            resolve("unsloth/gemma-3-1b-it-bnb-4bit"),
            "google/gemma-3-1b-it",
        )
        self.assertEqual(
            resolve("unsloth/Phi-3.5-mini-instruct"),
            "microsoft/Phi-3.5-mini-instruct",
        )

    def test_inference_uses_the_input_embedding_device(self):
        expected_device = "cuda:1"
        model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(
                weight=SimpleNamespace(device=expected_device)
            )
        )
        pipeline = SLMFineTunerPipeline()
        pipeline.model = model

        self.assertEqual(pipeline._inference_device(), expected_device)

    def test_training_and_export_paths_are_unique_per_execution(self):
        trainer = SimpleNamespace(add_callback=Mock(), train=Mock())
        pipeline = SimpleNamespace(
            model=SimpleNamespace(train=Mock()),
            use_unsloth=False,
            trainer=trainer,
            _init_trainer=Mock(),
        )
        first = app_ui3.JobContext()
        second = app_ui3.JobContext()

        app_ui3._train_worker(first, pipeline, 1, 1, 1e-4)
        first_training_path = pipeline.output_dir
        app_ui3._train_worker(second, pipeline, 1, 1, 1e-4)
        second_training_path = pipeline.output_dir

        export_pipeline = SimpleNamespace(
            export_to_gguf=Mock(side_effect=["first/Modelfile", "second/Modelfile"]),
            register_in_ollama=Mock(),
        )
        app_ui3._export_worker(first, export_pipeline, "q4_k_m", "test")
        app_ui3._export_worker(second, export_pipeline, "q4_k_m", "test")
        first_export_path = export_pipeline.export_to_gguf.call_args_list[0].kwargs["export_dir"]
        second_export_path = export_pipeline.export_to_gguf.call_args_list[1].kwargs["export_dir"]

        self.assertNotEqual(first_training_path, second_training_path)
        self.assertNotEqual(first_export_path, second_export_path)


if __name__ == "__main__":
    unittest.main()
