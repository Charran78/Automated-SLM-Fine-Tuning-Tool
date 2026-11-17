import os
import re
import sys
import shutil
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Intentar importar Unsloth (solo funcionará si hay GPU compatible)
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    UNSLOTH_AVAILABLE = False


class SLMFineTunerPipeline:
    def __init__(self, model_name="unsloth/Qwen2.5-1.5B-Instruct", max_seq_length=2048, output_dir="./output_model"):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.use_unsloth = UNSLOTH_AVAILABLE

    @staticmethod
    def _check_dependency(cmd_name):
        """Verifica si un binario existe en el PATH del sistema."""
        if shutil.which(cmd_name) is None:
            raise EnvironmentError(f"[ERROR CRÍTICO] La dependencia '{cmd_name}' no se encontró en el PATH del sistema.")

    @staticmethod
    def _cpu_fallback_model_name(model_name):
        """Resuelve modelos Unsloth conocidos a su repositorio base de Transformers."""
        model_name = re.sub(r"-bnb-4bit$", "", model_name)
        upstream_prefixes = {
            "unsloth/Qwen2.5-": "Qwen/Qwen2.5-",
            "unsloth/Llama-3.2-": "meta-llama/Llama-3.2-",
            "unsloth/gemma-3-": "google/gemma-3-",
            "unsloth/Phi-3.5-": "microsoft/Phi-3.5-",
        }
        for source_prefix, upstream_prefix in upstream_prefixes.items():
            if model_name.startswith(source_prefix):
                return upstream_prefix + model_name[len(source_prefix):]
        if model_name.startswith("unsloth/"):
            raise ValueError(
                f"No hay un modelo base de Transformers conocido para '{model_name}'. "
                "Indica el repositorio upstream directamente para usar el fallback CPU."
            )
        return model_name

    def _inference_device(self):
        """Devuelve el dispositivo de entrada real del modelo, incluso con device_map."""
        input_embeddings = self.model.get_input_embeddings()
        embedding_weight = getattr(input_embeddings, "weight", None)
        if embedding_weight is not None:
            return embedding_weight.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_model_and_tokenizer(self):
        """Paso 1: Carga modelo y tokenizador con Unsloth (GPU) o Fallback estándar PEFT (CPU)."""
        if self.use_unsloth:
            print(f"[+] GPU detectada. Cargando modelo con Unsloth (QLoRA 4-bit NF4): {self.model_name}...")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                dtype=compute_dtype,
            )
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
            )
        else:
            print("[!] Sin GPU compatible. Activando modo Fallback CPU (Transformers + PEFT estándar)...")
            fallback_model = self._cpu_fallback_model_name(self.model_name)
            print(f"[+] Cargando modelo en FP32 para CPU: {fallback_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                dtype=torch.float32,
                device_map="cpu"
            )
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

    def prepare_dataset(self, dataset_path_or_name, chat_template_name="qwen25", test_size=0.1):
        """Paso 2: Carga, normaliza/valida plantilla de chat y EOS, y divide en train/eval."""
        if "qwen" in chat_template_name.lower():
            chat_template_name = "qwen25"

        print(f"[+] Cargando dataset desde: {dataset_path_or_name}...")
        if dataset_path_or_name.endswith(".json") or dataset_path_or_name.endswith(".jsonl"):
            raw_dataset = load_dataset("json", data_files=dataset_path_or_name, split="train")
        else:
            raw_dataset = load_dataset(dataset_path_or_name, split="train")

        # Aplicar plantilla de chat de forma segura
        if self.use_unsloth:
            self.tokenizer = get_chat_template(self.tokenizer, chat_template=chat_template_name)

        # Validación robusta de chat_template y eos_token
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise ValueError("[ERROR] El tokenizador no tiene un 'chat_template' definido. No se puede formatear de forma segura.")

        if "qwen" in chat_template_name.lower() or "qwen" in self.model_name.lower():
            if "<|im_end|>" in self.tokenizer.get_vocab():
                self.tokenizer.eos_token = "<|im_end|>"
                self.tokenizer.pad_token = "<|im_end|>"
            else:
                print("[ADVERTENCIA] Token '<|im_end|>' no encontrado en el vocabulario. Verifica el tokenizador.")

        def formatting_prompts_func(examples):
            return {
                "text": self.tokenizer.apply_chat_template(
                    examples["messages"], tokenize=False, add_generation_prompt=False
                )
            }

        if "messages" in raw_dataset.column_names:
            raw_dataset = raw_dataset.map(formatting_prompts_func, batched=True)

        # Dividir en conjunto de entrenamiento y validación (held-out)
        if len(raw_dataset) > 10 and test_size > 0:
            split_ds = raw_dataset.train_test_split(test_size=test_size, seed=42)
            self.train_dataset = split_ds["train"]
            self.eval_dataset = split_ds["test"]
            print(f"[+] Dataset dividido: {len(self.train_dataset)} train / {len(self.eval_dataset)} eval.")
        else:
            self.train_dataset = raw_dataset
            self.eval_dataset = None
            print(f"[+] Dataset cargado sin split de evaluación ({len(self.train_dataset)} ejemplos).")

    def _init_trainer(self, epochs=3, batch_size=2, learning_rate=2e-4):
        """Inicializa y almacena una única instancia de SFTTrainer para entrenamiento y evaluación."""
        has_cuda = torch.cuda.is_available()
        use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        use_fp16 = has_cuda and not use_bf16
        grad_accum = 4

        common_args = dict(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch" if self.eval_dataset is not None else "no",
            optim="adamw_8bit" if self.use_unsloth else "adamw_torch",
            bf16=use_bf16,
            fp16=use_fp16,
            dataset_text_field="text",
            max_length=self.max_seq_length,
        )

        try:
            sft_config = SFTConfig(warmup_ratio=0.1, **common_args)
        except TypeError:
            # Fallback adaptativo: calcular warmup_steps como el 10% de los pasos totales estimados.
            # Nota: Esta estimación asume packing=False; si activas packing=True en el futuro,
            # los pasos por época serán menores debido al empaquetamiento de secuencias.
            num_examples = len(self.train_dataset) if self.train_dataset is not None else 100
            steps_per_epoch = max(1, num_examples // (batch_size * grad_accum))
            total_steps = steps_per_epoch * epochs
            warmup_steps = max(1, int(0.1 * total_steps))
            sft_config = SFTConfig(warmup_steps=warmup_steps, **common_args)

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=sft_config,
        )

    def evaluate_model(self, test_prompts=None, stage_label="Evaluación"):
        """Paso 3 (y 5): Evalúa pérdida reutilizando self.trainer y genera muestras cualitativas."""
        print(f"\n--- [{stage_label}] ---")
        if self.eval_dataset is not None:
            if self.trainer is None:
                self._init_trainer()
            self.model.train()
            if self.use_unsloth and hasattr(FastLanguageModel, "for_training"):
                self.model = FastLanguageModel.for_training(self.model)
            metrics = self.trainer.evaluate()
            print(f"[*] Eval Loss ({stage_label}): {metrics.get('eval_loss', 'N/A'):.4f}")

        if test_prompts:
            # En versiones modernas de Unsloth basta con model.eval() + no_grad() para inferencia
            self.model.eval()
            device = self._inference_device()
            with torch.no_grad():
                for prompt in test_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    inputs = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to(device)
                    outputs = self.model.generate(
                        input_ids=inputs,
                        max_new_tokens=128,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    response = self.tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
                    print(f"  > Prompt: {prompt}\n  > Respuesta ({stage_label}): {response.strip()}\n")

    def train(self, epochs=3, batch_size=2, learning_rate=2e-4):
        """
        Paso 4: Ejecuta SFT usando la API actualizada de TRL (SFTConfig).
        Nota de rendimiento: El valor por defecto batch_size=2 es seguro para GPUs de 6-8 GB VRAM.
        Para GPUs con >=12 GB de VRAM (ej. RTX 3060 12GB, 4070+), considera aumentar batch_size a 4 u 8.
        """
        print("[+] Iniciando entrenamiento supervisado (SFT)...")
        self.model.train()
        if self.use_unsloth and hasattr(FastLanguageModel, "for_training"):
            self.model = FastLanguageModel.for_training(self.model)

        self._init_trainer(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        trainer_stats = self.trainer.train()
        print("[+] Entrenamiento completado con éxito.")
        return trainer_stats

    def export_to_gguf(self, export_dir="./gguf_output", quant_type="q4_k_m"):
        """Paso 6: Exporta a GGUF cuantizado y genera el Modelfile (Nativo Unsloth en GPU o Fallback llama.cpp en CPU)."""
        os.makedirs(export_dir, exist_ok=True)
        modelfile_path = os.path.join(export_dir, "Modelfile")

        if self.use_unsloth:
            # Ruta simplificada oficial: fusiona, cuantiza y genera el Modelfile nativo en un solo paso
            print(f"[+] Usando save_pretrained_gguf de Unsloth (cuantización: {quant_type})...")
            self.model.save_pretrained_gguf(export_dir, self.tokenizer, quantization_method=quant_type)

            # Búsqueda resiliente del Modelfile (raíz o subdirectorios *_gguf/ según PR #4455)
            if not os.path.exists(modelfile_path):
                found_modelfile = None
                for root, _, files in os.walk(export_dir):
                    if "Modelfile" in files:
                        found_modelfile = os.path.join(root, "Modelfile")
                        break
                if found_modelfile:
                    modelfile_path = found_modelfile
                else:
                    raise FileNotFoundError(
                        f"[ERROR] Unsloth no generó ningún Modelfile en {export_dir} ni en sus subdirectorios."
                    )

            # Parche contra issue #2604 de Unsloth: asegurar que FROM apunte al GGUF cuantizado y no al F16
            gguf_files = []
            for root, _, files in os.walk(export_dir):
                for f in files:
                    if f.endswith(".gguf") and quant_type.lower() in f.lower():
                        gguf_files.append(os.path.join(root, f))

            if not gguf_files:
                raise FileNotFoundError(
                    f"[ERROR] No se encontró el archivo GGUF cuantizado '{quant_type}' en {export_dir}."
                )

            correct_gguf_path = os.path.abspath(gguf_files[0])
            with open(modelfile_path, "r", encoding="utf-8") as f:
                modelfile_content = f.read()

            modelfile_content = re.sub(
                r"^FROM .*$",
                f"FROM {correct_gguf_path}",
                modelfile_content,
                flags=re.MULTILINE,
            )
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)

            print(f"[+] GGUF ({os.path.basename(correct_gguf_path)}) y Modelfile verificados en: {modelfile_path}")
            return modelfile_path

        # Ruta Fallback CPU: fusión manual + convert_hf_to_gguf.py + llama-quantize
        print("[!] Modo CPU detectado: ejecutando ruta manual de fusión y conversión con llama.cpp...")
        merged_dir = os.path.join(export_dir, "merged_hf")
        f16_gguf_path = os.path.join(export_dir, "temp_model_f16.gguf")
        final_gguf_path = os.path.join(export_dir, f"model_ft_{quant_type}.gguf")

        try:
            if hasattr(self.model, "unsloth_make_training_easier"):
                print("[+] Preparando modelo con unsloth_make_training_easier() antes de fusionar...")
                self.model = self.model.unsloth_make_training_easier()
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)

            self._check_dependency("git")
            self._check_dependency("cmake")

            llama_dir = "llama.cpp"
            if not os.path.exists(llama_dir):
                try:
                    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp"], check=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"[ERROR] Falló la clonación de llama.cpp: {e}")

            convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
            try:
                subprocess.run(
                    [sys.executable, convert_script, merged_dir, "--outfile", f16_gguf_path, "--outtype", "f16"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"[ERROR] Falló la conversión HF -> GGUF (F16): {e}")

            quantize_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize" + (".exe" if os.name == "nt" else ""))
            if not os.path.exists(quantize_bin):
                build_dir = os.path.join(llama_dir, "build")
                try:
                    subprocess.run(["cmake", "-S", llama_dir, "-B", build_dir], check=True)
                    subprocess.run(
                        ["cmake", "--build", build_dir, "--config", "Release", "--target", "llama-quantize", "-j"],
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"[ERROR] Falló la compilación de llama-quantize mediante CMake: {e}")

            try:
                subprocess.run([quantize_bin, f16_gguf_path, final_gguf_path, quant_type.upper()], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"[ERROR] Falló la cuantización a {quant_type.upper()}: {e}")

        finally:
            # Limpieza garantizada de artefactos temporales pesados (F16 y pesos HF fusionados)
            if os.path.exists(f16_gguf_path):
                os.remove(f16_gguf_path)
            if os.path.exists(merged_dir):
                shutil.rmtree(merged_dir, ignore_errors=True)

        # Generar Modelfile fallback
        modelfile_content = (
            f'FROM {os.path.abspath(final_gguf_path)}\n\n'
            'TEMPLATE """{{ if .System }}<|im_start|>system\n'
            '{{ .System }}<|im_end|>\n'
            '{{ end }}{{ if .Prompt }}<|im_start|>user\n'
            '{{ .Prompt }}<|im_end|>\n'
            '{{ end }}<|im_start|>assistant\n"""\n'
            'PARAMETER stop "<|im_end|>"\n'
        )
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        return modelfile_path

    def register_in_ollama(self, modelfile_path, ollama_model_name="qwen2.5-ft"):
        """Paso 7: Verifica Ollama y registra el modelo usando el Modelfile generado."""
        self._check_dependency("ollama")
        if not os.path.exists(modelfile_path):
            raise FileNotFoundError(f"[ERROR] No se encontró el Modelfile en: {modelfile_path}")

        print(f"[+] Registrando modelo en Ollama como '{ollama_model_name}' desde {modelfile_path}...")
        try:
            subprocess.run(["ollama", "create", ollama_model_name, "-f", modelfile_path], check=True)
            print(f"[+] ¡Modelo '{ollama_model_name}' listo! Ejecuta: ollama run {ollama_model_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[ERROR] No se pudo crear el modelo en Ollama. ¿Está el servicio activo? Detalle: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automatized SLM Fine-Tuning Tool: SFT + LoRA -> GGUF -> Ollama"
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Ruta al dataset JSONL (columna 'messages') o nombre de dataset de Hugging Face."
    )
    parser.add_argument(
        "--model", "-m", default="unsloth/Qwen2.5-1.5B-Instruct",
        help="Modelo base (por defecto: unsloth/Qwen2.5-1.5B-Instruct)."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Épocas de entrenamiento (default: 3).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size por dispositivo (default: 2).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4).")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fracción para evaluación (default: 0.1).")
    parser.add_argument("--quant", default="q4_k_m", help="Tipo de cuantización GGUF (default: q4_k_m).")
    parser.add_argument("--ollama-name", default="qwen2.5-ft", help="Nombre del modelo en Ollama.")
    parser.add_argument(
        "--prompt", action="append", default=None,
        help="Prompt de prueba (se puede repetir). Si se omite, se usan prompts por defecto."
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Solo entrena y evalúa, sin exportar a GGUF ni registrar en Ollama."
    )
    args = parser.parse_args()

    prompts_prueba = args.prompt or [
        "¿Qué es LoRA y para qué sirve?",
        "Explícame en dos frases qué es un adaptador de bajo rango.",
    ]

    pipeline = SLMFineTunerPipeline(model_name=args.model)
    pipeline.load_model_and_tokenizer()
    pipeline.prepare_dataset(args.dataset, test_size=args.test_size)

    # 1. Baseline
    pipeline.evaluate_model(test_prompts=prompts_prueba, stage_label="Baseline (Pre-FT)")

    # 2. Entrenamiento
    pipeline.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

    # 3. Post-FT
    pipeline.evaluate_model(test_prompts=prompts_prueba, stage_label="Post-FT")

    if not args.skip_export:
        # 4. Exportación GGUF + Modelfile
        modelfile = pipeline.export_to_gguf(export_dir="./gguf_output", quant_type=args.quant)
        # 5. Registro en Ollama
        pipeline.register_in_ollama(modelfile, ollama_model_name=args.ollama_name)
