"""
Moteur d'inférence Dioula-AI — compatible Mac (MLX) et AWS (PyTorch)
Détecte automatiquement la plateforme au démarrage.
"""

import os
import platform

# Détection de la plateforme
_IS_APPLE_SILICON = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
)

# Chemins
FUSED_MODEL_PATH = "./model/dioula-llama-fused-dq"  # float16, compatible HuggingFace
BASE_MODEL_PATH  = "./model/llama-3.2-3b-mlx"       # Mac uniquement (MLX quantizé)
ADAPTER_PATH     = "./model/adapters"                # LoRA adapters (MLX format)

SYSTEM_PROMPT = (
    "Tu es Dioula-AI, un assistant expert en langue Dioula (Jula), "
    "parlée en Côte d'Ivoire et au Burkina Faso. "
    "Tu peux traduire entre le français et le Dioula, "
    "et converser naturellement dans les deux langues. "
    "Tes réponses sont courtes, précises et naturelles."
)

LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


# ===========================================================
# Backend MLX (Mac Apple Silicon)
# ===========================================================

class _MLXBackend:
    def __init__(self):
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        self._generate = generate
        self._make_sampler = make_sampler
        self._make_rep_penalty = make_repetition_penalty

        print("⏳ Chargement du modèle (MLX / Apple Silicon)...")
        if os.path.exists(BASE_MODEL_PATH) and os.path.exists(ADAPTER_PATH):
            print(f"  → Base      : {BASE_MODEL_PATH}")
            print(f"  → Adapteurs : {ADAPTER_PATH}")
            self.model, self.tokenizer = load(
                BASE_MODEL_PATH,
                adapter_path=ADAPTER_PATH
            )
        elif os.path.exists(FUSED_MODEL_PATH):
            print(f"  → Modèle fusionné : {FUSED_MODEL_PATH}")
            self.model, self.tokenizer = load(FUSED_MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"Modèle introuvable. Lance d'abord : bash 3_finetune.sh"
            )
        print("✅ Modèle MLX chargé !\n")

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        sampler   = self._make_sampler(temp=temperature)
        rep_pen   = self._make_rep_penalty(penalty=1.2, context_size=20)
        response  = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=[rep_pen],
            verbose=False,
        )
        return response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()


# ===========================================================
# Backend PyTorch (AWS / Linux)
# ===========================================================

class _TorchBackend:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        if not os.path.exists(FUSED_MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle fusionné introuvable : '{FUSED_MODEL_PATH}'\n"
                f"Sur AWS, seul le modèle fusionné (dequantizé) est supporté.\n"
                f"Génère-le sur Mac avec : mlx_lm.fuse --dequantize"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        print(f"⏳ Chargement du modèle (PyTorch / {device.upper()})...")
        print(f"  → Modèle : {FUSED_MODEL_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(FUSED_MODEL_PATH)
        self.model     = AutoModelForCausalLM.from_pretrained(
            FUSED_MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.device = device
        print(f"✅ Modèle PyTorch chargé sur {device.upper()} !\n")

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Garde seulement les tokens générés (pas le prompt)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response  = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()


# ===========================================================
# Classe principale (indépendante de la plateforme)
# ===========================================================

class DioulaInference:
    def __init__(self):
        if _IS_APPLE_SILICON:
            print("🍎 Plateforme : Mac Apple Silicon → backend MLX")
            self._backend = _MLXBackend()
        else:
            print("☁️  Plateforme : Linux/AWS → backend PyTorch")
            self._backend = _TorchBackend()

    def _format_prompt(self, message: str) -> str:
        return LLAMA_TEMPLATE.format(system=SYSTEM_PROMPT, user=message)

    def _detect_language(self, text: str) -> str:
        dioula_markers = [
            "ni ce", "i ni", "tɔgɔ", "bé", "kénéya", "dɛmɛ",
            "sɔrɔ", "kɔ", "fɛ", "ye", "ka", "ko", "la", "de"
        ]
        hits = sum(1 for m in dioula_markers if m in text.lower())
        return "dioula" if hits >= 2 else "français"

    def generate_response(
        self,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> dict:
        langue_input = self._detect_language(message)
        prompt       = self._format_prompt(message)
        response     = self._backend.generate(prompt, max_tokens, temperature)

        return {
            "reponse": response,
            "langue_detectee": langue_input,
            "langue_reponse": self._detect_language(response),
        }


# Singleton
_inference_engine = None

def get_engine() -> DioulaInference:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = DioulaInference()
    return _inference_engine
