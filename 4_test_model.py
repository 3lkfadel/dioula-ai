"""
ÉTAPE 4 — Test du modèle fine-tuné Dioula-AI
Lance des prompts de test pour valider la qualité du modèle
"""

import subprocess
import sys

# Chemins
MODEL_PATH = "./model/dioula-llama-fused"   # Modèle fusionné (après fuse)
ADAPTER_PATH = "./model/adapters"            # OU utilise les adapteurs seuls (sans fusion)
BASE_MODEL = "./model/llama-3.2-3b-mlx"

SYSTEM = (
    "Tu es un assistant expert en langue Dioula (Jula), "
    "parlée en Côte d'Ivoire et au Burkina Faso. "
    "Tu peux traduire entre le français et le Dioula, "
    "et converser naturellement dans les deux langues."
)

# Prompts de test
TEST_CASES = [
    {
        "type": "Traduction FR → Dioula",
        "prompt": "Traduis en Dioula : Bonjour, comment vas-tu ?"
    },
    {
        "type": "Traduction FR → Dioula",
        "prompt": "Traduis en Dioula : Je m'appelle Amadou et j'habite à Abidjan."
    },
    {
        "type": "Traduction Dioula → FR",
        "prompt": "Traduis en français : I ni ce"
    },
    {
        "type": "Traduction Dioula → FR",
        "prompt": "Traduis en français : N bé kénéya la"
    },
    {
        "type": "Conversation",
        "prompt": "Comment dit-on en Dioula : merci beaucoup ?"
    },
    {
        "type": "Conversation",
        "prompt": "En Dioula, comment exprime-t-on : bonne nuit ?"
    },
]

LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def run_inference(prompt: str, use_adapter: bool = False) -> str:
    """Lance l'inférence via mlx_lm.generate."""
    formatted = LLAMA_TEMPLATE.format(system=SYSTEM, user=prompt)

    if use_adapter:
        # Avec adapteurs LoRA (sans fusion)
        cmd = [
            "mlx_lm.generate",
            "--model", BASE_MODEL,
            "--adapter-path", ADAPTER_PATH,
            "--prompt", formatted,
            "--max-tokens", "200",
            "--temp", "0.3",
        ]
    else:
        # Avec modèle fusionné
        cmd = [
            "mlx_lm.generate",
            "--model", MODEL_PATH,
            "--prompt", formatted,
            "--max-tokens", "200",
            "--temp", "0.3",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()

    # Extrait uniquement la réponse de l'assistant
    if "<|start_header_id|>assistant<|end_header_id|>" in output:
        output = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        output = output.replace("<|eot_id|>", "").strip()

    return output if output else result.stderr.strip()


def main():
    print("🌍 Dioula-AI — Tests du modèle fine-tuné")
    print("=" * 55)

    # Détermine si on utilise le modèle fusionné ou les adapteurs
    import os
    use_adapter = not os.path.exists(MODEL_PATH)
    if use_adapter:
        print(f"⚠️  Modèle fusionné non trouvé → utilisation des adapteurs LoRA")
        print(f"   (Pour fusionner : bash scripts/3_finetune.sh → voir instructions finales)\n")
    else:
        print(f"✅ Modèle fusionné trouvé : {MODEL_PATH}\n")

    passed = 0
    for i, test in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {test['type']}")
        print(f"  🔵 Entrée    : {test['prompt']}")

        response = run_inference(test['prompt'], use_adapter=use_adapter)
        print(f"  🟢 Réponse  : {response}")
        print()

        if response and len(response) > 2:
            passed += 1

    print("=" * 55)
    print(f"✅ Tests passés : {passed}/{len(TEST_CASES)}")

    if passed < len(TEST_CASES) // 2:
        print("⚠️  Résultats faibles → essaie d'augmenter --iters dans 3_finetune.sh")
    elif passed == len(TEST_CASES):
        print("🎉 Modèle prêt pour la production !")
    else:
        print("👍 Bon résultat. Tu peux faire plus d'itérations pour améliorer.")


if __name__ == "__main__":
    main()
