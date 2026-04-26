#!/usr/bin/env python3
"""
Benchmark d'évaluation Dioula-AI — version corrigée
Utilise le modèle fusionné et un prompt plus strict
"""

import json
import subprocess

BENCHMARK = [
    {"fr": "Bonjour", "dioula": "I ni ce"},
    {"fr": "Bonsoir", "dioula": "I ni wula"},
    {"fr": "Bonne nuit", "dioula": "Su cɛ"},
    {"fr": "Bienvenue", "dioula": "I bisimila"},
    {"fr": "Au revoir", "dioula": "Kana taa"},
    {"fr": "Comment vas-tu ?", "dioula": "I ka kénéya ?"},
    {"fr": "Je vais bien", "dioula": "N ka kénéya"},
    {"fr": "Merci beaucoup", "dioula": "Aw ni ce"},
    {"fr": "S'il te plaît", "dioula": "Aw ni baara"},
    {"fr": "Pardon", "dioula": "Hakɛ tooro"},
    {"fr": "Ma mère", "dioula": "N ba"},
    {"fr": "Mon père", "dioula": "N fa"},
    {"fr": "Mon frère", "dioula": "N kɔrɔ"},
    {"fr": "Ma sœur", "dioula": "N teri"},
    {"fr": "Mon ami", "dioula": "N teri"},
    {"fr": "Mon fils", "dioula": "N denmisɛn"},
    {"fr": "La famille", "dioula": "Fadenya"},
    {"fr": "Le mari", "dioula": "Cɛ"},
    {"fr": "La femme", "dioula": "Muso"},
    {"fr": "Un", "dioula": "Kelen"},
    {"fr": "Deux", "dioula": "Fila"},
    {"fr": "Trois", "dioula": "Saba"},
    {"fr": "Quatre", "dioula": "Naani"},
    {"fr": "Cinq", "dioula": "Duuru"},
    {"fr": "Dix", "dioula": "Tan"},
    {"fr": "Cent", "dioula": "Kɛmɛ"},
    {"fr": "Mille", "dioula": "Waa kelen"},
    {"fr": "L'eau", "dioula": "Jii"},
    {"fr": "La nourriture", "dioula": "Dumu"},
    {"fr": "Le marché", "dioula": "Maarɛ"},
    {"fr": "La maison", "dioula": "So"},
    {"fr": "Le travail", "dioula": "Baara"},
    {"fr": "L'argent", "dioula": "Wari"},
    {"fr": "La route", "dioula": "Sira"},
    {"fr": "Le village", "dioula": "Dugu"},
    {"fr": "La ville", "dioula": "Dugu ba"},
    {"fr": "Je ne comprends pas", "dioula": "N ma a faamu"},
    {"fr": "Je comprends", "dioula": "N ye a faamu"},
    {"fr": "Où est le marché ?", "dioula": "Maarɛ bɛ min ?"},
    {"fr": "Combien ça coûte ?", "dioula": "A joli ?"},
    {"fr": "C'est trop cher", "dioula": "A gélén"},
    {"fr": "J'ai faim", "dioula": "Kɔnɔ bɛ n dɔn"},
    {"fr": "J'ai soif", "dioula": "Jii bɛ n sɔn"},
    {"fr": "Je suis fatigué", "dioula": "N dɛsɛra"},
    {"fr": "Dieu merci", "dioula": "Ala ka bato"},
]

MODEL_PATH = "./model/dioula-llama-fused"

# Prompt très strict — demande UNIQUEMENT la traduction, rien d'autre
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Tu es un traducteur Français-Dioula. "
    "Quand on te donne un mot ou une phrase en français, "
    "tu réponds UNIQUEMENT avec la traduction en Dioula. "
    "Pas d'explication. Pas de phrase. Juste la traduction.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Traduis en Dioula : {text}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def get_translation(text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(text=text)
    cmd = [
        "mlx_lm.generate",
        "--model", MODEL_PATH,
        "--prompt", prompt,
        "--max-tokens", "20",
        "--temp", "0.0",  # Complètement déterministe
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()

    # Nettoie les séparateurs mlx
    if "==========" in output:
        parts = output.split("==========")
        output = parts[1].strip() if len(parts) >= 2 else output

    # Nettoie les tokens spéciaux
    for token in ["<|eot_id|>", "<|end_of_text|>", "Prompt:", "Generation:", "Peak"]:
        if token in output:
            output = output.split(token)[0]

    # Prend seulement la première ligne non vide
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    return lines[0] if lines else ""


def score_exact(pred: str, ref: str) -> bool:
    return pred.strip().lower() == ref.strip().lower()


def score_partial(pred: str, ref: str) -> bool:
    ref_words = set(ref.lower().split())
    pred_words = set(pred.lower().split())
    if not ref_words:
        return False
    return len(ref_words & pred_words) / len(ref_words) >= 0.5


def run_benchmark():
    print("🌍 Dioula-AI — Benchmark d'évaluation")
    print("=" * 60)
    print(f"  Modèle  : {MODEL_PATH}")
    print(f"  Phrases : {len(BENCHMARK)}")
    print("=" * 60)
    print()

    results = []
    exact_ok = 0
    partial_ok = 0

    for i, item in enumerate(BENCHMARK, 1):
        fr = item["fr"]
        expected = item["dioula"]
        prediction = get_translation(fr)

        is_exact = score_exact(prediction, expected)
        is_partial = not is_exact and score_partial(prediction, expected)

        if is_exact:
            exact_ok += 1
            status = "✅ EXACT"
        elif is_partial:
            partial_ok += 1
            status = "🟡 PARTIEL"
        else:
            status = "❌ FAUX"

        print(f"[{i:02d}/{len(BENCHMARK)}] {fr}")
        print(f"  Attendu : {expected}")
        print(f"  Obtenu  : {prediction}")
        print(f"  {status}")
        print()

        results.append({
            "fr": fr,
            "expected": expected,
            "prediction": prediction,
            "exact": is_exact,
            "partial": is_partial,
        })

    total = len(BENCHMARK)
    pct_exact   = (exact_ok / total) * 100
    pct_global  = ((exact_ok + partial_ok) / total) * 100

    print("=" * 60)
    print("📊 RAPPORT FINAL")
    print("=" * 60)
    print(f"  Exact   : {exact_ok}/{total} ({pct_exact:.1f}%)")
    print(f"  Partiel : {partial_ok}/{total}")
    print(f"  Global  : {pct_global:.1f}%")
    print()

    if pct_exact >= 60:
        print("🎉 Excellent ! Modèle prêt pour la production.")
    elif pct_exact >= 35:
        print("👍 Bon résultat. Quelques itérations de plus amélioreraient le modèle.")
    else:
        print("⚠️  Le modèle a besoin de plus d'entraînement.")
        print("   → Relance avec --iters 5000 et --num-layers 24")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "score_exact_pct": pct_exact,
            "score_global_pct": pct_global,
            "exact": exact_ok,
            "partial": partial_ok,
            "total": total,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats → benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
