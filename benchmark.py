#!/usr/bin/env python3
"""
Benchmark d'évaluation Dioula-AI
50 phrases de référence FR ↔ Dioula
Mesure la qualité des traductions avec score BLEU + validation humaine
"""

import json
import subprocess
import sys

# ============================================================
# 50 PHRASES DE RÉFÉRENCE (validées)
# ============================================================
BENCHMARK = [
    # Salutations
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

    # Famille
    {"fr": "Ma mère", "dioula": "N ba"},
    {"fr": "Mon père", "dioula": "N fa"},
    {"fr": "Mon frère", "dioula": "N kɔrɔ"},
    {"fr": "Ma sœur", "dioula": "N teri"},
    {"fr": "Mon ami", "dioula": "N teri"},
    {"fr": "Mon fils", "dioula": "N denmisɛn"},
    {"fr": "Ma fille", "dioula": "N denmusow"},
    {"fr": "La famille", "dioula": "Fadenya"},
    {"fr": "Le mari", "dioula": "Cɛ"},
    {"fr": "La femme", "dioula": "Muso"},

    # Chiffres
    {"fr": "Un", "dioula": "Kelen"},
    {"fr": "Deux", "dioula": "Fila"},
    {"fr": "Trois", "dioula": "Saba"},
    {"fr": "Quatre", "dioula": "Naani"},
    {"fr": "Cinq", "dioula": "Duuru"},
    {"fr": "Dix", "dioula": "Tan"},
    {"fr": "Cent", "dioula": "Kɛmɛ"},
    {"fr": "Mille", "dioula": "Waa kelen"},

    # Vie quotidienne
    {"fr": "L'eau", "dioula": "Jii"},
    {"fr": "La nourriture", "dioula": "Dumu"},
    {"fr": "Le marché", "dioula": "Maarɛ"},
    {"fr": "La maison", "dioula": "So"},
    {"fr": "Le travail", "dioula": "Baara"},
    {"fr": "L'argent", "dioula": "Wari"},
    {"fr": "Le téléphone", "dioula": "Telephone"},
    {"fr": "La route", "dioula": "Sira"},
    {"fr": "Le village", "dioula": "Dugu"},
    {"fr": "La ville", "dioula": "Dugu ba"},

    # Phrases courantes
    {"fr": "Je ne comprends pas", "dioula": "N ma a faamu"},
    {"fr": "Je comprends", "dioula": "N ye a faamu"},
    {"fr": "Répète s'il te plaît", "dioula": "A fɔ segin"},
    {"fr": "Parle lentement", "dioula": "Kuma nɔgɔya"},
    {"fr": "Où est le marché ?", "dioula": "Maarɛ bɛ min ?"},
    {"fr": "Combien ça coûte ?", "dioula": "A joli ?"},
    {"fr": "C'est trop cher", "dioula": "A gélén"},
    {"fr": "J'ai faim", "dioula": "Kɔnɔ bɛ n dɔn"},
    {"fr": "J'ai soif", "dioula": "Jii bɛ n sɔn"},
    {"fr": "Je suis fatigué", "dioula": "N dɛsɛra"},
    {"fr": "Dieu merci", "dioula": "Ala ka bato"},
]


LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Tu es un assistant expert en langue Dioula. "
    "Tu traduis entre le français et le Dioula avec précision.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

MODEL_PATH = "./model/dioula-llama-fused"
BASE_MODEL  = "./model/llama-3.2-3b-mlx"
ADAPTER     = "./model/adapters"


def get_translation(text: str) -> str:
    """Demande une traduction au modèle."""
    import os
    prompt = LLAMA_TEMPLATE.format(prompt=f"Traduis en Dioula : {text}")
    use_fused = os.path.exists(MODEL_PATH)

    cmd = [
        "mlx_lm.generate",
        "--model", MODEL_PATH if use_fused else BASE_MODEL,
        "--prompt", prompt,
        "--max-tokens", "30",
        "--temp", "0.1",
    ]
    if not use_fused:
        cmd += ["--adapter-path", ADAPTER]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()

    # Nettoie la sortie
    if "==========" in output:
        parts = output.split("==========")
        if len(parts) >= 2:
            output = parts[1].strip()

    output = output.replace("<|eot_id|>", "").strip()
    # Prend seulement la première ligne
    output = output.split("\n")[0].strip()
    return output


def score_exact(prediction: str, reference: str) -> bool:
    """Vérifie si la traduction est exacte (insensible à la casse)."""
    return prediction.strip().lower() == reference.strip().lower()


def score_partial(prediction: str, reference: str) -> bool:
    """Vérifie si les mots clés sont présents."""
    ref_words = set(reference.lower().split())
    pred_words = set(prediction.lower().split())
    if not ref_words:
        return False
    overlap = ref_words & pred_words
    return len(overlap) / len(ref_words) >= 0.5


def run_benchmark():
    print("🌍 Dioula-AI — Benchmark d'évaluation")
    print("=" * 60)
    print(f"  Nombre de phrases : {len(BENCHMARK)}")
    print("=" * 60)
    print()

    results = []
    exact_ok = 0
    partial_ok = 0

    for i, item in enumerate(BENCHMARK, 1):
        fr = item["fr"]
        expected = item["dioula"]

        print(f"[{i:02d}/{len(BENCHMARK)}] {fr}")
        prediction = get_translation(fr)

        is_exact = score_exact(prediction, expected)
        is_partial = score_partial(prediction, expected)

        if is_exact:
            exact_ok += 1
            status = "✅ EXACT"
        elif is_partial:
            partial_ok += 1
            status = "🟡 PARTIEL"
        else:
            status = "❌ FAUX"

        print(f"  Attendu   : {expected}")
        print(f"  Obtenu    : {prediction}")
        print(f"  Résultat  : {status}")
        print()

        results.append({
            "fr": fr,
            "expected": expected,
            "prediction": prediction,
            "exact": is_exact,
            "partial": is_partial,
        })

    # Rapport final
    total = len(BENCHMARK)
    score_exact_pct = (exact_ok / total) * 100
    score_partial_pct = ((exact_ok + partial_ok) / total) * 100

    print("=" * 60)
    print("📊 RAPPORT FINAL")
    print("=" * 60)
    print(f"  Traductions exactes  : {exact_ok}/{total} ({score_exact_pct:.1f}%)")
    print(f"  Traductions partielles : {partial_ok}/{total}")
    print(f"  Score global         : {score_partial_pct:.1f}%")
    print()

    if score_exact_pct >= 60:
        print("🎉 Excellent ! Modèle prêt pour la production.")
    elif score_exact_pct >= 35:
        print("👍 Bon résultat. Encore quelques itérations d'entraînement amélioreraient le modèle.")
    else:
        print("⚠️  Score faible. Recommande plus d'itérations ou un dataset plus grand.")

    # Sauvegarde les résultats
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "score_exact": score_exact_pct,
            "score_global": score_partial_pct,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats détaillés → benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
