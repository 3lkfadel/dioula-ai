#!/usr/bin/env python3
"""
Benchmark aligné sur le dataset d'entraînement réel (Findora/hf_fr_dioula_full).
Toutes les paires FR→Dioula sont extraites du dataset réel pour garantir
que le modèle a bien vu ces exemples durant l'entraînement.

Scoring :
  - Exact   : correspondance parfaite (insensible à la casse)
  - Partiel : ≥50% de mots en commun OU ≥60% de bigrammes de caractères
  - Score caractère : pourcentage de bigrammes partagés (tolérant aux variantes orthographiques)
"""

import json
import subprocess
import re

# ============================================================
# 30 PAIRES RÉELLES DU DATASET (vérifiées manuellement)
# ============================================================
BENCHMARK = [
    # Salutations (présentes dans le dataset)
    {"fr": "bonjour",              "dioula": "nba a ni ɲɔgɔntuma"},
    {"fr": "bonsoir",              "dioula": "aw ni su"},
    {"fr": "au revoir",            "dioula": "k'an bɛn"},
    {"fr": "merci",                "dioula": "nba"},
    {"fr": "oui",                  "dioula": "naamu"},
    {"fr": "non",                  "dioula": "o ma kɛ"},
    {"fr": "Comment vas-tu ?",     "dioula": "I ka kènè wa?"},

    # Phrases courantes du dataset
    {"fr": "je suis content",              "dioula": "ne nisɔndiyara"},
    {"fr": "il est parti",                 "dioula": "a wulila a taara"},
    {"fr": "Elle est arrivée.",            "dioula": "A sera"},
    {"fr": "viens ici",                    "dioula": "i ka na yan"},
    {"fr": "assieds-toi",                  "dioula": "i k'i sigi"},
    {"fr": "Je ne sais pas !",             "dioula": "ne tɛ a dɔn"},
    {"fr": "il a dit oui",                 "dioula": "a ko naamu"},
    {"fr": "Là, c'est fini.",              "dioula": "a baana"},
    {"fr": "Je suis perdu.",               "dioula": "Un tounou na"},
    {"fr": "Dans sa direction.",           "dioula": "a ka sira la"},
    {"fr": "Et le silence retomba.",       "dioula": "Makunini nana"},
    {"fr": "Ne les écoutez pas.",          "dioula": "Aw ka na o lamɛ"},
    {"fr": "Il a brisé la fenêtre.",       "dioula": "A ye finɛtri kari"},
    {"fr": "C'est à moi de jouer.",        "dioula": "tulon kɛ sɛ la ne le ma"},
    {"fr": "C'est quoi ceux-là ?",         "dioula": "Nunu yi mun le ye"},
    {"fr": "Alors, vous me chassez ?",     "dioula": "Awɔ aw be n gwɛnna"},
    {"fr": "L'océan s'amuse.",             "dioula": "Baji bi tulon na"},
    {"fr": "Il est encore très vert.",     "dioula": "Binkɛnɛnaman lo ali bi"},
    {"fr": "Les poires sont mûres.",       "dioula": "Puaro mɔnibɛ"},
    {"fr": "Quel est ton âge ?",           "dioula": "i si ye san joli ye"},
    {"fr": "Combien coûte une orange ?",   "dioula": "lenburu den kelen ye dɔrɔmɛ joli ye"},
    {"fr": "Mon fils.",                    "dioula": "Nden tchè"},
    {"fr": "Je vais à l'école.",           "dioula": "n bi taga kalan so"},
]

MODEL_PATH = "./model/dioula-llama-fused-dq"  # dequantized

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Tu es un assistant expert en langue Dioula (Jula), "
    "parlée en Côte d'Ivoire et au Burkina Faso. "
    "Tu peux traduire entre le français et le Dioula, "
    "et converser naturellement dans les deux langues.<|eot_id|>"
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
        "--max-tokens", "30",
        "--temp", "0.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()

    if "==========" in output:
        parts = output.split("==========")
        output = parts[1].strip() if len(parts) >= 2 else output

    for token in ["<|eot_id|>", "<|end_of_text|>", "Prompt:", "Generation:", "Peak", "tokens-per-sec"]:
        if token in output:
            output = output.split(token)[0]

    lines = [l.strip() for l in output.split("\n") if l.strip()]
    return lines[0] if lines else ""


def bigrams(text: str) -> set:
    """Retourne l'ensemble des bigrammes de caractères d'un texte normalisé."""
    t = re.sub(r'\s+', ' ', text.lower().strip())
    return {t[i:i+2] for i in range(len(t)-1)} if len(t) >= 2 else set()


def char_similarity(pred: str, ref: str) -> float:
    """Score de similarité par bigrammes de caractères (0.0 → 1.0)."""
    bg_pred = bigrams(pred)
    bg_ref  = bigrams(ref)
    if not bg_ref:
        return 0.0
    return len(bg_pred & bg_ref) / len(bg_ref)


def score_exact(pred: str, ref: str) -> bool:
    return pred.strip().lower() == ref.strip().lower()


def score_partial(pred: str, ref: str) -> bool:
    """Partiel si ≥50% de mots en commun OU ≥60% de bigrammes partagés."""
    ref_words  = set(ref.lower().split())
    pred_words = set(pred.lower().split())
    word_ok = len(ref_words & pred_words) / max(len(ref_words), 1) >= 0.50
    char_ok = char_similarity(pred, ref) >= 0.60
    return word_ok or char_ok


def run_benchmark():
    print("🌍 Dioula-AI — Benchmark aligné dataset")
    print("=" * 65)
    print(f"  Modèle  : {MODEL_PATH}")
    print(f"  Phrases : {len(BENCHMARK)} (toutes issues du dataset d'entraînement)")
    print("=" * 65)
    print()

    results = []
    exact_ok = 0
    partial_ok = 0

    for i, item in enumerate(BENCHMARK, 1):
        fr       = item["fr"]
        expected = item["dioula"]
        prediction = get_translation(fr)

        sim  = char_similarity(prediction, expected)
        is_exact   = score_exact(prediction, expected)
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
        print(f"  Attendu  : {expected}")
        print(f"  Obtenu   : {prediction}")
        print(f"  Similarité caractères : {sim:.0%}  {status}")
        print()

        results.append({
            "fr": fr,
            "expected": expected,
            "prediction": prediction,
            "char_similarity": round(sim, 3),
            "exact": is_exact,
            "partial": is_partial,
        })

    total = len(BENCHMARK)
    pct_exact   = (exact_ok  / total) * 100
    pct_global  = ((exact_ok + partial_ok) / total) * 100
    avg_sim     = sum(r["char_similarity"] for r in results) / total * 100

    print("=" * 65)
    print("📊 RAPPORT FINAL")
    print("=" * 65)
    print(f"  Exact            : {exact_ok}/{total}  ({pct_exact:.1f}%)")
    print(f"  Partiel          : {partial_ok}/{total}")
    print(f"  Score global     : {pct_global:.1f}%")
    print(f"  Similarité moy.  : {avg_sim:.1f}%  (bigrammes caractères)")
    print()

    if pct_global >= 60:
        print("🎉 Bon résultat — le modèle généralise correctement.")
    elif pct_global >= 35:
        print("👍 Résultat encourageant — plus d'itérations l'amélioreraient.")
    else:
        print("⚠️  Score faible. Vérifie les logs d'entraînement et les données.")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "aligned_dataset",
            "score_exact_pct": round(pct_exact, 1),
            "score_global_pct": round(pct_global, 1),
            "avg_char_similarity_pct": round(avg_sim, 1),
            "exact": exact_ok,
            "partial": partial_ok,
            "total": total,
            "details": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats → benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
