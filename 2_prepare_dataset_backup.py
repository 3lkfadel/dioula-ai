"""
ÉTAPE 2 — Préparation du dataset pour le fine-tuning MLX
Lit le fichier dataset_raw.json local (pas HuggingFace)
"""

import json
import os
import random

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE    = os.path.join(OUTPUT_DIR, "dataset_raw.json")
TRAIN_FILE  = os.path.join(OUTPUT_DIR, "dataset_train.jsonl")
VALID_FILE  = os.path.join(OUTPUT_DIR, "dataset_valid.jsonl")
TEST_FILE   = os.path.join(OUTPUT_DIR, "dataset_test.jsonl")

TRAIN_RATIO = 0.90
VALID_RATIO = 0.05

SYSTEM_PROMPT = (
    "Tu es un assistant expert en langue Dioula (Jula), "
    "parlée en Côte d'Ivoire et au Burkina Faso. "
    "Tu peux traduire entre le français et le Dioula, "
    "et converser naturellement dans les deux langues."
)


def make_fr_to_dioula(fr, dioula):
    return {"text": (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Traduis en Dioula : {fr.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{dioula.strip()}<|eot_id|>"
    )}


def make_dioula_to_fr(fr, dioula):
    return {"text": (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Traduis en français : {dioula.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{fr.strip()}<|eot_id|>"
    )}


def make_conversation(fr, dioula):
    questions = [
        f"Comment dit-on en Dioula : « {fr.strip()} » ?",
        f"Quelle est la traduction Dioula de : {fr.strip()} ?",
        f"Dis-moi en Dioula : {fr.strip()}",
        f"En Dioula, comment exprime-t-on : {fr.strip()} ?",
    ]
    return {"text": (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{random.choice(questions)}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{dioula.strip()}<|eot_id|>"
    )}


def format_dataset():
    print(f"📥 Chargement du fichier local : {RAW_FILE}")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"✅ {len(raw_data)} exemples chargés")
    print(f"📋 Aperçu : FR='{raw_data[0]['source']}' | DY='{raw_data[0]['target']}'\n")

    all_examples = []
    print("🔄 Formatage des données...")

    for row in raw_data:
        fr     = str(row.get("source", "")).strip()
        dioula = str(row.get("target", "")).strip()

        if len(fr) < 2 or len(dioula) < 2:
            continue

        all_examples.append(make_fr_to_dioula(fr, dioula))
        all_examples.append(make_dioula_to_fr(fr, dioula))
        all_examples.append(make_conversation(fr, dioula))

    print(f"✅ {len(all_examples)} exemples générés (x3 par paire)\n")

    random.seed(42)
    random.shuffle(all_examples)

    n       = len(all_examples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_data = all_examples[:n_train]
    valid_data = all_examples[n_train:n_train + n_valid]
    test_data  = all_examples[n_train + n_valid:]

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  💾 {os.path.basename(path)} → {len(data)} exemples ({size_mb:.1f} MB)")

    print("📁 Sauvegarde JSONL :")
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(valid_data, VALID_FILE)
    save_jsonl(test_data,  TEST_FILE)

    print(f"\n🔍 Exemple d'entraînement :")
    print("-" * 60)
    print(train_data[0]["text"])
    print("-" * 60)
    print(f"\n✅ Dataset prêt !")
    print(f"   Train : {len(train_data)} | Valid : {len(valid_data)} | Test : {len(test_data)}")


if __name__ == "__main__":
    format_dataset()