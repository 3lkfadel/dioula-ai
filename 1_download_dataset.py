"""
ÉTAPE 1 — Téléchargement du dataset Dioula depuis HuggingFace
Dataset : Findora/hf_fr_dioula_full
"""

from datasets import load_dataset
import pandas as pd
import os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
RAW_FILE = os.path.join(OUTPUT_DIR, "dataset_raw.json")


def download_dataset():
    print("📥 Téléchargement du dataset Findora/hf_fr_dioula_full...")
    print("   (Cela peut prendre quelques minutes selon ta connexion)\n")

    dataset = load_dataset("Findora/hf_fr_dioula_full", split="train")

    print(f"✅ Dataset chargé : {len(dataset)} exemples")
    print(f"📋 Colonnes disponibles : {dataset.column_names}\n")

    # Affiche un aperçu des 5 premiers exemples
    print("🔍 Aperçu des données :")
    for i, example in enumerate(dataset.select(range(min(5, len(dataset))))):
        print(f"  [{i+1}] {example}")
    print()

    # Sauvegarde brute en JSON
    df = dataset.to_pandas()
    df.to_json(RAW_FILE, orient="records", force_ascii=False, indent=2)
    print(f"💾 Dataset brut sauvegardé → {RAW_FILE}")
    print(f"   Taille : {os.path.getsize(RAW_FILE) / 1024 / 1024:.1f} MB")

    return dataset


if __name__ == "__main__":
    download_dataset()
