#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_aws.sh — Installation complète Dioula-AI sur AWS EC2
# Instance recommandée : p3.2xlarge (V100 16GB) ou p4d.24xlarge (A100 80GB)
# OS : Ubuntu 22.04 LTS
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=== Dioula-AI — Setup AWS ==="
echo "Instance : $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'inconnue')"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'non détecté')"
echo ""

# ── 1. Dépendances système ────────────────────────────────────────────────────
echo "[1/6] Installation des dépendances système..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git curl

# ── 2. Environnement Python ───────────────────────────────────────────────────
echo "[2/6] Création de l'environnement Python..."
python3 -m venv env
source env/bin/activate

pip install --upgrade pip -q

# ── 3. Dépendances Python ─────────────────────────────────────────────────────
echo "[3/6] Installation des dépendances Python (PyTorch + HuggingFace + TRL)..."
pip install -r requirements-aws.txt -q
echo "    OK"

# ── 4. Téléchargement du modèle Llama 3.1 70B ────────────────────────────────
echo "[4/6] Téléchargement de Llama 3.1 70B Instruct..."
echo "    Ce modèle fait ~140 GB. Assurez-vous d'avoir assez d'espace disque."
echo "    Connectez-vous d'abord : huggingface-cli login"
echo ""

mkdir -p model

if [ -z "$HF_TOKEN" ]; then
    echo "    ATTENTION : variable HF_TOKEN non définie."
    echo "    Exportez votre token : export HF_TOKEN=hf_xxxxxxxxxxxx"
    echo "    Puis relancez ce script depuis l'étape 4 :"
    echo "    huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \\"
    echo "      --local-dir ./model/llama-3.1-70b --token \$HF_TOKEN"
else
    huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
        --local-dir ./model/llama-3.1-70b \
        --token "$HF_TOKEN"
    echo "    Modèle téléchargé dans ./model/llama-3.1-70b"
fi

# ── 5. Préparation du dataset ─────────────────────────────────────────────────
echo "[5/6] Préparation du dataset (38 027 paires → 114 075 exemples)..."
python3 2_prepare_dataset.py
echo "    Dataset prêt."

# ── 6. Lancement du fine-tuning ───────────────────────────────────────────────
echo "[6/6] Lancement du fine-tuning QLoRA..."
echo "    Durée estimée : 8-15h sur p3.2xlarge (V100) | 4-6h sur p4d (A100)"
echo ""

# Vérification GPU avant de lancer
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERREUR : aucun GPU CUDA détecté. Vérifiez les drivers NVIDIA.')
    exit(1)
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU : {gpu} | VRAM : {vram:.1f} GB')
if vram < 30:
    print('ATTENTION : VRAM < 30 GB. Le fine-tuning 70B nécessite au moins 40 GB (QLoRA 4-bit).')
    print('Considérez p3.16xlarge (4x V100 = 64 GB) ou p4d.24xlarge (8x A100).')
"

bash 3_finetune.sh

echo ""
echo "=== Setup terminé ==="
echo "Lancez l'API avec : uvicorn main:app --host 0.0.0.0 --port 8000"
