# Dioula-AI — Fine-tuning Llama 3.1 70B en Dioula

Modèle de traduction et de conversation **Français ↔ Dioula (Jula)**,
fine-tuné avec QLoRA sur **38 027 paires** de phrases propres et dédupliquées.

- **Modèle de base** : Llama 3.1 70B Instruct (Meta)
- **Méthode** : QLoRA 4-bit (PEFT + bitsandbytes + TRL)
- **Dataset** : 38 027 paires FR↔Dioula → 114 075 exemples (x3 augmentation)
- **Fine-tuning** : AWS EC2 GPU (p3/p4d)
- **Inférence** : AWS EC2 ou API FastAPI

---

## Structure du projet

```
dioula-ai/
├── 1_download_dataset.py   # Télécharge le dataset brut HuggingFace
├── 2_prepare_dataset.py    # Formate dataset_clean.json → JSONL (train/valid/test)
├── 3_finetune.sh           # Fine-tuning QLoRA Llama 3.1 70B (AWS GPU)
├── 4_test_model.py         # Test manuel du modèle
├── build_dataset.py        # Construit dataset_clean.json (HF + connaissances)
├── inference.py            # Moteur d'inférence (PyTorch)
├── main.py                 # API FastAPI
├── benchmark_aligned.py    # Benchmark FR→Dioula
├── dataset_clean.json      # Dataset propre 38 027 paires (versionné)
├── setup_aws.sh            # Script de setup complet sur AWS
├── requirements.txt        # Dépendances Mac (MLX, dev local)
├── requirements-aws.txt    # Dépendances AWS (PyTorch + TRL + bitsandbytes)
└── model/                  # Modèles et adapteurs (non versionné — voir S3/HF)
    ├── llama-3.1-70b/          # Modèle de base (à télécharger sur AWS)
    └── adapters/               # Poids LoRA après fine-tuning
```

> `model/` et les fichiers `.jsonl` ne sont pas dans le repo (trop lourds).
> Le dataset propre `dataset_clean.json` (~7 MB) est versionné.

---

## Démarrage rapide — AWS

### Prérequis

- Instance EC2 avec GPU : **p3.2xlarge** (V100 16GB, min) ou **p4d.24xlarge** (A100 80GB, recommandé)
- Ubuntu 22.04 LTS
- Accès au modèle `meta-llama/Meta-Llama-3.1-70B-Instruct` sur HuggingFace
  (demande gratuite sur https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- Token HuggingFace : https://huggingface.co/settings/tokens

### 1. Cloner le repo et lancer le setup

```bash
git clone https://github.com/TON_USERNAME/dioula-ai.git
cd dioula-ai

export HF_TOKEN=hf_xxxxxxxxxxxx   # ton token HuggingFace
bash setup_aws.sh
```

Le script fait tout automatiquement :
1. Installe les dépendances système
2. Crée l'environnement Python
3. Installe PyTorch + HuggingFace + TRL + bitsandbytes
4. Télécharge Llama 3.1 70B (~140 GB)
5. Prépare le dataset (114 075 exemples)
6. Lance le fine-tuning QLoRA

### 2. Setup manuel (étape par étape)

```bash
# Environnement Python
python3 -m venv env && source env/bin/activate
pip install -r requirements-aws.txt

# Télécharger Llama 3.1 70B
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --local-dir ./model/llama-3.1-70b

# Préparer le dataset
python3 2_prepare_dataset.py

# Lancer le fine-tuning
bash 3_finetune.sh
```

### 3. Lancer l'API après fine-tuning

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Dataset

Le dataset `dataset_clean.json` est construit depuis deux sources :

| Source | Paires |
|---|---|
| OBY632/merged-bambara-dioula-dataset (HuggingFace) | 37 561 |
| Connaissances linguistiques structurées | 466 |
| **Total unique (après déduplication)** | **38 027** |

Couverture thématique : salutations, famille, santé, nourriture, animaux, couleurs,
nombres, temps, lieux, transports, commerce, école, émotions, religion, agriculture,
proverbes, expressions idiomatiques.

Pour reconstruire le dataset depuis zéro :

```bash
python3 build_dataset.py
```

---

## Fine-tuning — Paramètres QLoRA

| Paramètre | Valeur |
|---|---|
| Modèle de base | Llama 3.1 70B Instruct |
| Quantization | 4-bit (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Modules ciblés | q_proj, k_proj, v_proj, o_proj |
| Époques | 3 |
| Batch size | 4 (+ grad accumulation ×4) |
| Learning rate | 2e-4 (cosine scheduler) |
| Max seq length | 512 tokens |

---

## Endpoints de l'API

### GET /health
```json
{ "status": "healthy", "model": "llama-3.1-70b-dioula-finetuned" }
```

### POST /api/v1/translate
```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"texte": "bonjour", "direction": "fr_to_dioula"}'
```
```json
{ "success": true, "reponse": "i ni sogoma", "latence_ms": 340 }
```

### POST /api/v1/chat
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Traduis en Dioula : merci beaucoup"}'
```

---

## Instances AWS recommandées

| Instance | GPU | VRAM | Fine-tuning 70B | Coût/h |
|---|---|---|---|---|
| p3.2xlarge | V100 | 16 GB | Limite (batch=1) | ~$3 |
| p3.8xlarge | 4× V100 | 64 GB | OK | ~$12 |
| p4d.24xlarge | 8× A100 | 320 GB | Idéal | ~$32 |
| g5.12xlarge | 4× A10G | 96 GB | OK | ~$16 |

> Pour le fine-tuning QLoRA 70B, minimum recommandé : **40 GB VRAM** (p3.8xlarge ou g5.12xlarge).

### Ports Security Group EC2

| Port | Usage |
|---|---|
| 22 | SSH |
| 8000 | API Dioula-AI |

### Lancement en production (systemd)

```bash
sudo nano /etc/systemd/system/dioula-ai.service
```

```ini
[Unit]
Description=Dioula-AI API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/dioula-ai
EnvironmentFile=/home/ubuntu/dioula-ai/.env
ExecStart=/home/ubuntu/dioula-ai/env/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable dioula-ai
sudo systemctl start dioula-ai
```
