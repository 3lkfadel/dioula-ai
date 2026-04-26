# Dioula-AI — Fine-tuning LLaMA 3.2 en Dioula

Modèle de traduction et de conversation **Français ↔ Dioula (Jula)**,
entraîné en local sur Mac M1 Pro avec MLX, déployable sur AWS via PyTorch.

Le moteur d'inférence détecte automatiquement la plateforme au démarrage :
- **Mac Apple Silicon** → backend MLX (rapide, optimisé Metal)
- **AWS / Linux** → backend PyTorch (GPU CUDA ou CPU)

---

## Stack technique

| Composant | Mac (dev) | AWS (prod) |
|---|---|---|
| Runtime ML | MLX + MLX-LM | PyTorch + Transformers |
| Modèle base | LLaMA 3.2 3B (converti MLX, 4-bit) | LLaMA 3.2 3B (float16, fusionné) |
| Fine-tuning | `mlx_lm.lora` (LoRA) | — (fait sur Mac) |
| API | FastAPI + Uvicorn | FastAPI + Uvicorn |
| Dataset | Findora/hf_fr_dioula_full (HuggingFace) | — |

---

## Structure du projet

```
dioula-ai/
├── 1_download_dataset.py     # Étape 1 — Télécharge le dataset HuggingFace
├── 2_prepare_dataset.py      # Étape 2 — Formate les données en JSONL
├── 3_finetune.sh             # Étape 3 — Lance le fine-tuning MLX (Mac)
├── 4_test_model.py           # Étape 4 — Teste le modèle manuellement
├── inference.py              # Moteur d'inférence (Mac + AWS)
├── main.py                   # API FastAPI
├── benchmark_aligned.py      # Benchmark sur paires réelles du dataset
├── requirements.txt          # Dépendances Mac (MLX)
├── requirements-aws.txt      # Dépendances AWS (PyTorch)
└── model/
    ├── llama-3.2-3b-mlx/        # Modèle base converti MLX (Mac uniquement)
    ├── adapters/                 # Poids LoRA entraînés
    └── dioula-llama-fused-dq/   # Modèle fusionné float16 (Mac + AWS)
```

> Les dossiers `model/` et les fichiers `.jsonl` ne sont pas dans le repo Git
> (trop lourds). Voir les sections Mac et AWS ci-dessous pour les obtenir.

---

## Partie 1 — Développement sur Mac (Apple Silicon)

### Prérequis

- Mac M1 / M2 / M3 avec au moins **16 GB de RAM unifiée**
- Python 3.10+
- Compte HuggingFace avec accès à `meta-llama/Llama-3.2-3B-Instruct`
  (demande d'accès gratuite sur https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### 1. Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Télécharger le modèle de base

```bash
# Connexion HuggingFace (token sur https://huggingface.co/settings/tokens)
huggingface-cli login

# Convertit LLaMA 3.2 3B en format MLX quantizé 4-bit (~1.8 GB)
mlx_lm.convert \
  --hf-path meta-llama/Llama-3.2-3B-Instruct \
  --mlx-path ./model/llama-3.2-3b-mlx \
  -q
```

### 3. Télécharger et préparer le dataset

```bash
# Télécharge le dataset Dioula depuis HuggingFace (~13 MB)
python3 1_download_dataset.py

# Génère les fichiers JSONL d'entraînement (train / valid / test)
# Chaque paire FR/Dioula produit 3 exemples (x3 augmentation)
python3 2_prepare_dataset.py
```

### 4. Fine-tuning (LoRA)

```bash
bash 3_finetune.sh
```

Ce script lance `mlx_lm.lora` avec les paramètres suivants :
- `--iters 3000` — itérations (augmenter à 10 000+ pour la production)
- `--num-layers 16` — couches LoRA
- `--learning-rate 1e-5`
- `--batch-size 2`
- `--save-every 500` — sauvegarde un checkpoint toutes les 500 itérations

Les poids LoRA sont sauvegardés dans `model/adapters/`.

### 5. Générer le modèle fusionné (pour AWS)

Après l'entraînement, fusionne les poids LoRA dans le modèle de base.
Cette étape génère un modèle **float16 standard** (~6.4 GB) lisible sur AWS.

```bash
mlx_lm.fuse \
  --model ./model/llama-3.2-3b-mlx \
  --adapter-path ./model/adapters \
  --save-path ./model/dioula-llama-fused-dq \
  --dequantize
```

> Important : utilise toujours `--dequantize`. Sans ce flag, la fusion sur un
> modèle quantizé (4-bit) produit un résultat corrompu.

### 6. Tester le modèle

```bash
# Test manuel avec quelques phrases
python3 4_test_model.py

# Benchmark sur 30 paires réelles du dataset
python3 benchmark_aligned.py
```

### 7. Lancer l'API en local

```bash
uvicorn main:app --reload --port 8000
```

L'API est disponible sur http://localhost:8000  
Documentation interactive : http://localhost:8000/docs

---

## Partie 2 — Déploiement sur AWS

### Prérequis AWS

- Instance **EC2** recommandée : `g4dn.xlarge` (GPU T4, 16 GB VRAM) ou `g5.xlarge` (GPU A10G)
- Pour tester sans GPU : `t3.large` (CPU seulement, inférence lente)
- OS : Ubuntu 22.04 LTS
- Python 3.10+

### Ce qu'il faut uploader

Le modèle fusionné (`model/dioula-llama-fused-dq/`, ~6.4 GB) est trop lourd
pour GitHub. Deux options pour le mettre à disposition sur AWS :

**Option A — HuggingFace Hub (recommandé)**

```bash
# Sur Mac, après le fine-tuning :
huggingface-cli login
huggingface-cli upload TON_USERNAME/dioula-llama-fused-dq \
  ./model/dioula-llama-fused-dq
```

Puis sur AWS :

```bash
huggingface-cli download TON_USERNAME/dioula-llama-fused-dq \
  --local-dir ./model/dioula-llama-fused-dq
```

**Option B — Amazon S3**

```bash
# Sur Mac :
aws s3 cp ./model/dioula-llama-fused-dq/ \
  s3://TON_BUCKET/dioula-llama-fused-dq/ --recursive

# Sur AWS EC2 :
aws s3 cp s3://TON_BUCKET/dioula-llama-fused-dq/ \
  ./model/dioula-llama-fused-dq/ --recursive
```

### Déploiement sur EC2

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/dioula-ai.git
cd dioula-ai

# 2. Environnement Python
python3 -m venv env
source env/bin/activate

# 3. Installer les dépendances AWS (PyTorch, pas MLX)
pip install -r requirements-aws.txt

# 4. Récupérer le modèle (voir options A ou B ci-dessus)
mkdir -p model
# ... télécharger dioula-llama-fused-dq/ dans model/

# 5. Lancer l'API
uvicorn main:app --host 0.0.0.0 --port 8000
```

> `inference.py` détecte automatiquement qu'il est sur Linux et charge
> le backend PyTorch. Aucune modification de code nécessaire.

### Ports à ouvrir dans le Security Group EC2

| Port | Protocole | Usage |
|---|---|---|
| 22 | TCP | SSH |
| 8000 | TCP | API Dioula-AI |

### Lancement en production (avec systemd)

Pour que l'API redémarre automatiquement après un reboot :

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
ExecStart=/home/ubuntu/dioula-ai/env/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable dioula-ai
sudo systemctl start dioula-ai
```

---

## Endpoints de l'API

### GET /
Vérifie que l'API est en ligne.

### GET /health
```json
{ "status": "healthy", "model": "llama-3.2-3b-dioula-finetuned" }
```

### POST /api/v1/chat
Conversation libre en Français ou Dioula.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Traduis en Dioula : bonjour"}'
```

```json
{
  "success": true,
  "session_id": "...",
  "reponse": "a bɛn",
  "langue_detectee": "français",
  "langue_reponse": "dioula",
  "mode": "traduction",
  "latence_ms": 340
}
```

### POST /api/v1/translate
Traduction directe avec direction explicite.

```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"texte": "au revoir", "direction": "fr_to_dioula"}'
```

---

## État du modèle

Le modèle est actuellement entraîné sur **3 000 itérations** (moins d'une époque
sur le dataset complet). Les traductions courtes fonctionnent partiellement.

Pour un niveau production :
- Relancer `3_finetune.sh` avec `--iters 10000` (voire 20 000)
- Re-générer le modèle fusionné avec `mlx_lm.fuse --dequantize`
- Re-uploader sur HuggingFace Hub ou S3

## Exigences matérielles

| | Mac (dev) | AWS (prod) |
|---|---|---|
| RAM | 16 GB unifiée | 16 GB VRAM (GPU) |
| Stockage | 30 GB | 15 GB |
| Puce | M1 Pro | T4 / A10G |
| Temps d'inférence | ~0.5s | ~0.3s (GPU) |
