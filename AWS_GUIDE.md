# Guide AWS — Dioula-AI (Llama 3.1 70B)

Ce guide explique pas à pas comment déployer le fine-tuning et l'API Dioula-AI
sur AWS, depuis la création de l'instance jusqu'au lancement de l'API en production.

---

## Sommaire

1. [Prérequis](#1-prérequis)
2. [Créer l'instance EC2](#2-créer-linstance-ec2)
3. [Se connecter en SSH](#3-se-connecter-en-ssh)
4. [Installer les dépendances](#4-installer-les-dépendances)
5. [Cloner le repo GitHub](#5-cloner-le-repo-github)
6. [Configurer le token HuggingFace](#6-configurer-le-token-huggingface)
7. [Télécharger Llama 3.1 70B](#7-télécharger-llama-31-70b)
8. [Préparer le dataset](#8-préparer-le-dataset)
9. [Lancer le fine-tuning](#9-lancer-le-fine-tuning)
10. [Sauvegarder les adapteurs sur S3](#10-sauvegarder-les-adapteurs-sur-s3)
11. [Lancer l'API](#11-lancer-lapi)
12. [Mettre l'API en production (systemd)](#12-mettre-lapi-en-production-systemd)
13. [Tester l'API](#13-tester-lapi)
14. [Dépannage](#14-dépannage)

---

## 1. Prérequis

Avant de commencer, tu as besoin de :

- Un **compte AWS** avec droits EC2 : https://aws.amazon.com
- Un **compte HuggingFace** avec accès à Llama 3.1 70B :
  - Créer un compte : https://huggingface.co/join
  - Demander l'accès (gratuit, ~1h) : https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
  - Créer un token : https://huggingface.co/settings/tokens → "New token" → Read
- Le **repo GitHub** du projet (on le clone sur l'instance)

---

## 2. Créer l'instance EC2

### Choisir l'instance

Pour le fine-tuning de Llama 3.1 70B en QLoRA 4-bit, tu as besoin d'au moins **40 GB de VRAM GPU**.

| Instance | GPU | VRAM | Utilisation | Prix/heure |
|---|---|---|---|---|
| `p3.8xlarge` | 4× V100 | 64 GB | Fine-tuning (correct) | ~$12 |
| `g5.12xlarge` | 4× A10G | 96 GB | Fine-tuning (bien) | ~$16 |
| `p4d.24xlarge` | 8× A100 | 320 GB | Fine-tuning (idéal) | ~$32 |

> Pour juste faire tourner l'API (inférence sans fine-tuning), une `g4dn.2xlarge` (T4, 16 GB) suffit.

### Étapes dans la console AWS

1. Va sur https://console.aws.amazon.com/ec2
2. Clique **"Launch Instance"**
3. Remplis les champs :
   - **Name** : `dioula-ai-finetune`
   - **AMI** : cherche `Deep Learning AMI GPU PyTorch 2.3 (Ubuntu 22.04)` — c'est une image AWS qui a déjà CUDA, PyTorch et les drivers GPU installés
   - **Instance type** : `g5.12xlarge` (recommandé)
   - **Key pair** : crée une nouvelle clé ou utilise une existante → télécharge le fichier `.pem`
   - **Storage** : mets **500 GB** (le modèle 70B fait ~140 GB, les données ~60 GB)
4. Dans **"Network settings"**, clique "Edit" et ajoute une règle :
   - Type : Custom TCP | Port : `8000` | Source : `0.0.0.0/0`
5. Clique **"Launch Instance"**

---

## 3. Se connecter en SSH

Récupère l'**IP publique** de ton instance dans la console EC2 (colonne "Public IPv4 address").

```bash
# Sur ton Mac, depuis le dossier où est ton fichier .pem
chmod 400 ta-cle.pem

ssh -i ta-cle.pem ubuntu@TON_IP_PUBLIQUE
```

Exemple :
```bash
ssh -i dioula-key.pem ubuntu@54.123.456.789
```

> Si ça te demande "Are you sure you want to continue connecting?" → tape `yes`

---

## 4. Installer les dépendances

Une fois connecté en SSH sur l'instance :

```bash
# Mise à jour du système
sudo apt-get update && sudo apt-get upgrade -y

# Vérifie que le GPU est bien détecté
nvidia-smi
```

Tu dois voir quelque chose comme :
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.x   Driver Version: 535.x   CUDA Version: 12.2             |
| GPU  Name        Persistence-M| Bus-Id  Disp.A | Volatile Uncorr. ECC     |
| 0    NVIDIA A10G   Off          | ...            |                         |
+-----------------------------------------------------------------------------+
```

Si `nvidia-smi` ne fonctionne pas → l'instance n'a pas de GPU, vérifie le type d'instance.

```bash
# Installation des outils de base
sudo apt-get install -y git python3-pip python3-venv tmux htop
```

---

## 5. Cloner le repo GitHub

```bash
# Clone le projet
git clone https://github.com/TON_USERNAME/dioula-ai.git
cd dioula-ai

# Crée l'environnement Python
python3 -m venv env
source env/bin/activate

# Installe les dépendances AWS
pip install --upgrade pip
pip install -r requirements-aws.txt
```

> L'installation prend 5-10 minutes (PyTorch, bitsandbytes, etc.)

---

## 6. Configurer le token HuggingFace

```bash
# Méthode 1 — Variable d'environnement (recommandé)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Méthode 2 — Login interactif
huggingface-cli login
# Il te demande le token, colle-le et appuie sur Entrée
```

Pour ne pas avoir à refaire ça à chaque session, ajoute le token dans `.bashrc` :
```bash
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

---

## 7. Télécharger Llama 3.1 70B

> **Le modèle fait ~140 GB. Le téléchargement prend 30-60 minutes selon la bande passante AWS.**
> Utilise `tmux` pour que ça continue même si ta connexion SSH coupe.

```bash
# Lance une session tmux pour ne pas perdre le téléchargement si SSH coupe
tmux new -s download

# Dans la session tmux :
source env/bin/activate
mkdir -p model

huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --local-dir ./model/llama-3.1-70b \
  --token $HF_TOKEN

# Pour détacher tmux sans couper : Ctrl+B puis D
# Pour y revenir plus tard : tmux attach -t download
```

Vérifie que le téléchargement est complet :
```bash
ls -lh model/llama-3.1-70b/
# Tu dois voir des fichiers .safetensors qui font chacun ~4-5 GB
du -sh model/llama-3.1-70b/
# Doit afficher ~140G
```

---

## 8. Préparer le dataset

```bash
# Le dataset_clean.json est déjà dans le repo (38 027 paires propres)
# Cette commande génère les fichiers JSONL d'entraînement
python3 2_prepare_dataset.py
```

Tu dois voir :
```
📥 Chargement du fichier local : dataset_clean.json
✅ 38027 exemples chargés
✅ 114075 exemples générés (x3 par paire)
💾 train.jsonl → 102667 exemples (51.4 MB)
💾 valid.jsonl → 5703 exemples (2.9 MB)
💾 test.jsonl → 5705 exemples (2.9 MB)
✅ Dataset prêt !
```

---

## 9. Lancer le fine-tuning

> **Le fine-tuning dure plusieurs heures. Lance-le dans tmux.**

```bash
# Nouvelle session tmux pour le fine-tuning
tmux new -s finetune

# Dans la session tmux :
source env/bin/activate

# Lance le fine-tuning
bash 3_finetune.sh
```

Pour **suivre la progression** sans couper le fine-tuning :
```bash
# Dans un autre terminal SSH
tmux attach -t finetune
# Pour détacher sans couper : Ctrl+B puis D
```

### Ce que tu vas voir pendant le fine-tuning

```
Fine-tuning Dioula-AI — Llama 3.1 70B — dataset propre 38 027 paires
trainable params: 167,772,160 || all params: 70,722,772,992 || trainable%: 0.2372
{'loss': 1.8432, 'learning_rate': 0.0002, 'epoch': 0.01}
{'loss': 1.6210, 'learning_rate': 0.00019, 'epoch': 0.05}
...
```

La loss doit **descendre progressivement** (bonne signe). Si elle reste stable ou monte → problème.

### Durée estimée

| Instance | Durée estimée | Coût estimé |
|---|---|---|
| g5.12xlarge (4× A10G) | ~6-8h | ~$100-130 |
| p3.8xlarge (4× V100) | ~10-14h | ~$120-170 |
| p4d.24xlarge (8× A100) | ~3-4h | ~$100-130 |

> Astuce : utilise des **Spot Instances** pour économiser 60-70% sur le coût.

---

## 10. Sauvegarder les adapteurs sur S3

Une fois le fine-tuning terminé, les poids LoRA sont dans `model/adapters/final/`.
Sauvegarde-les sur S3 pour ne pas les perdre si l'instance s'arrête.

```bash
# Crée un bucket S3 (une seule fois)
aws s3 mb s3://dioula-ai-models --region eu-west-1

# Upload les adapteurs
aws s3 cp model/adapters/final/ s3://dioula-ai-models/adapters/final/ --recursive

# Vérifie l'upload
aws s3 ls s3://dioula-ai-models/adapters/final/
```

Pour récupérer les adapteurs plus tard (sur une nouvelle instance) :
```bash
mkdir -p model/adapters/final
aws s3 cp s3://dioula-ai-models/adapters/final/ model/adapters/final/ --recursive
```

---

## 11. Lancer l'API

```bash
source env/bin/activate

# Lancement simple (test)
uvicorn main:app --host 0.0.0.0 --port 8000
```

L'API est accessible depuis ton navigateur ou en ligne de commande :
```
http://TON_IP_PUBLIQUE:8000
http://TON_IP_PUBLIQUE:8000/docs   ← interface interactive
```

---

## 12. Mettre l'API en production (systemd)

Pour que l'API **redémarre automatiquement** si l'instance reboot :

```bash
# Crée le fichier de service
sudo nano /etc/systemd/system/dioula-ai.service
```

Colle ce contenu (remplace `/home/ubuntu/dioula-ai` par ton chemin si différent) :

```ini
[Unit]
Description=Dioula-AI API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/dioula-ai
Environment="HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx"
ExecStart=/home/ubuntu/dioula-ai/env/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Active et démarre le service
sudo systemctl daemon-reload
sudo systemctl enable dioula-ai
sudo systemctl start dioula-ai

# Vérifie que ça tourne
sudo systemctl status dioula-ai

# Voir les logs en temps réel
sudo journalctl -u dioula-ai -f
```

---

## 13. Tester l'API

Depuis ton Mac (ou n'importe où), remplace `TON_IP_PUBLIQUE` par l'IP de ton instance :

### Test basique
```bash
curl http://TON_IP_PUBLIQUE:8000/
```
Réponse attendue :
```json
{"service": "Dioula-AI", "status": "running", "langues": ["dioula", "français"]}
```

### Vérifier la santé
```bash
curl http://TON_IP_PUBLIQUE:8000/health
```

### Traduire du français en Dioula
```bash
curl -X POST http://TON_IP_PUBLIQUE:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"texte": "bonjour comment vas-tu", "direction": "fr_to_dioula"}'
```
Réponse attendue :
```json
{
  "success": true,
  "texte_original": "bonjour comment vas-tu",
  "traduction": "i ni sogoma, i ka kɛnɛ wa ?",
  "direction": "fr_to_dioula",
  "langue_source": "français",
  "langue_cible": "dioula",
  "latence_ms": 850
}
```

### Traduire du Dioula en français
```bash
curl -X POST http://TON_IP_PUBLIQUE:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"texte": "i ni sogoma", "direction": "dioula_to_fr"}'
```

### Chat libre
```bash
curl -X POST http://TON_IP_PUBLIQUE:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Comment dit-on merci beaucoup en Dioula ?"}'
```

### Interface web interactive
Ouvre dans ton navigateur :
```
http://TON_IP_PUBLIQUE:8000/docs
```
Tu peux tester tous les endpoints directement depuis l'interface.

---

## 14. Dépannage

### `nvidia-smi` ne fonctionne pas
```bash
# Vérifie le type d'instance (doit avoir GPU)
curl http://169.254.169.254/latest/meta-data/instance-type

# Réinstalle les drivers NVIDIA si nécessaire
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Erreur `CUDA out of memory` pendant le fine-tuning
Le modèle ne rentre pas en mémoire. Solutions :
```bash
# Option 1 — Réduit le batch size dans 3_finetune.sh
# Cherche per_device_train_batch_size=4 et mets 2 ou 1

# Option 2 — Réduit la longueur des séquences
# Cherche max_seq_length=512 et mets 256
```

### Le téléchargement HuggingFace s'interrompt
```bash
# Relance simplement la même commande, elle reprend où elle s'est arrêtée
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --local-dir ./model/llama-3.1-70b \
  --token $HF_TOKEN
```

### L'API ne répond pas
```bash
# Vérifie que le port 8000 est ouvert dans le Security Group EC2
# Console AWS → EC2 → Security Groups → ton groupe → Inbound rules
# Doit avoir : TCP 8000 0.0.0.0/0

# Vérifie que l'API tourne
sudo systemctl status dioula-ai
sudo journalctl -u dioula-ai -n 50
```

### Erreur `Permission denied` sur le fichier .pem
```bash
chmod 400 ta-cle.pem
```

### Libérer de l'espace disque
```bash
# Voir l'espace utilisé
df -h
du -sh model/*

# Vider le cache HuggingFace si besoin
rm -rf ~/.cache/huggingface/
```

---

## Récapitulatif des commandes essentielles

```bash
# Connexion SSH
ssh -i ta-cle.pem ubuntu@TON_IP_PUBLIQUE

# Activer l'environnement Python
source env/bin/activate

# Télécharger le modèle
huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
  --local-dir ./model/llama-3.1-70b --token $HF_TOKEN

# Préparer les données
python3 2_prepare_dataset.py

# Lancer le fine-tuning (dans tmux)
tmux new -s finetune
bash 3_finetune.sh

# Sauvegarder sur S3
aws s3 cp model/adapters/final/ s3://dioula-ai-models/adapters/final/ --recursive

# Lancer l'API
uvicorn main:app --host 0.0.0.0 --port 8000

# Voir les logs de l'API
sudo journalctl -u dioula-ai -f
```
