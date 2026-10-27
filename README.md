# Reflex

Tri des signalements de phishing et d'arnaques pour les hôpitaux et les mairies, en deux vitesses.

Un agent envoie ce qui lui paraît louche (photo d'écran, capture, mail, QR code, message vocal).
Il reçoit une réponse claire en quelques secondes, et le domaine malveillant est déjà bloqué côté IT.

Projet du Meta Global AI Developer Hackathon (26 oct. – 4 nov. 2026), track Indie, thème « Build for Your World ».

## Comment ça marche

```
signalement (Telegram) ──► Étage 0 : extraction locale, gratuite
                           texte (OCR), QR codes, URLs, en-têtes du mail, transcription du vocal
                                   │
                                   ▼
                           Étage 1 : Laya (petit modèle local, fine-tuné)
                           p(phishing) très basse ──► fermé
                           p(phishing) très haute ──► bloqué
                           entre les deux        ──► étage 2
                                   │
                                   ▼
                           Étage 2 : agent Meta (Muse Spark ; Muse Glimmer pour le sensible)
                           ouvre le lien dans un navigateur isolé, regarde la page,
                           SAM 3.1 entoure les preuves sur l'image
                                   │
                                   ▼
                           Actions : blocage DNS (Pi-hole), autres destinataires, alerte
```

Garde-fou : l'agent n'a pas le droit de conclure « légitime » sur un cas que Laya juge risqué sans preuve concrète.

## Ce qui existe aujourd'hui

| Fichier | Rôle |
|---|---|
| `reflex/extract.py` | Étage 0 : URLs (y compris `hxxp://exemple[.]com`), mails `.eml`, OCR, QR codes, conversion audio, signaux sur les domaines |
| `reflex/state.py` | Ce que Laya lit. **Même fonction à l'entraînement et à l'inférence** |
| `reflex/triage.py` | Étage 1 : Laya et les seuils fermer / bloquer / escalader |
| `reflex/meta.py` | Client Meta Model API : Muse Spark, SAM 3.1, Voice Transcribe ; Muse Glimmer via un fournisseur |
| `reflex/sam_format.py` | Lecture de la sortie de SAM 3.1 (boîtes) |
| `reflex/pipeline.py` | Chaîne d'un signalement : étage 0 puis Laya, indépendante de Telegram |
| `reflex/bot.py` | Bot Telegram (`reflex-bot`) : captures, photos, `.eml`, vocaux, textes et liens |
| `reflex/replies.py` | Réponses du bot (anglais ou français), liens rendus non cliquables |
| `scripts/smoke_meta.py` | Vérifie en une minute que ta clé donne accès à tout |
| `scripts/build_laya_dataset.py` | Fabrique les données d'entraînement de Laya |
| `scripts/eval_laya.py` | Mesure Laya et choisit les seuils de la cascade |
| `notebooks/finetune_laya_kaggle.ipynb` | Fine-tune de Laya sur GPU Kaggle gratuit |

## Installation (Mac)

```bash
brew install tesseract tesseract-lang ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[laya,data,dev,bot]"
cp .env.example .env        # puis colle ta clé MODEL_API_KEY
chmod 600 .env              # lisible par ton compte seulement
pytest                      # tout doit passer
```

## Ordre des opérations

1. **Accès Meta** : `python scripts/smoke_meta.py`
   Teste les modèles disponibles, Muse Spark sur une fausse page de connexion, SAM 3.1 (« QR code », « text field »…),
   et Voice Transcribe en français (audio généré avec `say`). Résultats dans `data/smoke/`.
2. **Données Laya** : `python scripts/build_laya_dataset.py --per-class 3000`
   puis `laya-train --data data/laya/train.jsonl --base multilingual --dry-run` pour vérifier le format.
   Options : `--translate-fr 300` (traductions françaises par Muse Spark) et `--synth-voice 200` (appels d'arnaque synthétiques).
3. **Fine-tune** : envoie `data/laya/` sur Kaggle dans un dataset **privé** `reflex-laya-data` et lance le notebook.
   Récupère `laya-reflex.zip`, dézippe dans `models/`, puis mets `LAYA_MODEL=./models/laya-reflex` dans `.env`.
4. **Pi-hole** : définis `PIHOLE_PASSWORD` dans `.env`, puis `docker compose up -d pihole`
   (interface sur http://localhost:8081/admin, accessible depuis ce Mac uniquement).
5. **Bot Telegram** : mets le token de @BotFather dans `TELEGRAM_BOT_TOKEN`, puis lance `reflex-bot`.
   Le bot ne répond qu'aux identifiants listés dans `TELEGRAM_ALLOWED_USERS` : envoie-lui `/start`,
   il te donne ton identifiant, ajoute-le, puis relance-le. Réponses en anglais par défaut (`BOT_LANG=fr` pour le français).

## Ce qu'on mesure

- la part des signalements réglés par Laya seul (sans appel au gros modèle) ;
- les attaques ratées (phishing fermé à tort) et les faux blocages ;
- la latence et le coût pour 1 000 signalements, comparés à un agent Meta seul.

## Données et licences

- [ealvaradob/phishing-dataset](https://huggingface.co/datasets/ealvaradob/phishing-dataset) — Apache-2.0.
  Compile quatre sources publiques : e-mails ([Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)),
  SMS ([Mendeley Data](https://data.mendeley.com/datasets/f45bkkt8pr/1)),
  URLs ([Kaggle](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls)),
  pages web ([Mendeley Data](https://data.mendeley.com/datasets/n96ncsr5g4/1)).
- [PhreshPhish](https://huggingface.co/datasets/phreshphish/phreshphish) (Dalton et al., 2025,
  [arXiv:2507.10854](https://arxiv.org/abs/2507.10854)) — CC BY 4.0, réservé à la recherche anti-phishing.
- [Laya](https://huggingface.co/convaiinnovations/laya) — Apache-2.0
- Aucune donnée réelle d'hôpital. Le mode Contributor de Muse Spark (moins cher, Meta peut entraîner dessus)
  ne sert qu'aux données publiques ou synthétiques.

## Sécurité et données personnelles

- Ces corpus publics contiennent des données personnelles réelles (e-mails d'entreprise, noms, numéros) :
  les fichiers générés restent dans `data/` (hors git), le dataset Kaggle reste privé, rien n'est republié.
- Les clés restent dans `.env`, jamais commité. Les captures et `.eml` de vrais signalements ne vont ni dans git,
  ni vers le mode Contributor.
- Pi-hole n'écoute que sur `127.0.0.1` et ne démarre pas sans `PIHOLE_PASSWORD`.
