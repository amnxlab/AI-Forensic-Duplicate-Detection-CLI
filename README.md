# AI‑Enhanced Forensic Duplicate Detection CLI (Dupli‑HQ)

A reproducible, modular CLI that bridges classical hashing and deep‑learning embeddings for file‑level forensic analysis and near‑duplicate detection.

Dupli‑HQ unifies deterministic integrity checking (MD5/SHA‑1/SHA‑256) with AI‑based perceptual and semantic similarity. It operates as a cross‑platform command‑line interface designed for digital forensics, data integrity management and research on hybrid edge‑AI pipelines. The system loads all models dynamically, logs every action, and writes versioned reports for reproducibility. It never trains models—every neural network runs in inference mode.

## ✨ Feature highlights

- **Multi‑layer duplicate detection.** Exact duplicates are caught with cryptographic hashes; perceptual near‑duplicates are detected with 64‑bit pHash and deep embeddings for images and text/code.
- **Hybrid AI integration.** Integrates models such as CLIP, DINOv2, ResNet, EfficientNet, SBERT‑Deep and CodeBERT. Each model plays a specific role (see below) and all run inference‑only.
- **Interactive and scripted CLI.** Offers an interactive shell (cli_shell.py) and a one‑shot interface (commands.py) with modes for snapshots, duplicate scans, folder tracking and file comparison.
- **Automation modules.** Includes a daily snapshot generator, a duplicate scanner and a live folder tracker that produces JSON and text reports.
- **Modular loader.** Finds the project root and lazily imports all components via loader.py, so there are no hard‑coded paths and new models can be plugged in easily.
- **Reproducible outputs.** Every run stores timestamped reports under reports/ and logs commands for auditing.

## 🧱 Architecture overview

The repository is organised into three major components:

- **CLI Interface** (cli_shell.py and commands.py) – provides an interactive shell and a one‑shot argument parser for automation. Both interfaces call automation modules and ensure that AI models are available via boot diagnostics.
- **Automation modules** – daily_snapshot.py, scan_duplicates.py and folder_tracker.py. These scripts perform snapshotting, duplicate scanning and continuous monitoring respectively. They read and write JSON under reports/ and support differential analysis.
- **AI and hashing back‑end** – housed under src/ai_model/ and src/cli_tool/hashing/. Hash utilities compute MD5/SHA hashes; perceptual_hash.py implements a visual 64‑bit pHash for images. Each AI model is wrapped in its own file with load_model() and extract_features() functions; these are loaded dynamically via loader.py.

### Directory structure (simplified)

```
src/
├── ai_model/            # Deep models (CLIP, DINOv2, ResNet, EfficientNet, SBERT‑Deep, CodeBERT, …)
├── cli_tool/
│   ├── automation/      # daily_snapshot.py, scan_duplicates.py, folder_tracker.py
│   ├── hashing/         # hash_utils.py, perceptual_hash.py
│   └── interface/       # cli_shell.py, commands.py, logger.py
└── loader.py            # project root finder and dynamic importer
reports/
├── snapshots/           # per‑folder AIHASH snapshots
├── diffs/               # snapshot difference reports
├── scan/                # duplicate scan reports
└── tracker_alerts.txt   # live tracker log
```

## 🧠 Deep‑learning models & roles

Dupli‑HQ integrates several pretrained neural networks, each selected for its strengths. All models run in evaluation mode and output fixed‑length embeddings. These vectors are compared with cosine similarity and never trained or updated at runtime.

| Model | Network (type) | Primary role & usage | Why this model? |
|-------|----------------|----------------------|-----------------|
| CLIP | ViT‑B/32 Vision Transformer (multimodal) | Generates cross‑modal image embeddings during snapshot and comparison modes. | Robust to diverse visual styles and aligns image and text spaces. |
| DINOv2 | ViT‑Base Vision Transformer (patch14, reg4) | Used in duplicate scans: after a pHash pre‑filter, DINOv2 provides high‑fidelity visual features for perceptual similarity. | Self‑supervised training yields robustness to small transforms and lighting changes. |
| ResNet‑50 | ResNet‑50 convolutional neural network | Complementary image embeddings combined with DINOv2; the maximum cosine similarity across these vectors decides near duplicates. | A proven baseline; its coarse‑to‑fine features help recover matches missed by transformers. |
| EfficientNet | EfficientNet‑B1/B3 convolutional networks | Optional lightweight extractors available for experiments and fall‑back; not used in the default duplicate pipeline. | Provides a high‑accuracy yet efficient alternative when resource budgets are tight. |
| SBERT‑Deep | all‑mpnet‑base‑v2 Transformer for text | Produces 768‑dimensional semantic embeddings for text files (snapshot) and for text/code duplicate scans. | Captures fine‑grained semantic similarity across documents and code comments. |
| CodeBERT | codebert‑base Transformer for code | Encodes structural and functional semantics of source code. Concatenated with SBERT‑Deep vectors to detect near duplicates in code. | Models syntax‑aware features that SBERT alone cannot capture. |

### Why these networks?

The chosen networks balance accuracy and efficiency. DINOv2 and ResNet‑50 provide complementary vision embeddings; EfficientNet is kept as a lightweight backup. CLIP enables cross‑modal robustness and uses a vision transformer tuned for image–text alignment. SBERT‑Deep and CodeBERT capture semantics and code structure respectively. Alternatives like generic CNNs or shallow embeddings were avoided because they either lack robustness to visual transformations or cannot model code structure.

## 📦 File‑type routing & similarity pipelines

Dupli‑HQ routes files to specialised pipelines based on their type and combines deterministic and neural methods. All comparisons use cosine similarity on L2‑normalised embeddings, and threshold values are configurable.

### Image files

1. **Exact match** – Compute SHA‑256; identical hashes signal an exact duplicate.
2. **Perceptual pre‑filter** – Use a 64‑bit pHash; if the Hamming distance is below a threshold, proceed to deep comparison.
3. **Deep check** – Extract embeddings with DINOv2 and ResNet‑50; the maximum cosine similarity between these vectors determines a near duplicate.
4. **Snapshot embedding** – For snapshot mode, CLIP produces an image embedding stored alongside the SHA‑256 hash for later semantic drift detection.

### Text and code files

1. **Exact match** – Compare SHA‑256 for perfect duplicates.
2. **Hybrid embedding** – Extract SBERT‑Deep embeddings; for code files, also extract CodeBERT and concatenate vectors. The concatenated representation reduces false positives by combining semantics and structure.
3. **Decision** – Compute cosine similarity on the (possibly concatenated) embeddings; a configurable threshold (default 0.75) marks near duplicates.

### Binaries and archives

Binaries and archives are compared only by SHA‑256 hashes; no embedding is computed.

## 🛠 Modes of operation

| Mode | Description |
|------|-------------|
| snapshot | Creates a snapshot of a folder, storing for each file its SHA‑256 and (if supported) an AI embedding; stored under reports/snapshots/. Future snapshots are compared to detect additions, deletions or semantic drift. |
| duplicates | Scans a folder for exact and near duplicates using the pipelines described above. Results are written as JSON in reports/scan/. |
| tracker | Runs a daemon that periodically snapshots folders, detects changes, optionally triggers duplicate scans and logs alerts in reports/tracker_alerts.txt. The default interval is 30 seconds. |
| compare | One‑off comparison between two files. When invoked with --auto, automatically selects the appropriate model based on file type. |

## 🧪 Example usage

```bash
# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Snapshot a folder
python src/cli_tool/interface/commands.py --mode snapshot --folder ./data

# Find duplicates in a folder
python src/cli_tool/interface/commands.py --mode duplicates --folder ./data

# Compare two files automatically
python src/cli_tool/interface/commands.py --mode compare --file1 img1.png --file2 img2.png --auto

# Launch interactive shell
python src/cli_tool/interface/cli_shell.py
dupli-hq> set path ./dataset
dupli-hq> set mode duplicates
dupli-hq> run
```

## 🔧 Extending the tool

To add a new AI model, create a module under `src/ai_model/` with the following functions:

```python
def load_model():
    """Load and return the model (and tokenizer if needed); call model.eval()."""

def get_transform():
    """(Optional) Return any preprocessing transforms for images/text."""

# For image models
def extract_features(path, model, transform):
    """Return a 1‑D NumPy embedding for an image file."""

# For text/code models
def extract_features_from_file(path, model_or_tokenizer, maybe_model=None):
    """Return a 1‑D NumPy embedding for text or code."""
```

Register the module in `loader.module_tree` and, if appropriate, integrate it into `scan_duplicates.py` or `daily_snapshot.py`. Maintain deterministic outputs—vectors must be 1‑D NumPy arrays and properly normalised.

## 📔 Reproducibility & quality notes

- All neural models run with fixed weights and deterministic preprocessing; there is no training at runtime.
- Cosine similarity is used for all embedding comparisons; ensure vectors are L2‑normalised when necessary.
- For large folders, duplicate scanning is O(N²) per file‑type group. The pHash pre‑filter drastically reduces the number of deep comparisons.
- Reports are timestamped and never overwritten; snapshots and diffs allow you to audit changes over time.

## 📄 License

All Rights Reserved. No part of this repository may be reproduced, distributed or transmitted without prior written permission. See LICENSE for details.
