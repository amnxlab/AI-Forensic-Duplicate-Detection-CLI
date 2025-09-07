import os
import mimetypes
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loader import clip_model, sbert_deep_model, codebert_model, utilhash

# ====== Paths ======
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SNAPSHOT_DIR = os.path.join(BASE_REPORTS_DIR, "snapshots")
REPORT_DIR = os.path.join(BASE_REPORTS_DIR, "diffs")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ====== Logging ======
def status(msg):
    print(f"[*] {msg}")

def info(msg):
    print(f"[+] {msg}")

def warning(msg):
    print(f"[!] {msg}")

# ====== File type detection ======
COMPOUND_BIN_EXTS = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz")
TEXT_EXTS = {
    ".txt", ".md", ".log", ".sh", ".bash", ".zsh", ".conf", ".ini", ".cfg",
    ".profile", ".bashrc", ".zshrc", ".py", ".c", ".cpp", ".java", ".js"
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
BINARY_EXTS = {".appimage", ".deb", ".rpm", ".tar", ".gz", ".zip", ".bin", ".run"}

def detect_file_type(file_path):
    lower = file_path.lower()

    # Handle compound archive extensions first
    for ext in COMPOUND_BIN_EXTS:
        if lower.endswith(ext):
            return "binary"

    _, ext = os.path.splitext(lower)

    if ext in TEXT_EXTS:
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in BINARY_EXTS:
        return "binary"

    # Fallback to mimetype
    mtype, _ = mimetypes.guess_type(file_path)
    if mtype:
        if mtype.startswith("text/"):
            return "text"
        if mtype.startswith("image/"):
            return "image"
        if mtype.startswith("application/"):
            return "binary"

    # Heuristic: try opening as text
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(2048)
            return "text"
        except Exception:
            return "unknown"

    return "unknown"

# ====== Helpers ======
def sanitize_path(path):
    return path.replace("/", "_").replace("\\", "_").strip("_")

def generate_snapshot_filename(folder_path):
    now = datetime.now()
    folder_id = sanitize_path(folder_path)
    return f"{folder_id}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

def extract_timestamp_from_name(filename):
    try:
        time_str = "_".join(filename.split("_")[-2:]).replace(".txt", "")
        return datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None

def load_model_for_type(file_type, file_path=None):
    if file_type == "image":
        try:
            model, preprocess = clip_model.load_model()
            return model, preprocess, clip_model
        except Exception:
            return None, None, None
    elif file_type == "text":
        if file_path and file_path.endswith((".sh", ".py", ".c", ".cpp", ".java", ".js")):
            try:
                sbert = sbert_deep_model.load_model()
                tokenizer, codebert = codebert_model.load_model()
                return (sbert, (tokenizer, codebert)), None, "hybrid"
            except Exception:
                return None, None, None
        else:
            try:
                model = sbert_deep_model.load_model()
                return model, None, sbert_deep_model
            except Exception:
                return None, None, None
    return None, None, None

def hash_file(file_path, model, extra, module, file_type):
    try:
        if file_type == "image":
            return module.extract_features(file_path, model, extra)
        elif file_type == "text":
            if module == "hybrid":
                sbert, (tokenizer, codebert) = model
                vec1 = sbert_deep_model.extract_features_from_file(file_path, sbert)
                vec2 = codebert_model.extract_features_from_file(file_path, tokenizer, codebert)
                if vec1 is not None and vec2 is not None:
                    return np.concatenate([vec1, vec2])
                return vec1 if vec1 is not None else vec2
            else:
                return module.extract_features_from_file(file_path, model)
    except Exception:
        return None
    return None

# ====== Snapshot generation ======
def generate_snapshot(folder_path):
    """
    Always include a SHA-256 hash for every file (authoritative byte-level change detection).
    Optionally include AI vector for text/images if model succeeds.
    Entry format:
        {"mode": "AIHASH", "hash": "<sha256>", "value": [<floats>] (optional)}
    """
    snapshot = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)

            # Always compute byte-level hash
            file_hash = utilhash.compute_sha256(full_path)
            entry = {"mode": "AIHASH", "hash": file_hash}

            # Try AI features for supported types
            file_type = detect_file_type(full_path)
            if file_type in ["image", "text"]:
                model, extra, module = load_model_for_type(file_type, full_path)
                if model is not None:
                    vec = hash_file(full_path, model, extra, module, file_type)
                    if vec is not None:
                        entry["value"] = vec.tolist()

            snapshot[full_path] = entry
    return snapshot

# ====== Snapshot save/load (backward compatible) ======
# New line format (AI + hash):
#   filepath::AIHASH::<sha256>::v1/v2/v3...   (vector part may be empty)
# Old formats still supported:
#   filepath::AI::comma,separated,floats
#   filepath::HASH::<sha256>

def save_snapshot(snapshot, filename):
    path = os.path.join(SNAPSHOT_DIR, filename)
    with open(path, 'w') as f:
        for filepath, entry in snapshot.items():
            mode = entry.get("mode")
            if mode == "AIHASH":
                sha = entry["hash"]
                vec = entry.get("value")
                if vec is not None:
                    # Use "/" to avoid commas clashing with legacy AI format
                    f.write(f"{filepath}::AIHASH::{sha}::{'/'.join(map(str, vec))}\n")
                else:
                    f.write(f"{filepath}::AIHASH::{sha}::\n")
            elif mode == "AI":
                f.write(f"{filepath}::AI::{','.join(map(str, entry['value']))}\n")
            elif mode == "HASH":
                f.write(f"{filepath}::HASH::{entry['value']}\n")
            else:
                # Fallback: treat as hash-only if present
                sha = entry.get("hash")
                if sha:
                    f.write(f"{filepath}::AIHASH::{sha}::\n")
                else:
                    f.write(f"{filepath}::HASH::{entry.get('value','')}\n")
    return path

def load_snapshot(filename):
    path = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(path):
        return None
    snapshot = {}
    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if "::AIHASH::" in line:
                # filepath::AIHASH::<sha256>::<v1/v2/v3...>  (vector part may be empty)
                before, after = line.split("::AIHASH::", 1)
                filepath = before
                if "::" in after:
                    sha, vec_part = after.split("::", 1)
                else:
                    sha, vec_part = after, ""
                entry = {"mode": "AIHASH", "hash": sha}
                if vec_part:
                    try:
                        vec = list(map(float, vec_part.split("/")))
                        if len(vec) > 0:
                            entry["value"] = vec
                    except Exception:
                        # Corrupt or unexpected vector format; keep only hash
                        pass
                snapshot[filepath] = entry

            elif "::AI::" in line:
                filepath, vec_str = line.split("::AI::", 1)
                try:
                    vector = list(map(float, vec_str.split(","))) if vec_str else []
                except Exception:
                    vector = []
                snapshot[filepath] = {"mode": "AI", "value": vector}

            elif "::HASH::" in line:
                filepath, hashval = line.split("::HASH::", 1)
                snapshot[filepath] = {"mode": "HASH", "value": hashval}
    return snapshot

# ====== Compare ======
def compare_snapshots(prev, current, sim_threshold=0.999999):
    """
    Change detection:
      1) Prefer byte-level SHA detection (authoritative).
      2) If hashes equal and both embeddings exist, use cosine similarity as enrichment.
    """
    changed = []

    # Files that are new or modified
    for path, entry in current.items():
        if path not in prev:
            changed.append((path, "NEW"))
            continue

        prev_entry = prev[path]

        # Unify hash access
        cur_hash = entry.get("hash")
        prev_hash = prev_entry.get("hash")

        if cur_hash is None and entry.get("mode") == "HASH":
            cur_hash = entry.get("value")
        if prev_hash is None and prev_entry.get("mode") == "HASH":
            prev_hash = prev_entry.get("value")

        # 1) Byte-level check
        if cur_hash and prev_hash and cur_hash != prev_hash:
            changed.append((path, "MODIFIED (bytes changed)"))
            continue

        # 2) Optional vector-level comparison
        cur_vec = entry.get("value")
        prev_vec = prev_entry.get("value")
        if cur_vec is not None and prev_vec is not None:
            try:
                if len(cur_vec) != len(prev_vec):
                    changed.append((path, "MODIFIED (embedding shape changed)"))
                else:
                    sim = float(cosine_similarity([cur_vec], [prev_vec])[0][0])
                    if sim < sim_threshold:
                        changed.append((path, f"MODIFIED (Similarity: {sim:.8f})"))
            except Exception:
                # If similarity computation fails, ignore vector comparison
                pass

    # Files that disappeared
    for path in prev.keys() - current.keys():
        changed.append((path, "DELETED"))

    return changed

# ====== Load latest snapshot before a given filename (same prefix) ======
def load_latest_snapshot(before_filename=None):
    if not before_filename:
        return None, None
    current_time = extract_timestamp_from_name(before_filename)
    current_prefix = "_".join(before_filename.split("_")[:-2])
    snapshots = []
    for f in os.listdir(SNAPSHOT_DIR):
        if not f.endswith(".txt") or not f.startswith(current_prefix):
            continue
        file_time = extract_timestamp_from_name(f)
        if file_time and current_time and file_time < current_time:
            snapshots.append((file_time, f))
    if not snapshots:
        return None, None
    snapshots.sort()
    latest_name = snapshots[-1][1]
    return latest_name, load_snapshot(latest_name)

# ====== Main ======
def main(folder, sim_threshold=0.999999):
    snapshot_filename = generate_snapshot_filename(folder)
    snapshot = generate_snapshot(folder)
    snapshot_path = save_snapshot(snapshot, snapshot_filename)
    info(f"Snapshot saved: {snapshot_path}")

    prev_name, prev_snapshot = load_latest_snapshot(before_filename=snapshot_filename)
    if prev_snapshot:
        info(f"Comparing with previous snapshot: {prev_name}")
        changes = compare_snapshots(prev_snapshot, snapshot, sim_threshold=sim_threshold)
        if changes:
            diff_name = f"diff_{snapshot_filename.replace('.txt', '')}_vs_{prev_name.replace('.txt','')}.txt"
            report_path = os.path.join(REPORT_DIR, diff_name)
            with open(report_path, "w") as f:
                for path, msg in changes:
                    f.write(f"{path} ==> {msg}\n")
            warning(f"Changes detected: {len(changes)} (saved to {report_path})")
        else:
            status("No significant changes since last snapshot.")
    else:
        status("No previous snapshot to compare.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder to snapshot")
    parser.add_argument("--sim-threshold", type=float, default=0.999999,
                        help="Cosine similarity threshold for AI vectors (only used when hashes are equal).")
    args = parser.parse_args()
    main(args.folder, args.sim_threshold)