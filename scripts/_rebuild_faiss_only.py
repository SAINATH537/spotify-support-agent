"""
scripts/_rebuild_faiss_only.py
Build FAISS index only from pre-cached embeddings and saved TF-IDF LR classifier.
Uses only absolute paths to avoid CWD issues.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rebuild_faiss")

config_path = ROOT / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

processed_dir = ROOT / cfg["data"]["processed_dir"]
golden_dir = ROOT / cfg["data"]["golden_dir"]
embed_model = cfg["embedding"]["model_name"]
embed_batch = cfg["embedding"]["batch_size"]
cache_dir = ROOT / cfg["embedding"]["cache_dir"]
index_path = ROOT / cfg["retrieval"]["index_path"]
seed = cfg.get("random_seed", 42)

logger.info("ROOT: %s", ROOT)
logger.info("Index will be saved to: %s (absolute)", index_path.resolve())

# Load interactions
df = pd.read_csv(processed_dir / "spotify_interactions.csv")
logger.info("Loaded %d interactions", len(df))

# Exclude golden set
golden_template = golden_dir / "spotify_golden_set_annotation_template_250.csv"
golden_ids: set[str] = set()
if golden_template.exists():
    gdf = pd.read_csv(golden_template)
    golden_ids = set(gdf["tweet_id"].astype(str))
    logger.info("Loaded %d golden IDs to exclude", len(golden_ids))

before = len(df)
df = df[~df["customer_tweet_id"].astype(str).isin(golden_ids)].copy()
n_excluded = before - len(df)
logger.info("Excluded %d golden examples. %d remain.", n_excluded, len(df))

# Verify
remaining = set(df["customer_tweet_id"].astype(str))
assert len(remaining & golden_ids) == 0, "Golden IDs still in index!"
logger.info("Golden exclusion verified: 0 overlap")

# Load or train TF-IDF LR for intent prediction
from src.intent.baseline import TFIDFLogisticRegression, MajorityClassifier
from src.intent.taxonomy import IntentTaxonomy
from sklearn.model_selection import train_test_split

tfidf_path = processed_dir / "tfidf_lr_clf.joblib"
maj_path = processed_dir / "majority_clf.joblib"

if not tfidf_path.exists():
    logger.info("TF-IDF LR not found — training now...")
    taxonomy = IntentTaxonomy(config_path)
    if "weak_intent" not in df.columns:
        df["weak_intent"] = taxonomy.weak_label_batch(df["customer_message"].tolist())
    labels = df["weak_intent"].tolist()
    texts_all = df["customer_message"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        texts_all, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    maj = MajorityClassifier()
    maj.fit(X_train, y_train)
    maj.save(maj_path)
    tfidf_lr = TFIDFLogisticRegression(random_state=seed)
    tfidf_lr.fit(X_train, y_train)
    tfidf_lr.save(tfidf_path)
    logger.info("Classifiers trained and saved")
else:
    tfidf_lr = TFIDFLogisticRegression.load(tfidf_path)
    logger.info("TF-IDF LR loaded from %s", tfidf_path)

# Generate embeddings via subprocess (torch DLL isolation)
import hashlib
import subprocess
import tempfile
import json as _json

texts = df["customer_message"].tolist()
cache_key = hashlib.md5(("retrieval_" + "||".join(texts) + embed_model).encode()).hexdigest()[:16]
cache_path = cache_dir / f"{cache_key}.npy"

if cache_path.exists():
    logger.info("Embedding cache hit: %s", cache_path)
    embeddings = np.load(cache_path).astype(np.float32)
else:
    logger.info("Embedding cache miss — generating via subprocess...")
    embed_script = ROOT / "scripts" / "_run_embed_tmp.py"
    texts_tmp = Path(tempfile.mktemp(suffix=".json"))
    embs_tmp = Path(tempfile.mktemp(suffix=".npy"))
    _json.dump(texts, open(texts_tmp, "w", encoding="utf-8"))
    embed_script.write_text(
        'import sys, json, numpy as np, hashlib\n'
        'sys.path.insert(0, sys.argv[1])\n'
        'texts = json.load(open(sys.argv[2], encoding="utf-8"))\n'
        'model_name = sys.argv[3]\n'
        'cache_dir = sys.argv[4]\n'
        'out_path = sys.argv[5]\n'
        'import torch\n'
        'from sentence_transformers import SentenceTransformer\n'
        'from pathlib import Path\n'
        'cache_key = hashlib.md5(("retrieval_" + "||".join(texts) + model_name).encode()).hexdigest()[:16]\n'
        'cache_path = Path(cache_dir) / f"{cache_key}.npy"\n'
        'if cache_path.exists():\n'
        '    print("Cache hit:", cache_path)\n'
        '    np.save(out_path, np.load(cache_path))\n'
        '    sys.exit(0)\n'
        'print(f"Embedding {len(texts)} texts...")\n'
        'model = SentenceTransformer(model_name)\n'
        'embs = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype("float32")\n'
        'Path(cache_dir).mkdir(parents=True, exist_ok=True)\n'
        'np.save(str(cache_path), embs)\n'
        'np.save(out_path, embs)\n'
        'print(f"Done shape={embs.shape}")\n'
    )
    proc = subprocess.run(
        [sys.executable, str(embed_script), str(ROOT), str(texts_tmp), embed_model, str(cache_dir), str(embs_tmp)],
        capture_output=True, text=True
    )
    print("Subprocess stdout:", proc.stdout[-1000:])
    if proc.returncode != 0:
        print("Subprocess FAILED:\n", proc.stderr[-1000:])
        sys.exit(1)
    embeddings = np.load(embs_tmp).astype(np.float32)
    texts_tmp.unlink(missing_ok=True)
    embs_tmp.unlink(missing_ok=True)
    embed_script.unlink(missing_ok=True)

logger.info("Embeddings shape: %s", embeddings.shape)
assert embeddings.shape[0] == len(df)

# Predict intents
predicted_intents = tfidf_lr.predict(df["customer_message"].tolist())
df = df.copy()
df["intent"] = predicted_intents

# Build metadata
metadata = df[[
    "interaction_id", "conversation_id", "customer_tweet_id", "brand_tweet_id",
    "customer_message", "brand_response", "resolution_type", "intent", "weak_intent", "timestamp",
]].to_dict("records")
for m in metadata:
    m["customer_tweet_id"] = str(m["customer_tweet_id"])
    m["brand_tweet_id"] = str(m["brand_tweet_id"])

# Build FAISS index
from src.retrieval.faiss_store import FAISSStore
t0 = time.perf_counter()
store = FAISSStore()
store.build(embeddings, metadata, intent_key="intent")

# Save with absolute path
abs_index_path = index_path.resolve()
store.save(abs_index_path)
elapsed = time.perf_counter() - t0

# Verify save worked
assert (abs_index_path / "index.faiss").exists(), "index.faiss not found after save!"
assert (abs_index_path / "metadata.json").exists(), "metadata.json not found after save!"
logger.info("Save verified at: %s", abs_index_path)

# Save manifest
manifest = {
    "embedding_model": embed_model,
    "n_indexed": store.size,
    "n_excluded_golden": n_excluded,
    "vector_dim": store.dim,
    "index_type": "IndexFlatIP (exact cosine similarity)",
    "top_k_default": cfg["retrieval"]["top_k"],
    "index_path": str(abs_index_path),
    "runtime_seconds": round(elapsed, 1),
    "seed": seed,
    "intent_classifier": "TFIDFLogisticRegression",
    "split_level": "tweet",
    "split_warning": "Conversation-level split not yet implemented",
}
manifest_path = processed_dir / "index_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

logger.info("=" * 60)
logger.info("FAISS INDEX BUILT AND SAVED")
logger.info("  Vectors   : %d", store.size)
logger.info("  Dim       : %d", store.dim)
logger.info("  Golden ex.: %d", n_excluded)
logger.info("  Runtime   : %.1f s", elapsed)
logger.info("  Path      : %s", abs_index_path)
logger.info("=" * 60)
print(json.dumps(manifest, indent=2))
