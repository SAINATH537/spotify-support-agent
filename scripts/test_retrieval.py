#!/usr/bin/env python
"""
scripts/test_retrieval.py
─────────────────────────────────────────────────────────────────────────────
CLI retrieval demo. Queries the FAISS index and shows top-K results.

Usage:
  python scripts/test_retrieval.py --query "My premium subscription stopped working"
  python scripts/test_retrieval.py --query "App crashes on Android" --top-k 5
  python scripts/test_retrieval.py --query "..." --intent "Playback / Audio"
  python scripts/test_retrieval.py --sanity-check   # 25-query sanity pass
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # suppress INFO during demo


def _load_index(cfg: dict):
    from src.retrieval.faiss_store import FAISSStore
    index_path = ROOT / cfg["retrieval"]["index_path"]
    if not (Path(str(index_path)) / "index.faiss").exists():
        print(f"\nERROR: FAISS index not found at {index_path}")
        print("Run:  python scripts/build_index.py")
        sys.exit(1)
    return FAISSStore.load(index_path)


def _embed(query: str, model_name: str) -> "np.ndarray":
    from src.retrieval.embeddings import embed_single
    return embed_single(query, model_name=model_name)


def _print_results(query: str, results: list[dict], top_k: int) -> None:
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("RETRIEVAL DEMO")
    print(SEP)
    print(f"\nQuery:\n  {query}\n")
    if not results:
        print("  (no results returned)")
        return
    for i, r in enumerate(results[:top_k], 1):
        print(f"Result {i}")
        print(f"  Similarity  : {r.get('similarity', 0):.4f}")
        print(f"  Intent      : {r.get('intent', r.get('weak_intent', 'N/A'))}")
        print(f"  Resolution  : {r.get('resolution_type', 'N/A')}")
        print(f"  Tweet IDs   : customer={r.get('customer_tweet_id','?')} "
              f"brand={r.get('brand_tweet_id','?')}")
        cust = r.get("customer_message", "")
        print(f"  Customer    : {cust[:100]}{'...' if len(cust)>100 else ''}")
        resp = r.get("brand_response", "")
        print(f"  SpotifyCares: {resp[:120]}{'...' if len(resp)>120 else ''}")
        print()
    print(SEP)


SANITY_QUERIES = [
    # Playback / Audio
    "My music stops playing randomly while I'm listening",
    "Shuffle keeps repeating the same songs",
    "Audio quality is really bad, sounds muffled",
    "Offline downloaded songs won't play",
    "Songs skip after a few seconds",
    # App / Device / Technical
    "Spotify app keeps crashing on my iPhone",
    "App freezes when I try to open it on Android",
    "Desktop app won't open on Windows 10",
    "Spotify is extremely slow and lagging",
    "The app won't connect to my Bluetooth speaker",
    # Account / Login / Security
    "I can't log into my account, password not working",
    "Someone hacked my account",
    "I forgot my password and can't reset it",
    # Premium / Subscription
    "My premium subscription stopped working",
    "I cancelled premium but still getting charged",
    "How do I upgrade to family plan",
    "Student discount not applying correctly",
    # Billing / Payment
    "I was charged twice this month",
    "I want a refund for my subscription",
    "My payment keeps getting declined",
    # Playlist / Library
    "My liked songs disappeared from my library",
    "Playlist I created is gone",
    # Content Availability
    "Song is not available in my country",
    "Album I used to listen to has been removed",
    # Ads
    "Getting way too many ads even on premium",
    # Feature
    "How do I use Spotify on my car screen",
]


def run_sanity_check(store, embed_model: str, top_k: int) -> None:
    print("\n" + "=" * 62)
    print("RETRIEVAL SANITY CHECK — 25 representative queries")
    print("=" * 62)
    observations = []

    for i, query in enumerate(SANITY_QUERIES, 1):
        vec = _embed(query, embed_model)
        results = store.search(vec, top_k=top_k)
        top = results[0] if results else None
        if top:
            sim = top.get("similarity", 0)
            resp = top.get("brand_response", "")[:80]
            intent = top.get("intent", top.get("weak_intent", "?"))
            cust_ctx = top.get("customer_message", "")[:60]
            flag = ""
            if sim < 0.30:
                flag = " ⚠️ LOW-SIM"
            if len(resp) < 20:
                flag += " ⚠️ SHORT-RESP"
            observations.append((query, sim, flag))
            print(f"\n[{i:02d}] Query : {query[:70]}")
            print(f"      Top-1  : sim={sim:.3f}{flag}")
            print(f"      Intent : {intent}")
            print(f"      Cust   : {cust_ctx}")
            print(f"      Reply  : {resp}")
        else:
            observations.append((query, 0.0, " ⚠️ NO RESULTS"))
            print(f"\n[{i:02d}] Query : {query[:70]}")
            print(f"      ⚠️  No results returned")

    print("\n" + "=" * 62)
    print("SANITY CHECK SUMMARY")
    print("=" * 62)
    sims = [s for _, s, _ in observations]
    flagged = [(q, f) for q, _, f in observations if f]
    print(f"Queries tested   : {len(SANITY_QUERIES)}")
    print(f"Avg top-1 sim    : {sum(sims)/len(sims):.3f}")
    print(f"Min top-1 sim    : {min(sims):.3f}")
    print(f"Max top-1 sim    : {max(sims):.3f}")
    print(f"Flagged queries  : {len(flagged)}")
    for q, f in flagged:
        print(f"  {f} — {q[:60]}")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS Retrieval Demo")
    parser.add_argument("--query", type=str, default=None,
                        help="Customer message to retrieve similar cases for")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of results to return (default: 3)")
    parser.add_argument("--intent", type=str, default=None,
                        help="Optional intent filter for intent-aware retrieval")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run 25-query sanity check over the index")
    args = parser.parse_args()

    if not args.query and not args.sanity_check:
        parser.print_help()
        print("\nExample:")
        print('  python scripts/test_retrieval.py --query "My premium subscription stopped working"')
        sys.exit(0)

    config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    embed_model = cfg["embedding"]["model_name"]
    top_k = args.top_k

    print(f"Loading FAISS index from {cfg['retrieval']['index_path']}...")
    store = _load_index(cfg)
    print(f"Index loaded: {store.size} vectors, dim={store.dim}")

    if args.sanity_check:
        run_sanity_check(store, embed_model, top_k=3)

    if args.query:
        vec = _embed(args.query, embed_model)
        results = store.search(vec, top_k=top_k, intent=args.intent)
        _print_results(args.query, results, top_k)


if __name__ == "__main__":
    main()
