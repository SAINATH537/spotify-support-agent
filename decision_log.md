# Engineering Decision Log
### SpotifyCares AI Support Agent

> Each entry explains a non-obvious decision, alternatives considered, why the decision was made, and its trade-offs.

---

## DEC-01: Why SpotifyCares?

**Decision**: SpotifyCares was selected as the target brand.

**Alternatives considered**:
- AppleSupport (~90k tweets, high volume)
- AmazonHelp (~166k tweets, most tweets)
- Delta (~44k tweets, similar size to Spotify)

**Rationale**:
- 43,265 brand tweets — large enough to train classifiers and build a FAISS index, small enough for fast iteration
- Domain is consumer software: intents are well-scoped (login, billing, playback, app, subscription)
- Brand has a recognisable support style: friendly, emoji-using, escalation-heavy for account issues
- Problem space is familiar to engineers evaluating this submission

**Trade-off**: Smaller dataset than Amazon/Apple; rare intents (Content Availability, Ads) have fewer examples.

---

## DEC-02: Why ~10 Intents?

**Decision**: The taxonomy has exactly 10 intent classes.

**Alternatives considered**:
- 5 classes (merge App+Playback, merge Premium+Billing): simpler but conflates distinct user needs
- 15+ classes (separate Offline vs Streaming, Podcast vs Music): more precise but requires more labelled data
- Data-driven clustering (k-means on embeddings): no human interpretability guarantee

**Rationale**:
- 10 classes balance granularity vs. learnability
- Each class maps to a distinct customer need and resolution path
- High-risk classes (Account, Billing) are separated to enable precise escalation logic
- Matches the data's natural distribution as seen in keyword prevalence analysis

**Trade-off**: "Other / Ambiguous" absorbs ~20-30% of data; splitting it further requires more labelled data.

---

## DEC-03: Why Macro F1 as the Headline Metric?

**Decision**: Macro F1 is the primary headline classification metric.

**Alternatives considered**:
- Accuracy: misleading with imbalanced classes (App/Device is ~20% of data; guessing it always gives 20% accuracy)
- Weighted F1: biased toward majority class; tells you how often the common case works, not the tail
- Per-class F1 only: useful for analysis but not a single headline number

**Rationale**:
- Every intent must work, not just the majority one
- A Billing classifier that fails 90% of the time but handles App/Device well should not look good
- Macro F1 penalises equally for each class regardless of frequency

**Trade-off**: Macro F1 can be gamed by over-predicting rare classes; per-class results are always reported alongside.

---

## DEC-04: Why FAISS with IndexFlatIP?

**Decision**: FAISS IndexFlatIP with normalised vectors (= cosine similarity).

**Alternatives considered**:
- ChromaDB / Weaviate / Qdrant: managed vector DBs — unnecessary infra for this scale
- FAISS HNSW (approximate): faster for millions of vectors but introduces non-determinism and approximation error
- BM25 (sparse retrieval): keyword-based, misses semantic similarity
- Exact numpy cosine: no batch SIMD optimisation, 10–100x slower at scale

**Rationale**:
- IndexFlatIP is exact (no approximation), deterministic, and sufficient for ≤50k vectors
- Local file — no network dependency during inference
- FAISS is battle-tested at scale; trivial to upgrade to HNSW if dataset grows
- Cosine similarity via normalised inner product is the right metric for sentence-transformer outputs

**Trade-off**: Not suitable for millions of vectors without indexing upgrade; index must fit in RAM.

---

## DEC-05: Why Filter Historical Evidence Before Retrieval?

**Decision**: Only "retrieval-worthy" brand responses enter the FAISS index (troubleshooting, informational, clarification_request).

**Alternatives considered**:
- Index all brand responses and rely on similarity to filter naturally
- Post-filter: retrieve all, then filter by response type at query time

**Rationale**:
- Acknowledgement messages ("Thanks for reaching out!") are semantically similar to many queries but useless as evidence
- DM escalation messages ("Please DM us") would be retrieved as evidence for nearly every query and cause the LLM to suggest DM-ing unnecessarily
- Pre-filtering keeps the index clean and improves retrieval quality

**Trade-off**: Some conversations where the "closing" contains actionable info are lost; edge case acceptable for v1.

---

## DEC-06: Why a Deterministic Escalation Policy (Not LLM-Decided)?

**Decision**: Escalation is decided by a rule-based policy layer, not by asking the LLM.

**Alternatives considered**:
- Ask the LLM "Should I escalate?" as part of the generation prompt
- Train a classifier on escalation labels
- Use a calibrated confidence threshold from the LLM

**Rationale**:
- LLM-decided escalation is non-deterministic and non-auditable
- If the LLM hallucinates (or fails), it might auto-handle a security-critical issue
- Rules are inspectable, testable, and modifiable during a live interview
- High-risk intents (Account, Billing) *must always* escalate regardless of confidence — a rule is the only way to guarantee this
- Thresholds can be tuned via a simple threshold sweep (see `evaluate_escalation.py`)

**Trade-off**: Rules cannot capture nuanced cases; a fine-tuned classifier would handle edge cases better, but at the cost of transparency.

---

## DEC-07: Why Freeze the Golden Set?

**Decision**: The 250-example golden evaluation set is frozen before any classifier training, threshold tuning, or prompt development.

**Alternatives considered**:
- Use the golden set iteratively to improve the system (train-on-test)
- Create a separate "development" evaluation set and keep the golden for final submission

**Rationale**:
- Evaluating on data that influenced the system design gives inflated numbers
- Frozen set ensures honest reporting: metrics reflect true generalisation
- This is the most important principle in ML evaluation integrity

**Trade-off**: With only 250 examples, the golden set has high variance; confidence intervals should be reported but are omitted for brevity.

---

## DEC-08: Why Measure LLM Judge Agreement with Humans?

**Decision**: 50 generated responses are scored by both the LLM judge and a human, with Spearman correlation and Cohen's kappa reported.

**Alternatives considered**:
- Trust the LLM judge unconditionally
- Report only LLM judge scores

**Rationale**:
- LLM judges have known biases (length preference, self-consistency bias, style preferences)
- Without human validation, the judge scores are unverified opinions
- The spec explicitly requires this — but more importantly, it's the right engineering practice
- Spearman correlation measures ordinal agreement; Cohen's kappa measures categorical agreement

**Trade-off**: Human scoring is laborious; 50 examples may be insufficient to detect subtle biases.

---

## DEC-09: Why Report Automation Coverage vs. Unsafe Auto-Handle Rate?

**Decision**: The primary product metric is the trade-off between automation coverage (fraction auto-handled) and unsafe auto-handle rate (fraction that should have escalated but didn't).

**Alternatives considered**:
- Report only accuracy / F1 for escalation
- Report only automation coverage

**Rationale**:
- The product team cares about two things: "how much can we automate?" and "how often do we dangerously auto-handle?"
- These are in fundamental tension: increasing automation always increases some risk
- A threshold sweep plot makes this trade-off visible and allows the product team to choose their operating point
- "Unsafe auto-handle" is the most dangerous failure mode — the safe failure is always to escalate

**Trade-off**: Reporting this metric requires gold escalation labels, which the annotation template currently only partially covers.

---

## DEC-10: Why all-MiniLM-L6-v2?

**Decision**: sentence-transformers `all-MiniLM-L6-v2` is used for all embeddings.

**Alternatives considered**:
- `all-mpnet-base-v2`: better quality but 2× slower, 3× larger
- `text-embedding-ada-002` (OpenAI): excellent quality but requires API call per inference (latency + cost)
- `paraphrase-multilingual-MiniLM-L12-v2`: multilingual support but lower English quality
- `BAAI/bge-base-en-v1.5`: marginally better on MTEB benchmarks but less widely deployed

**Rationale**:
- 22M params, 384-dim output, ~5ms per batch on CPU
- Top of its size class on MTEB semantic similarity benchmarks
- No GPU required; runs on any machine
- Widely deployed → reliable behaviour, good community support
- No API cost

**Trade-off**: Lower quality than large models; an upgrade to `all-mpnet-base-v2` would improve retrieval recall by ~5-8%.

---

## DEC-11: Why Not Build a Full RAG with LangChain/LlamaIndex?

**Decision**: The RAG pipeline is implemented from scratch without orchestration frameworks.

**Alternatives considered**:
- LangChain: widely used, many integrations
- LlamaIndex: good for document Q&A
- Haystack: production-grade NLP pipeline

**Rationale**:
- PROOF > COMPLEXITY — the submission must be understandable and modifiable in an interview
- The pipeline is only 4 steps: embed → retrieve → generate → decide. No orchestration framework needed.
- Frameworks add opaque abstractions that make debugging harder during live review
- Every component in this system is < 200 lines of readable Python

**Trade-off**: More boilerplate code; would not scale to complex multi-hop retrieval without refactoring.

---

## DEC-12: Why Cache Embeddings on Disk?

**Decision**: Sentence-transformer embeddings are cached to disk using numpy `.npy` files keyed by MD5 hash of the input.

**Alternatives considered**:
- Recompute on every run
- Cache in memory (RAM cache)
- Use a key-value store (Redis, SQLite)

**Rationale**:
- Embedding 40k+ interactions takes ~5–15 minutes on CPU
- The data doesn't change between `build_index.py` runs
- Disk cache + MD5 hash = deterministic reuse with zero infrastructure
- Critical for the "< 15 minute reproduce" target

**Trade-off**: Cache invalidation requires deleting `.npy` files manually if the model or data changes. A content-addressed store (like DVC) would be more robust.
