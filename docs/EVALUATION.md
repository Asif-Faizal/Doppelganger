# Evaluation methodology

Two different questions need two different evaluations. Do not mix them.

| | Question | Ground truth | Metrics |
| --- | --- | --- | --- |
| `SAME_PERSON` | Is this person in the gallery? | Identity labels | top-1/5/10, FNIR, FPIR, ROC/EER, calibration |
| `LOOKALIKE` | Do the results look like the selfie? | Human judgement | nDCG@k, precision@k, recall@k, MRR |

Everything below is offline and deterministic (seeded). Nothing downloads a dataset or a model; you
supply a directory laid out as `root/<identity>/<image>.jpg` (LFW has exactly this layout).

## 1. SAME_PERSON: identification and threshold calibration

```bash
# 1. embeddings with the production pipeline (use --workers = number of cores)
python scripts/evaluate.py extract --dataset /data/lfw --dataset-name LFW --workers 4 \
    --out eval_out/sface_clean.npz
# 2. protocol, metrics, calibration file for the service
python scripts/evaluate.py identify --gallery eval_out/sface_clean.npz \
    --write-calibration data/calibration.json --out eval_out/identify.json
# 3. robustness: re-embed degraded copies of the same images and use them as probes only
#    (--no-gate measures the embedding itself; the gated runs measure how many photos the
#    quality gate would reject). Degradations: jpeg10 down2 down3 blur2 blur4 dark bright noise20 occlude
python scripts/evaluate.py extract --dataset /data/lfw --workers 4 --no-gate --out eval_out/sface_ng_none.npz
python scripts/evaluate.py extract --dataset /data/lfw --workers 4 --no-gate --degrade down3 \
    --out eval_out/sface_ng_down3.npz
python scripts/evaluate.py identify --gallery eval_out/sface_ng_none.npz --probes eval_out/sface_ng_down3.npz
```

Then set `DOPPEL_CALIBRATION_FILE=data/calibration.json`.

### Protocol (`identify`)

* **Held-out calibration.** Identities are split into a calibration half and a test half (50/50,
  seeded, *identity-disjoint*). Thresholds and the logistic calibration are fitted on the
  calibration half only and then applied unchanged to the test half. The halves are the same size
  on purpose: false-positive identification rate grows with gallery size, so a threshold calibrated
  on a much smaller gallery under-estimates FPIR on a bigger one.
* **Open-set 1:N.** Within a half, 80 % of identities that have at least two photos are *enrolled*
  with one random photo as the reference; their other photos are *mated* probes. Everything else
  (identities left out of the gallery, and identities with a single photo) are *non-mated* probes
  that must not match anything.
* **Metrics.**
  * Closed-set top-1/5/10: the true identity is among the k best gallery entries.
  * **FNIR(t)**: share of mated searches where the true mate is not the best candidate with score
    ≥ t. **FPIR(t)**: share of non-mated searches that return any candidate with score ≥ t. Both
    are reported with Wilson 95 % intervals and the raw counts.
  * Verification (pairwise) ROC: AUC, EER, TAR at FAR 1e-2/1e-3/1e-4/1e-5 (genuine pairs =
    all same-identity pairs, impostor pairs = up to 1 000 000 random cross-identity pairs).
  * Photo-gallery retrieval: every photo is in the gallery, each probe is searched against all
    *other* photos; hit@k, precision@k, mAP. This is the closest match to "search a selfie in a pile
    of photos" (`SAME_PERSON`/`LOOKALIKE` both).
* **Failures to acquire** (no face found, or the quality gate rejects the photo) are counted as
  misses for mated probes and are reported separately; they are never silently dropped.
* **Calibration file.** `P(same | cosine) = sigmoid(a·cos + b)` fitted on calibration-half pairs
  with class balancing (assumes a 1:1 prior, which no real gallery has), plus the cosine threshold
  at the requested pairwise false-accept rate (`--far`, default 1e-4). Rule of thumb for 1:N:
  `FPIR ≈ gallery_size × FAR` for small values, so choose `--far` from the gallery size you expect
  and the false-match rate you can tolerate.

### Caveats that apply to every number

* Public datasets (LFW) are mostly frontal, well-lit, centred press photos. Real selfies and
  gallery photos are harder. **Re-run this on photos that look like your gallery** before relying
  on a threshold.
* The training data of the SFace weights is not documented, so overlap with LFW identities cannot
  be excluded; the LFW numbers may be optimistic.
* A calibrated confidence is only valid for the population it was fitted on.

## 2. LOOKALIKE: human-judgement process

There is no objective ground truth for "looks like"; the only defensible evaluation is human
judgement collected blind. The tooling pools candidates from one or more models so that raters
judge a fixed, de-duplicated list without seeing which model, rank or score produced it.

```bash
# a) pool: top-10 of each model for 50 random queries, shuffled, blind
python scripts/evaluate.py lookalike-sheet --npz eval_out/sface_clean.npz eval_out/dlib_none.npz \
    --queries 50 --depth 10 --out eval_out/judging_sheet.csv
# b) raters fill in the `grade` column (the key file with ranks stays with the organiser)
# c) score every model against the judgments
python scripts/evaluate.py lookalike-score --npz eval_out/sface_clean.npz eval_out/dlib_none.npz \
    --judgments eval_out/judging_sheet.csv --depth 10
```

Rating guidelines to give raters (adapt as needed):

| Grade | Meaning |
| --- | --- |
| 0 | Not similar. |
| 1 | Slightly similar (shares one or two features). |
| 2 | Clearly similar; a casual observer would note the resemblance. |
| 3 | Strong look-alike; could be mistaken for the query at a glance. |

Process rules:

* Use at least 3 raters per pair and average or take the median; report inter-rater agreement
  (for example Krippendorff's alpha) alongside the scores. A model's score is meaningless if raters
  disagree wildly.
* Raters must not know the model, the rank or the similarity score. Do not show same-identity
  information.
* Queries should be sampled from the population you care about (not only celebrities). Judge faces
  only of people who agreed, or public figures in a research setting; do not circulate gallery
  photos outside the organisation.
* Unjudged candidates count as grade 0 (this penalises a model that returns items outside the pool;
  state this when comparing a *new* model against judgments collected for another).
* Report nDCG@k (gain `2^grade − 1`), precision@k and recall@k (relevant = grade ≥ 2 by default),
  and MRR, with the number of queries.

No human-judgement scores are reported in this repository: no raters were available, so the process
and the scorer (`tests/test_evalkit.py` covers it with synthetic judgments) are provided but **no
LOOKALIKE quality number has been measured**.
