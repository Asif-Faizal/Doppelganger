# Measured results

Everything here was **measured on the machine below with the commands at the end**; raw JSON outputs
are in [`docs/results/`](results/). Nothing is extrapolated. Where something was *not* measured
that is stated in [What was not measured](#what-was-not-measured).

## Hardware and software

| | |
| --- | --- |
| CPU | 4 cores, 1 thread/core, Intel Xeon (virtualised; `/proc/cpuinfo` reports only "Intel(R) Xeon(R) Processor"; flags include AVX-512 and AMX), 8 MiB L2 (4 instances), 320 MiB L3 |
| RAM | 15.6 GB |
| GPU | none |
| OS | Linux 6.12 (x86_64, glibc 2.39), Python 3.12.3 |
| Libraries | numpy 2.5.4, faiss-cpu 1.16.0, opencv-python-headless 4.14.0.94 |
| Models | YuNet 2023mar + SFace 2021dec, `model_id = sface-2021dec.yunet-2023mar.p1` |

Timing runs (index, service, inference, HTTP) were run with nothing else active on the machine.
Accuracy runs were executed in parallel with other jobs, so **no latency is reported from them**.
This is a shared virtual machine: expect different absolute numbers elsewhere.

## 1. Recognition accuracy on LFW (SAME_PERSON)

Data: Labeled Faces in the Wild, all 13 233 images / 5 749 identities (public research dataset,
sha256-checked archive; see [MODEL_LICENSES.md](MODEL_LICENSES.md#evaluation-data)). Protocol and
caveats: [EVALUATION.md](EVALUATION.md). In short: identity-disjoint 50/50 split into a
*calibration* half and a *test* half, thresholds fixed on the calibration half only, one enrolment
photo per enrolled identity, open-set (non-enrolled people are searched too), failures to acquire
counted as misses. Test half: **687 enrolled identities**, 3 178 mated probes, 2 770 non-mated
probes.

### Production configuration (quality gate on)

| Metric (test half) | Result |
| --- | --- |
| Top-1 / top-5 / top-10 identification | **98.71 % / 99.09 % / 99.12 %** |
| Failures to acquire (gate rejections) | 16 of 13 233 images (0.12 %), all `face_blurry`; 2 mated and 4 non-mated probes in the test half |
| Verification EER (pairwise, 169 k genuine / 1 M impostor pairs) | 0.11 % |
| Verification TAR at FAR 1e-2 / 1e-3 / 1e-4 | 99.94 % / 99.89 % / 99.71 % |
| Photo-gallery retrieval (6 629 photos, 3 000 probes): hit@1 / hit@5 / hit@10, mAP | 99.37 % / 99.70 % / 99.73 %, 0.989 |

Open-set operating points. Thresholds were chosen on the calibration half for the stated FPIR and
then applied unchanged to the test half (95 % Wilson intervals in brackets):

| Threshold (cosine) | Chosen for | Test FNIR | Test FPIR |
| --- | --- | --- | --- |
| 0.401 | calibration FPIR 10 % | 1.6 % [1.2, 2.1] | 12.6 % [11.4, 13.9] |
| 0.455 | calibration FPIR 1 % | 3.4 % [2.8, 4.1] | **1.2 %** [0.85, 1.7] |
| 0.502 | calibration FPIR 0.1 % | 6.4 % [5.6, 7.3] | 0.18 % [0.08, 0.42] |
| **0.363** | vendor default (pairwise) | 1.4 % [1.0, 1.9] | **40.3 %** [38.5, 42.2] |

**The vendor's 0.363 is a pairwise-verification threshold.** Used for 1:N search against 687
enrolled people it returned a candidate for 40 % of people who were *not* in the gallery. This is
why `SAME_PERSON` reports `threshold_source: vendor_default_uncalibrated` until you install a
calibration, and why the README warns about it. FPIR grows roughly linearly with gallery size
(`FPIR ≈ gallery × FAR` while small): the thresholds above are valid for galleries of **about
700 identities** and must be raised for larger ones.

The calibration actually produced from this run (logistic `a=47.12, b=-16.46`, pairwise
FAR 1e-4 → threshold 0.4151) is kept as an *example*, not as a default:
[`results/calibration_sface_lfw.example.json`](results/calibration_sface_lfw.example.json).
It is a class-balanced fit on a public dataset and is not valid for other photo populations.

### Robustness: degraded probes (quality gate off, enrolment photos clean)

Same protocol; only the *probe* images are degraded. "Gate" columns show how many of the 13 233
images the production quality gate would have rejected instead of embedding.

| Probe condition | Top-1 | Top-5 | EER | TAR@FAR 1e-3 | FNIR / FPIR at the 1 %-calibrated threshold | Photo hit@1 | Gate rejects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 98.77 % | 99.15 % | 0.11 % | 99.88 % | 3.3 % / 1.2 % | 99.30 % | 0.12 % |
| darker (gamma 3.0) | 97.64 % | 98.68 % | 0.34 % | 99.39 % | 7.6 % / 1.3 % | 97.83 % | 2.28 % |
| brighter (gamma 0.35) | 98.71 % | 99.09 % | 0.16 % | 99.79 % | 3.8 % / 1.1 % | 98.97 % | 1.30 % |
| down-scaled 3x (250 px → 83 px) | 97.83 % | 98.84 % | 0.40 % | 99.10 % | 6.5 % / 1.4 % | 97.80 % | **99.21 %** |
| Gaussian noise σ=20 | 96.10 % | 97.86 % | 1.44 % | 96.29 % | 12.1 % / 1.0 % | 95.13 % | 0.67 % |
| lower-face band masked (mask-like) | 97.61 % | 98.62 % | 0.38 % | 99.30 % | 8.3 % / 0.9 % | 97.83 % | 0.14 % |
| Gaussian blur σ=4 | 83.32 % | 92.45 % | 3.94 % | 74.33 % | 48.5 % / 0.8 % | 85.00 % | **99.75 %** |
| JPEG quality 10 (gate on in both) | 96.85 % | n/a | 1.04 % | n/a | 12.9 % / 1.1 % | n/a | 0.03 % |

Reading the table: with the gate on, heavy down-scaling and blur are *rejected* (HTTP 422
`face_quality_insufficient`) rather than answered with a poor match, which is the intended behaviour;
the accuracy columns in those two rows say how well the embedder would have done had the gate not
been there. Noise and darkening are the conditions where accuracy degrades while the gate still
lets photos through.

### Comparison with the original dlib pipeline (same protocol, no gate)

| Pipeline | Condition | Top-1 | Top-5 | EER | TAR@FAR 1e-3 | Photo hit@1 | Failures to acquire (test probes) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SFace + YuNet** | clean | 98.77 % | 99.15 % | 0.11 % | 99.88 % | 99.30 % | 0 |
| dlib HOG + ResNet (old) | clean | 95.10 % | 98.81 % | 0.61 % | 95.75 % | 97.00 % | 24 |
| **SFace + YuNet** | darker | 97.64 % | 98.68 % | 0.34 % | 99.39 % | 97.83 % | 1 |
| dlib | darker | 90.92 % | 95.60 % | 1.02 % | 90.59 % | 94.73 % | 184 |
| **SFace + YuNet** | down-scaled 3x | 97.83 % | 98.84 % | 0.40 % | 99.10 % | 97.80 % | 2 |
| dlib | down-scaled 3x | 90.57 % | 96.41 % | 1.39 % | 86.41 % | 94.40 % | 138 |

The old pipeline's own default threshold (euclidean distance 0.6) applied as a 1:N threshold
returned a candidate for **78.8 %** of non-enrolled probes on this gallery (FNIR 5.0 %), so the
original service would have reported a "match" for most strangers. dlib was not run on the blur,
noise, bright or mask conditions (each dlib extraction takes about 8 minutes on this machine).

Caveats for the comparison: (1) the training data of the SFace ONNX is not documented, so overlap
with LFW identities is possible and the SFace figures may be optimistic ([MODEL_LICENSES.md](MODEL_LICENSES.md));
(2) dlib's enrolment/probe sets differ marginally because its failures to acquire differ (685 vs 686
enrolled identities); (3) LFW is easy, frontal, centred press photography.

## 2. Vector index (synthetic vectors)

128-d unit vectors in identity clusters of 1-8 (intra-identity cosine about 0.6-0.7), 1 000 queries,
FAISS using 2 OpenMP threads (`DOPPEL_INDEX_THREADS` default). Synthetic data: this measures speed
and memory, and the recall of *approximate* search relative to exact search on the same synthetic
vectors; it says nothing about recognition accuracy.

**Exact flat inner-product index (what the service uses):**

| Vectors | Build | Snapshot size / load | Extra RSS | p50 / p95 / p99 (k=10, 1 query) | Queries/s with 1 / 4 / 8 client threads |
| --- | --- | --- | --- | --- | --- |
| 10 000 | <0.01 s | 5.0 MB / 3 ms | 0.6 MB | 0.23 / 0.50 / 0.63 ms | 4 626 / 7 434 / 8 340 |
| 100 000 | 0.03 s | 49.6 MB / 47 ms | 48.9 MB | 1.82 / 1.97 / 2.36 ms | 961 / 1 042 / 1 061 |

**HNSW (benchmark only, not in the service)**: M=32, efConstruction=200.

| Vectors | efSearch | recall@10 vs exact | p50 / p99 | Build |
| --- | --- | --- | --- | --- |
| 10 000 | 32 / 64 / 128 | 98.2 % / 99.4 % / 99.9 % | 0.04 / 0.07, 0.07 / 0.10, 0.15 / 0.19 ms | 0.8 s |
| 100 000 | 32 / 64 / 128 | 92.8 % / 95.3 % / 97.3 % | 0.08 / 0.11, 0.14 / 0.17, 0.26 / 0.34 ms | 22.8 s |

At 100 000 vectors exact search costs about 2 ms, so approximate search buys nothing that matters
and costs recall; the service keeps the exact index. Beyond roughly a million faces this trade-off
should be re-measured (it was not, see below).

## 3. Service-level search (SQLite + FAISS, no inference)

`SearchService.search_vector` over a real SQLite database with N faces (bulk-inserted synthetic
rows): index sync check, FAISS query, hit hydration from SQLite (`fetch_hits`), de-duplication,
response objects. 1 000 queries, top-10.

| Faces | SQLite size | Index snapshot | Cold start: rebuild index from SQLite | Warm start: load snapshot | RSS after load | LOOKALIKE p50 / p95 / p99 | SAME_PERSON p50 / p95 / p99 | Queries/s with 1 / 4 / 8 client threads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 000 | 11.3 MB | 5.0 MB | 0.04 s | 7 ms | 159 MB | 1.54 / 1.75 / 1.91 ms | 1.52 / 1.70 / 2.07 ms | 546 / 363 / 233 |
| 100 000 | 113 MB | 49.6 MB | 0.32 s | 61 ms | 369 MB | 3.36 / 3.84 / 4.25 ms | 3.31 / 3.48 / 3.93 ms | 357 / 320 / 227 |

Throughput does **not** increase with client threads here: this path is dominated by Python and
SQLite row hydration under the GIL, and eight threads oversubscribe the 4 cores. It is far above
what the inference stage can feed (next section), so it was not optimised.

## 4. Inference latency (detect + embed, one face)

Single process, sequential, 10 warm-up images discarded.

| Input | OpenCV threads | Detect p50 / p95 | Embed p50 / p95 | Total p50 / p95 / p99 |
| --- | --- | --- | --- | --- |
| LFW 250×250 (300 images) | 1 | 3.0 / 3.3 ms | 21.7 / 22.4 ms | 24.8 / 25.5 / 27.7 ms |
| LFW 250×250 (300 images) | 4 | 2.0 / 2.2 ms | 8.0 / 8.4 ms | 10.0 / 10.5 / 11.4 ms |
| Phone-size photos (5 distinct images from this repo, 300×390 to 2048×1365, 200 samples) | 1 | 10.4 / 60.9 ms | 21.4 / 24.2 ms | 31.7 / 85.1 / 89.4 ms |
| same | 4 | 5.2 / 25.2 ms | 8.6 / 10.5 ms | 13.5 / 34.6 / 45.5 ms |

The inference process peaked at about 340 MB RSS. Detection time scales with image size (the
detector input is capped at `DOPPEL_DETECT_MAX_SIDE`); embedding is a fixed-size crop.

## 5. End to end over HTTP with a real 15 k-face gallery

All of LFW imported through the worker with the real models, then selfie searches over loopback
(API and worker as two processes, `DOPPEL_INFERENCE_INSTANCES=2`, 1 OpenCV thread each, worker
concurrency 2):

| Measurement | Result |
| --- | --- |
| Import: 13 233 files (≈ 250 px JPEGs) | **281 s wall** (about 47 files/s) including worker start-up; `completed`, 0 failed |
| Outcome | 15 054 faces indexed (every eligible face of every photo), 16 `no_face` (quality gate), 2 `duplicate` (identical files, sha256 dedupe), 0 failed |
| API restart afterwards | index loaded from the snapshot (`rebuilt_on_load: false`), 15 054 vectors |

`POST /v1/faces/search`, LOOKALIKE, top-10, 300 requests per level, client-measured wall-clock
including multipart upload and JSON encoding:

| Client concurrency | p50 | p95 | p99 | Throughput | Responses |
| --- | --- | --- | --- | --- | --- |
| 1 | 35.7 ms | 39.8 ms | 48.4 ms | 28.9 req/s | 279 × 200, 21 × 422 |
| 2 | 33.9 ms | 38.7 ms | 42.0 ms | 61.2 req/s | 279 × 200, 21 × 422 |
| 4 | 54.0 ms | 61.7 ms | 100.4 ms | 76.1 req/s | 279 × 200, 21 × 422 |

* 7 % of the query photos (21 of 300) were rejected with `multiple_faces_ambiguous`: LFW crops
  often contain a second face of at least half the area of the subject, and the service refuses to
  guess (`face_index` resolves it). These fast failures are included in the latency numbers.
* In an earlier run with default settings, the per-principal search rate limit (30 requests/minute)
  answered 429 for most of the burst, which confirms the limiter works; the table above is from a
  run with the limits raised to 1 000 000 per minute.
* With 2 inference instances on 4 cores, concurrency 4 queues behind the pool, hence the higher
  latency. Raise `DOPPEL_INFERENCE_INSTANCES` only together with core count.

## What was not measured

* **Docker images.** No Docker was available on the build machine; the Dockerfile/Compose files are
  untested, and so are image size and container start-up time.
* **Gallery scale beyond 100 000 vectors** (for example 1 M), and memory behaviour there.
* **Real selfies and real galleries.** All accuracy numbers are on LFW with synthetic degradations;
  nothing was measured on phone selfies, other lighting/pose/age conditions, or masks other than a
  synthetic band. Demographic differences in accuracy (a well-documented issue in face recognition)
  were **not** measured; LFW is demographically skewed.
* **LOOKALIKE quality.** No human raters were available, so no judgement-based score exists; only
  the process and scorer are provided ([EVALUATION.md](EVALUATION.md#2-lookalike-human-judgement-process)).
* **dlib** on blur, noise, bright and mask conditions, and dlib latency.
* **Multi-process / multi-host throughput**, GPU execution, and long-running (soak) behaviour.

## Reproduce

```bash
# accuracy (needs LFW unpacked as /data/lfw/<person>/<image>.jpg and models/*.onnx)
python scripts/evaluate.py extract --dataset /data/lfw --dataset-name LFW --workers 4 --out eval_out/lfw_sface_clean.npz
python scripts/evaluate.py extract --dataset /data/lfw --dataset-name LFW --workers 4 --no-gate --out eval_out/lfw_sface_ng_none.npz
python scripts/evaluate.py extract --dataset /data/lfw --workers 4 --no-gate --degrade down3 --out eval_out/lfw_sface_ng_down3.npz
python scripts/evaluate.py identify --gallery eval_out/lfw_sface_clean.npz --write-calibration eval_out/calibration.json --out eval_out/identify.json
python scripts/evaluate.py identify --gallery eval_out/lfw_sface_ng_none.npz --probes eval_out/lfw_sface_ng_down3.npz
# dlib baseline: same commands with  /opt/dlibenv/bin/python ... extract --pipeline dlib   (requirements-dlib.txt)

# index / service / inference
python scripts/benchmark_index.py index   --sizes 10000 100000 --out bench_out/index.json
python scripts/benchmark_index.py service --sizes 10000 100000 --out bench_out/service.json
python scripts/benchmark_index.py inference --images /data/lfw --count 300 --threads 1 4

# end to end: start API + worker, import, then
python -m app.cli import /data/lfw --recursive --wait
DOPPEL_TOKEN=... python scripts/bench_http.py --url http://127.0.0.1:8000 --images /data/lfw --count 300 --concurrency 1 2 4
```

The LFW seed, split and sampling are fixed (`--seed 7`), so the accuracy numbers reproduce exactly
for identical embeddings; latency numbers will differ on other hardware.
