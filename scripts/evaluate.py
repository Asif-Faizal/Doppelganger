#!/usr/bin/env python
"""Offline evaluation harness: extract embeddings, measure identification, calibrate thresholds.

Subcommands
  extract          detect + embed every image of a `root/<identity>/<image>` tree into an .npz
  identify         open-set 1:N identification (top-k, FNIR/FPIR), verification (ROC), photo-gallery
                   retrieval, thresholds calibrated on a held-out identity split, optional
                   calibration.json for the service
  lookalike-sheet  build a blind, pooled judging sheet for the LOOKALIKE human-judgement process
  lookalike-score  score model rankings against the filled-in sheet (nDCG/precision/recall/MRR)

Nothing here downloads data or models. Run `extract` with the production interpreter for
`--pipeline sface` and with a dlib-enabled interpreter for `--pipeline dlib`.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import platform
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evalkit

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --- degradations (applied to probe images only) -------------------------------------------------


def _jpeg(img: Any, quality: int) -> Any:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _down(img: Any, factor: int) -> Any:
    import cv2

    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA
    )
    return small


def _blur(img: Any, sigma: float) -> Any:
    import cv2

    return cv2.GaussianBlur(img, (0, 0), sigma)


def _gamma(img: Any, gamma: float) -> Any:
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return lut[img]


def _noise(img: Any, sigma: float) -> Any:
    rng = np.random.default_rng(0)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _occlude_lower(img: Any) -> Any:
    out = img.copy()
    h, w = out.shape[:2]
    out[int(h * 0.60) : int(h * 0.80), int(w * 0.28) : int(w * 0.72)] = (
        0  # mask-like band on LFW crops
    )
    return out


DEGRADATIONS: dict[str, Callable[[Any], Any]] = {
    "none": lambda im: im,
    "jpeg10": lambda im: _jpeg(im, 10),
    "down2": lambda im: _down(im, 2),
    "down3": lambda im: _down(im, 3),
    "blur2": lambda im: _blur(im, 2.0),
    "blur4": lambda im: _blur(im, 4.0),
    "dark": lambda im: _gamma(im, 3.0),
    "bright": lambda im: _gamma(im, 0.35),
    "noise20": lambda im: _noise(im, 20.0),
    "occlude": _occlude_lower,
}

# --- extraction ----------------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _init_worker(
    pipeline: str, models_dir: str, threads: int, gate: bool, select: str, degrade: str
) -> None:
    import cv2

    cv2.setNumThreads(threads)
    _STATE["pipeline"] = pipeline
    _STATE["gate"] = gate
    _STATE["select"] = select
    _STATE["degrade"] = DEGRADATIONS[degrade]
    if pipeline == "sface":
        from app.config import Settings
        from app.ml.opencv_models import build_sface_pipeline_parts
        from app.ml.pipeline import FacePipeline

        settings = Settings(models_dir=Path(models_dir), _env_file=None)  # type: ignore[call-arg]
        det, emb, info = build_sface_pipeline_parts(
            Path(models_dir), max_side=settings.detect_max_side, threads=threads
        )
        _STATE["pipe"] = FacePipeline(det, emb, info, settings)
    else:
        from app.ml.dlib_adapter import build_dlib_pipeline_parts

        det, emb, info = build_dlib_pipeline_parts()
        _STATE["det"], _STATE["emb"], _STATE["info"] = det, emb, info


def _choose(faces: list[Any], shape: tuple[int, ...], select: str) -> Any:
    if select == "largest":
        return max(faces, key=lambda f: f.face.area if hasattr(f, "face") else f.area)
    h, w = shape[:2]

    def dist(f: Any) -> float:
        face = f.face if hasattr(f, "face") else f
        x, y, fw, fh = face.bbox
        return float((x + fw / 2 - w / 2) ** 2 + (y + fh / 2 - h / 2) ** 2)

    return min(faces, key=dist)


def _extract_one(path: str) -> tuple[str, np.ndarray[Any, Any] | None, str, float, float]:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return path, None, "unreadable", 0.0, 0.0
    img = _STATE["degrade"](img)
    t0 = time.perf_counter()
    if _STATE["pipeline"] == "sface":
        pipe = _STATE["pipe"]
        analysis = pipe.detect(img)
        t1 = time.perf_counter()
        pool = analysis.candidates if not _STATE["gate"] else analysis.eligible
        if not pool:
            reason = (
                "no_face"
                if not analysis.candidates
                else "quality:"
                + ",".join(sorted({r for c in analysis.candidates for r in c.reasons}))
            )
            return path, None, reason, t1 - t0, 0.0
        cand = _choose(pool, img.shape, _STATE["select"])
        vec = pipe.embed(img, cand)
        return path, vec, "ok", t1 - t0, time.perf_counter() - t1
    faces = _STATE["det"].detect(img)
    t1 = time.perf_counter()
    if not faces:
        return path, None, "no_face", t1 - t0, 0.0
    face = _choose(faces, img.shape, _STATE["select"])
    vec = _STATE["emb"].embed(img, face)
    return path, vec, "ok", t1 - t0, time.perf_counter() - t1


def cmd_extract(a: argparse.Namespace) -> int:
    root = Path(a.dataset)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if a.list:
        wanted = set(Path(a.list).read_text().split())
        images = [p for p in images if str(p.relative_to(root)) in wanted]
    if a.max_images and len(images) > a.max_images:
        rng = random.Random(a.seed)
        images = sorted(rng.sample(images, a.max_images))
    if not images:
        print("no images found", file=sys.stderr)
        return 2
    print(
        f"extracting {len(images)} images, pipeline={a.pipeline}, "
        f"workers={a.workers}, degrade={a.degrade}"
    )
    wall0 = time.perf_counter()
    init = (a.pipeline, a.models_dir, a.threads, not a.no_gate, a.face_select, a.degrade)
    paths = [str(p) for p in images]
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers, _init_worker, init) as pool:
            results = list(pool.imap(_extract_one, paths, chunksize=16))
    else:
        _init_worker(*init)
        results = [_extract_one(p) for p in paths]
    wall = time.perf_counter() - wall0

    names = np.array([str(p.relative_to(root)) for p in images])
    labels = np.array([p.relative_to(root).parts[0] for p in images])
    valid = np.array([r[1] is not None for r in results])
    dim = next(r[1].shape[0] for r in results if r[1] is not None)
    emb = np.zeros((len(results), dim), dtype=np.float32)
    for i, r in enumerate(results):
        if r[1] is not None:
            emb[i] = r[1]
    reasons = np.array([r[2] for r in results])
    detect_s = np.array([r[3] for r in results])
    embed_s = np.array([r[4] for r in results])
    if a.pipeline == "sface":
        from app.ml.manifest import load_manifest  # noqa: F401
        from app.ml.opencv_models import build_sface_pipeline_parts

        info = build_sface_pipeline_parts(Path(a.models_dir), max_side=640, threads=1)[2]
    else:
        from app.ml.dlib_adapter import build_dlib_pipeline_parts

        info = build_dlib_pipeline_parts()[2]
    meta = {
        "model_id": info.model_id,
        "metric": "cosine" if info.metric == "cosine" else "l2",
        "dataset": a.dataset_name or root.name,
        "degrade": a.degrade,
        "quality_gate": (not a.no_gate) and a.pipeline == "sface",
        "face_select": a.face_select,
        "images": len(images),
        "usable": int(valid.sum()),
        "wall_seconds": round(wall, 1),
        "workers": a.workers,
        "threads_per_worker": a.threads,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        names=names,
        labels=labels,
        valid=valid,
        emb=emb,
        reasons=reasons,
        detect_s=detect_s,
        embed_s=embed_s,
        meta=json.dumps(meta),
    )
    fta: dict[str, int] = {}
    for r in reasons.tolist():
        if r != "ok":
            fta[r] = fta.get(r, 0) + 1
    print(json.dumps({**meta, "failures_to_acquire": fta}, indent=2))
    return 0


# --- identification report -----------------------------------------------------------------------


def _load(path: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"rate": k / n if n else None, "count": k, "of": n, "ci95": [round(lo, 4), round(hi, 4)]}


def cmd_identify(a: argparse.Namespace) -> int:
    g = _load(a.gallery)
    p = _load(a.probes) if a.probes else g
    if not np.array_equal(g["names"], p["names"]):
        print(
            "gallery and probe files must cover the same images in the same order", file=sys.stderr
        )
        return 2
    metric = g["meta"]["metric"]
    labels = g["labels"]
    valid_g = g["valid"]
    cal_ids, test_ids = evalkit.split_identities(labels, a.seed, a.cal_fraction)

    def run(identities: set[str], seed: int) -> tuple[evalkit.Split, np.ndarray[Any, Any]]:
        split = evalkit.build_split(labels, valid_g, identities, seed, a.enrolled_fraction)
        is_gallery = np.zeros(len(labels), dtype=bool)
        is_gallery[split.gallery] = True
        mix = np.where(is_gallery[:, None], g["emb"], p["emb"])
        pv = p["valid"]
        mated_ok = pv[split.mated]
        non_ok = pv[split.non_mated]
        split = evalkit.Split(
            split.gallery,
            split.mated[mated_ok],
            split.non_mated[non_ok],
            split.mated_fta + int((~mated_ok).sum()),
            split.non_mated_fta + int((~non_ok).sum()),
        )
        return split, mix

    cal_split, cal_emb = run(cal_ids, a.seed + 1)
    test_split, test_emb = run(test_ids, a.seed + 2)
    _, _, ns_c = evalkit.top1_scores(cal_emb, labels, cal_split, metric)
    ms_t, ok_t, ns_t = evalkit.top1_scores(test_emb, labels, test_split, metric)

    report: dict[str, Any] = {
        "model_id": g["meta"]["model_id"],
        "dataset": g["meta"]["dataset"],
        "probe_degradation": p["meta"]["degrade"],
        "quality_gate": g["meta"]["quality_gate"],
        "metric": metric,
        "protocol": {
            "seed": a.seed,
            "calibration_identities": len(cal_ids),
            "test_identities": len(test_ids),
            "enrolled_fraction": a.enrolled_fraction,
            "gallery_size_test": len(test_split.gallery),
            "mated_probes_test": int(len(test_split.mated) + test_split.mated_fta),
            "non_mated_probes_test": int(len(test_split.non_mated) + test_split.non_mated_fta),
            "failures_to_acquire_test": {
                "mated": test_split.mated_fta,
                "non_mated": test_split.non_mated_fta,
            },
            "note": "one enrolment photo per enrolled identity; thresholds are fixed on the "
            "identity-disjoint calibration partition and only then applied to the test partition",
        },
        "closed_set_test": evalkit.closed_set_topk(
            test_emb, labels, test_split, metric, (1, 5, 10)
        ),
    }

    total_m = len(ms_t) + test_split.mated_fta

    def point(t: float) -> tuple[Any, Any]:
        fnir, fpir = evalkit.fnir_fpir(ms_t, ok_t, ns_t, np.array([t]), test_split.mated_fta)
        return (
            _rate(round(float(fnir[0]) * total_m), total_m),
            _rate(round(float(fpir[0]) * len(ns_t)), len(ns_t)),
        )

    operating: list[dict[str, Any]] = []
    for target in a.target_fpir:
        t = evalkit.threshold_for_fpir(ns_c, target)
        fnir_r, fpir_r = point(t)
        operating.append(
            {
                "target_fpir_on_calibration": target,
                "threshold": round(t, 5),
                "test_fnir": fnir_r,
                "test_fpir": fpir_r,
            }
        )
    for t in a.fixed_thresholds:
        fnir_r, fpir_r = point(t)
        operating.append({"fixed_threshold": t, "test_fnir": fnir_r, "test_fpir": fpir_r})
    report["open_set_operating_points_test"] = operating

    gen_c, imp_c = evalkit.pair_scores(
        g["emb"], labels, valid_g, cal_ids, metric, a.seed + 3, n_impostor=a.impostor_pairs
    )
    pair_valid = p["valid"] & g["valid"]
    gen_t, imp_t = evalkit.pair_scores(
        p["emb"], labels, pair_valid, test_ids, metric, a.seed + 4, n_impostor=a.impostor_pairs
    )
    report["verification_test"] = {
        "genuine_pairs": len(gen_t),
        "impostor_pairs": len(imp_t),
        **evalkit.roc_summary(gen_t, imp_t),
    }
    far_table = {}
    for far in (1e-2, 1e-3, 1e-4, 1e-5):
        if len(imp_c) * far >= 1:
            t = float(np.quantile(imp_c, 1 - far))
            far_table[f"{far:g}"] = {
                "threshold": round(t, 5),
                "test_far": float((imp_t >= t).mean()),
                "test_tar": float((gen_t >= t).mean()),
            }
    report["verification_thresholds_from_calibration"] = far_table

    report["photo_gallery_retrieval_test"] = evalkit.leave_one_out_retrieval(
        np.where(p["valid"][:, None], p["emb"], g["emb"]),
        labels,
        pair_valid & np.isin(labels, list(test_ids)),
        metric,
        max_probes=a.max_probes,
        seed=a.seed,
    )

    if a.write_calibration:
        if metric != "cosine":
            print("calibration.json is only written for cosine models", file=sys.stderr)
        else:
            if len(imp_c) * a.far < 1:
                print("not enough impostor pairs for the requested FAR", file=sys.stderr)
                return 2
            ca, cb = evalkit.fit_logistic(gen_c, imp_c)
            thr = float(np.quantile(imp_c, 1 - a.far))
            Path(a.write_calibration).write_text(
                json.dumps(
                    {
                        "model_id": g["meta"]["model_id"],
                        "dataset": f"{g['meta']['dataset']} "
                        f"(identity-disjoint calibration partition, seed {a.seed})",
                        "logistic": {"a": ca, "b": cb},
                        "same_person_threshold": round(thr, 5),
                        "same_person_threshold_target_far": a.far,
                        "n_pairs": int(len(gen_c) + len(imp_c)),
                        "prior": "class-balanced (1:1); not the prior of any real gallery",
                        "note": "Fitted on public research data. Re-run scripts/evaluate.py on "
                        "photos "
                        "that look like your gallery before relying on it.",
                    },
                    indent=2,
                )
                + "\n"
            )
            report["calibration_written"] = {
                "path": a.write_calibration,
                "a": ca,
                "b": cb,
                "threshold": thr,
            }

    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


# --- LOOKALIKE human-judgement process -----------------------------------------------------------


def _rank_for(npz: dict[str, Any], query: str, k: int) -> list[str]:
    names = npz["names"].tolist()
    qi = names.index(query)
    if not npz["valid"][qi]:
        return []
    idx = np.flatnonzero(npz["valid"])
    sim = evalkit.similarity(npz["emb"][qi : qi + 1], npz["emb"][idx], npz["meta"]["metric"])[0]
    order = np.argsort(-sim)
    out = [names[int(idx[o])] for o in order if int(idx[o]) != qi]
    return out[:k]


def cmd_lookalike_sheet(a: argparse.Namespace) -> int:
    runs = [_load(f) for f in a.npz]
    names = runs[0]["names"].tolist()
    if any(r["names"].tolist() != names for r in runs[1:]):
        print("all npz files must cover the same images", file=sys.stderr)
        return 2
    rng = random.Random(a.seed)
    usable = [n for n, v in zip(names, runs[0]["valid"].tolist(), strict=True) if v]
    queries = rng.sample(usable, min(a.queries, len(usable)))
    rows: list[dict[str, str]] = []
    key: list[dict[str, Any]] = []
    for q in queries:
        pooled: dict[str, dict[str, int]] = {}
        for run in runs:
            for rank, cand in enumerate(_rank_for(run, q, a.depth), start=1):
                pooled.setdefault(cand, {})[run["meta"]["model_id"]] = rank
        cands = sorted(pooled)
        rng.shuffle(cands)
        for c in cands:
            rows.append({"query": q, "candidate": c, "grade": ""})
            key.append({"query": q, "candidate": c, "ranks": pooled[c]})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query", "candidate", "grade"])
        w.writeheader()
        w.writerows(rows)
    Path(a.out + ".key.json").write_text(json.dumps(key, indent=1))
    print(
        f"wrote {len(rows)} blind pairs for {len(queries)} queries to {a.out} "
        f"(key: {a.out}.key.json)"
    )
    return 0


def cmd_lookalike_score(a: argparse.Namespace) -> int:
    judg: dict[str, dict[str, int]] = {}
    with open(a.judgments, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["grade"].strip() == "":
                continue
            judg.setdefault(row["query"], {})[row["candidate"]] = int(row["grade"])
    report = {}
    for f in a.npz:
        run = _load(f)
        rankings = {q: _rank_for(run, q, a.depth) for q in judg}
        report[run["meta"]["model_id"]] = evalkit.ranking_metrics(
            rankings, judg, relevant_grade=a.relevant_grade
        )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--dataset", required=True, help="root/<identity>/<image>")
    e.add_argument("--dataset-name")
    e.add_argument("--pipeline", choices=["sface", "dlib"], default="sface")
    e.add_argument("--models-dir", default="models")
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--max-images", type=int, default=0)
    e.add_argument("--list", help="file of relative image paths to restrict to")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--degrade", choices=sorted(DEGRADATIONS), default="none")
    e.add_argument("--no-gate", action="store_true", help="sface only: disable the quality gate")
    e.add_argument("--face-select", choices=["center", "largest"], default="center")
    e.set_defaults(fn=cmd_extract)

    i = sub.add_parser("identify")
    i.add_argument("--gallery", required=True, help="clean npz (enrolment images, calibration)")
    i.add_argument("--probes", help="npz of the same images, degraded, used for probes")
    i.add_argument("--seed", type=int, default=7)
    i.add_argument(
        "--cal-fraction",
        type=float,
        default=0.5,
        help="share of identities used for calibration; 0.5 keeps the calibration gallery the "
        "same size as the test gallery (FPIR grows with gallery size)",
    )
    i.add_argument("--fixed-thresholds", type=float, nargs="*", default=[0.363])
    i.add_argument("--enrolled-fraction", type=float, default=0.8)
    i.add_argument("--target-fpir", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    i.add_argument("--impostor-pairs", type=int, default=1_000_000)
    i.add_argument("--max-probes", type=int, default=3000)
    i.add_argument("--write-calibration")
    i.add_argument(
        "--far", type=float, default=1e-4, help="pair-level FAR for same_person_threshold"
    )
    i.add_argument("--out")
    i.set_defaults(fn=cmd_identify)

    s = sub.add_parser("lookalike-sheet")
    s.add_argument("--npz", nargs="+", required=True)
    s.add_argument("--queries", type=int, default=50)
    s.add_argument("--depth", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_lookalike_sheet)

    c = sub.add_parser("lookalike-score")
    c.add_argument("--npz", nargs="+", required=True)
    c.add_argument("--judgments", required=True)
    c.add_argument("--depth", type=int, default=10)
    c.add_argument("--relevant-grade", type=int, default=2)
    c.set_defaults(fn=cmd_lookalike_score)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
