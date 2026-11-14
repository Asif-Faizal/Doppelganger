# Model provenance and licensing

This file records where the model weights come from, under which terms they are published, and
what is **not** known. It is a good-faith engineering record, not legal advice. Read the
"Unresolved" section before deploying outside a private, non-commercial setting.

## Models used in production

| Role | Model | File | sha256 | Size |
| --- | --- | --- | --- | --- |
| Detector | OpenCV Zoo YuNet `2023mar` | `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | 232 589 B |
| Embedder | OpenCV Zoo SFace `2021dec` | `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | 38 696 353 B |

Both files are pinned in [`models/manifest.json`](../models/manifest.json) to exact
`opencv/opencv_zoo` commits (YuNet `f12e12798e8314f7c074a6656816c048dcc95b7a`, SFace
`ba91a3b91d00d76e86540d4013f944bd6b514e39`). The sha256 and size values equal the Git LFS pointer
`oid`/`size` published in that repository at those commits, and they were re-computed on the
downloaded files. The service verifies the sha256 before loading a file and refuses to start (or
to ingest/search) on a mismatch. Weights are never fetched at startup or at request time; the only
download path is the explicit operator command `python -m app.cli models fetch`.

### YuNet (face detector)

* Source: `opencv_zoo/models/face_detection_yunet`; derived from
  [`ShiqiYu/libfacedetection.train`](https://github.com/ShiqiYu/libfacedetection.train).
* License of the zoo directory: **MIT** (copyright Shiqi Yu). Upstream training repository:
  **BSD-3-Clause**.
* Training task: WIDER Face (face detection only; no identity labels are involved).

### SFace (face embedder)

* Source: `opencv_zoo/models/face_recognition_sface`; upstream method/code
  [`zhongyy/SFace`](https://github.com/zhongyy/SFace) (MobileFaceNet trained with the SFace loss);
  ONNX conversion by the OpenCV Zoo contributors.
* License of the zoo directory: **Apache-2.0**.
* Output: 128-d vector, compared with cosine similarity after L2 normalisation (vendor-suggested
  verification threshold 0.363, which this project treats as *uncalibrated*, see
  [BENCHMARKS.md](BENCHMARKS.md)).

## Unresolved licensing questions (SFace)

1. **No upstream license file.** The upstream `zhongyy/SFace` repository publishes no `LICENSE`.
   The Apache-2.0 grant is the one stated by the OpenCV Zoo for the redistributed ONNX file; this
   project could not independently confirm that the upstream authors granted that license for the
   weights.
2. **Training data of this exact ONNX is not stated.** The SFace paper reports experiments on
   CASIA-WebFace, VGGFace2 and MS-Celeb-1M. Which dataset produced the released file is not
   documented by the zoo. MS-Celeb-1M was withdrawn by its publisher, and several of these
   datasets were scraped without per-image consent. Whether weights trained on them can be used
   commercially, or in a given jurisdiction (for example under biometric-privacy law), is a legal
   question this repository cannot answer.
3. **Evaluation overlap.** Because the training set is unknown, identity overlap between the
   training data and public benchmarks (for example LFW) cannot be ruled out; the measured numbers
   in [BENCHMARKS.md](BENCHMARKS.md) may therefore be optimistic.

If any of this is unacceptable for your use, swap the embedder: the pipeline is isolated behind
`FaceEmbedder` (`app/ml/base.py`), the active `model_id` is stored with every embedding, and the
index is per model, so a replacement can be provisioned and the gallery re-processed with
`python -m app.cli reprocess --all` without touching the old vectors.

## Models considered and excluded

| Candidate | Why it is not used |
| --- | --- |
| InsightFace model packs (`buffalo_l`, ArcFace/`w600k_r50`, ...) | The pretrained weights are released for **non-commercial research use only**; the code license (MIT) does not cover them. |
| FaceNet (`davidsandberg/facenet`, `facenet-pytorch`) | Needs a large deep-learning stack (TensorFlow/PyTorch) for a CPU-first service; pretrained weights are trained on VGGFace2 / CASIA-WebFace, with the same data-provenance questions as above and no clear advantage over SFace at this size. The previous README claimed FaceNet but the code never used it. |
| Cloud APIs (AWS Rekognition, Azure Face, Google, ...) | Excluded by the brief: no paid or cloud services. |

## Legacy baseline (evaluation and migration only)

The original code used dlib's HOG detector and `dlib_face_recognition_resnet_model_v1`
(shipped through the `face_recognition_models` package). It is kept as an **optional, non-production
adapter** (`app/ml/dlib_adapter.py`, `requirements-dlib.txt`) so that existing 128-d vectors can be
interpreted and so the new pipeline can be compared against the old one.

* The dlib library is Boost-licensed; its author states the pretrained model weights are released
  into the public domain.
* The weights were trained on a mix of datasets that includes VGG Face, FaceScrub and images
  scraped from the web by the author. The same data-provenance caveat applies.
* dlib embeddings use a **euclidean** metric (distance threshold about 0.6) and are **not**
  comparable with SFace vectors; they are never mixed in one index (see
  [MIGRATION_FROM_DLIB.md](MIGRATION_FROM_DLIB.md)).

## Other runtime dependencies

All runtime dependencies are pinned in `requirements.txt` and are permissively licensed (BSD, MIT,
Apache-2.0, HPND for Pillow). `fastapi` depends on `opentelemetry-api`, which is the API package
only: no exporter or SDK is installed, so nothing is transmitted. The service itself makes no
outbound network calls.

## Evaluation data

The measurements in `docs/BENCHMARKS.md` use the public **Labeled Faces in the Wild** (LFW) archive,
downloaded from the figshare mirror used by scikit-learn and checked against its published sha256
(`055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`). LFW is distributed for
non-commercial research evaluation; the images are **not** redistributed in this repository and
are not used to train or enrol anything in a deployed service. Only derived aggregate metrics are
published.
