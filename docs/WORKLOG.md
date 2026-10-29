<!-- ZOVIVE-EDGE WORKLOG & AI CONTEXT HANDOFF -->
# Worklog: Day 1 Hailo-8L deployment (28-29 Oct 2026)

Every claim below is marked:

- **verified**: checked in this repository, on the laptop, or on the Pi by the
  owner; the evidence is named.
- **reported**: stated in earlier notes and not yet checked.
- **open**: known not to be done.

Hailo performance counts as verified only when it was measured on the real Pi.

## 1. Status

| Item | Status |
|---|---|
| Root cause of the bad detections (sloth_bear 1.00 on every frame) | **verified** on the laptop; see section 2 |
| HEF the Pi runs | **verified** by the owner on 29 Oct: R1, `b999f784...c1f1`, at `/opt/zovive/models/detector.hef`; see section 3 |
| `zovive-detect` running on the live CP Plus RTSP feed | **verified** by the owner on 29 Oct (PID 863182 at the time) |
| R1 validation | B0 and B1 **passed**; AP50 **failed** on elephant by 0.2 points and the owner **accepted** it as an exception; gates C and D **open**; see section 6 |
| Live elephant (0.67-0.76) and gaur detections, no latency issues | **reported** by the owner; whether elephants were in view and the FPS are not recorded; see section 6 |
| Split-head DFL decoder and per-class NMS (`73d8ea4`, `84652ff` on `develop`) | **verified** in the repository; see section 4 |
| `/var/lib/zovive` and `/var/log/zovive` re-owned to `zovive1` to fix `PermissionError: [Errno 13]` | **reported**; this was a production change on the Pi |
| 85.37 ms end-to-end, 11.7 FPS | **reported** for the overnight HEF, which R1 has replaced; not reproduced; see section 5 |

## 2. Root cause

The HEF that produced the failures (`D:\BEST_MODELS_FINAL\detector.hef`,
sha256 `2d460f58...c967`) was compiled **without its model script**. The
`normalization([0,0,0],[255,255,255])` line never reached the compiler, so
the network received 0-255 pixels where it was trained on 0-1.

- Evidence: the build's DFC logs (copies in
  `zovive-ml/hailo_builds/det150e_h8l_r1/evidence/`) have no "Loading model
  script" line, and the float model fed 0-255 input reproduces the
  sloth_bear 1.00 output on the laptop.
- Contributing factors:
  - The build used optimization level 0.
  - `calib_500.npy` holds only deer and gaur (sampled 48 frames), with no
    other species, no empty frames and no letterbox bars.

The original notes blamed a stride-32 quantization underflow in `conv105`.
`conv105` is the stride-32 class head; its saturated output is a symptom of
the missing normalization, not a separate cause. Recalibration alone does
not fix it. The normalization line in the model script does.

## 3. Detector HEFs by hash

Identify a HEF by its sha256, never by its file name.

| sha256 | File (laptop copy) | Build | Status |
|---|---|---|---|
| `b999f7842b6bcc0326a57de06c66958f38d521310ac5e03a79bc4c4a52bfc1f1` | `zovive-ml/hailo_builds/det150e_h8l_r1/hef/zovive_det150e_h8l_r1.hef`, 48,530,992 B, 29 Oct 05:21 UTC | R1: compiled from optimized HAR `31fc382e...e35f`; normalization, optimization level 1, `calib_1024` | **deployed** (section 1); accepted AP50 exception (section 6) |
| `7aa106253c550b62edcc8aff3922bdf89f0d9ece71543909b15e0366fc3d4993` | `ZOVIVE_FINAL_150E_BEST.hef`, 54,175,048 B, 29 Oct 02:38 IST | normalization present, optimization level 0, `calib_500` (deer and gaur only) | **not validated**; replaced on the Pi by R1 |
| `2d460f582a0a64cf68bd3c9731f42f1725ef6dfcfc8a4f08e9cbf4374d6cc967` | `detector.hef`, 48,505,071 B, 28 Oct | no model script (no /255), optimization level 0, `calib_500` | **faulty**; must not run in production |

All three come from `ZOVIVE_FINAL_150E_BEST.onnx` (sha256 `7fa08795...1224`;
the R1 copy is hash-verified, the other two carry its network name). That
ONNX differs from this repository's `models/detector.onnx`
(sha256 `404de609...8652`), and this repository's `models/manifest.json`
records no `sha256_hef`. **Open:** record the deployed ONNX and HEF hashes in
the manifest (a production change that needs the owner's approval).

## 4. Post-processing (verified in the repository)

`inference/yolo_postprocess.py` exports both decoders:

- `decode_split_heads`: Hailo. It reads six NHWC head outputs, gates
  candidates on the inverse-sigmoid logit threshold, rebuilds boxes from the
  DFL bins, then runs per-class NMS.
- `decode_yolo_detections`: ONNX, from the single `[1, 4 + classes, 8400]`
  output, with the same per-class NMS.

Known differences, not yet decided:

- `decode_split_heads` drops boxes 8 px or smaller in the 640 letterbox
  (`yolo_postprocess.py:193`); `decode_yolo_detections` does not. No test
  box is that small, so the R1 results below are unaffected, but the filter
  works against small, distant animals.
- Neither decoder drops boxes centred in the letterbox padding any more;
  the version in `6524fb9` did.

## 5. Performance on the Pi

R1: no FPS, latency, memory or temperature figures have been measured yet
(gate D). The owner reports no latency issues on the live feed.

The original notes report 85.37 ms end-to-end and 11.7 FPS for the overnight
`ZOVIVE_FINAL_150E_BEST.hef`. These figures were not reproduced, and neither
the command nor the frame source was recorded.

Acceptance (decision D5) needs all of the following, measured on the Pi with
the candidate HEF:

- At least 5 FPS (10 FPS stretch), p50 and p95 latency, memory and
  temperature.
- Parity with the ONNX reference:
  - matched >= 0.90
  - same species >= 0.97
  - mean IoU >= 0.85
  - mean score difference <= 0.05
- AP50 drop of at most 3 points per class.
- Elephant-to-sloth-bear errors and empty-frame false positives no worse
  than the ONNX baseline.

## 6. R1 rebuild (deployed)

Folder: `zovive-ml/hailo_builds/det150e_h8l_r1/`. Commands, hashes, logs and
full results are in its `BUILD.md`.

- The model script loads the normalization line, sets optimization level 1
  and compression level 0, and uses 256 calibration entries.
- The calibration set comes from the train split only: 128 images per
  species plus 128 empty frames, letterboxed with this repository's
  `letterbox()`.
- Compile: `hailo compiler har/02_optimized.har --hw-arch hailo8l
  --output-dir hef`, default compiler settings, 29 Oct 04:57-05:21 UTC, six
  contexts. Log: `hef/hailo_sdk.core.log`.

### Gates

| Gate | Where | What | Result |
|---|---|---|---|
| B0 | laptop | float emulator vs ONNX, 100 test images | **passed**: identical outputs |
| B1 | laptop | quantized emulator vs ONNX, 100 test images | **passed**: matched 0.961, same species 1.000, mean IoU 0.967, mean score difference 0.047 |
| AP50 | laptop | quantized emulator vs ONNX vs ground truth, full test split (679 images, 730 boxes, 104 empty) | **failed** on elephant (3.2 points; limit 3.0); accepted as an exception |
| C | Pi | parity, using `pi/dump_detections.py` | **open** |
| D | Pi | FPS, p50/p95 latency, memory, temperature | **open** |

### AP50 findings

| Class | GT boxes | AP50 ONNX | AP50 R1 | Drop (points) | Recall at 0.45, ONNX to R1 |
|---|---|---|---|---|---|
| deer | 52 | 0.970 | 0.973 | -0.3 | 0.96 to 0.94 |
| gaur | 114 | 0.959 | 0.957 | 0.2 | 0.90 to 0.89 |
| leopard | 50 | 0.954 | 0.962 | -0.8 | 0.94 to 0.94 |
| elephant | 156 | 0.875 | 0.843 | **3.2** | 0.81 to 0.81 |
| sloth_bear | 55 | 0.951 | 0.925 | 2.6 | 0.87 to 0.78 |
| tiger | 105 | 0.891 | 0.889 | 0.2 | 0.84 to 0.79 |
| wild_boar | 198 | 0.959 | 0.935 | 2.4 | 0.91 to 0.88 |
| mAP50 | 730 | 0.937 | 0.926 | 1.1 | |

- Empty test images with a detection at 0.45: ONNX 1 of 104, R1 0 of 104.
- Elephant boxes called sloth bear at 0.45: 1 for both.

**Accepted exception (owner, 29 Oct 2026):** the elephant AP50 drop of 3.2
points, 0.2 over the 3.0 limit, is accepted as an operational compromise,
given the two-day deadline and zero detections on the 104 empty test images.
R1b (16-bit output heads) and R1-GPU were not started.

### Known limitations to watch on the live feed

- At the deployed 0.45 threshold, sloth bear recall falls from 0.87 to 0.78
  and tiger from 0.84 to 0.79, so some animals the ONNX would report are
  missed.
- Elephant precision falls from 0.93 to 0.89: more wild boar, tiger and
  sloth bear boxes are labelled elephant.
- The empty test images are not this camera's scenes. An empty desk frame
  (`test_detected.jpg` in the repository root, which also has boxes drawn on
  it) scores elephant 0.88 on R1 and 0.93 on the ONNX. That false positive
  comes from the model, not the quantization, and should be retested on a
  clean camera frame. (`real_wildlife.jpg`, an elephant photo, is correctly
  elephant: 0.95 on both.) Check the snapshots behind the live elephant
  detections (0.67-0.76) against what was actually in view.

## 7. Deployment layout

- HEF: `/opt/zovive/models/detector.hef` is R1 (**verified** by the owner).
- Service: `zovive-detect.service` (systemd) (**verified** running by the
  owner).
- Code: `/opt/zovive`, said to be a symlink to `/home/zovive1/zovive-edge`
  (**reported**). `docs/STORAGE_LAYOUT.md` describes `/opt/zovive` as the
  code location.
- Virtual environment: `/opt/zovive/.venv/` (**reported**).
- Branch on the Pi: `develop` (**reported**).
- Network: the Pi in use answers at 192.168.1.111. `docs/NETWORK_MAP.md`
  gives node 1's Pi as 192.168.1.101. The original notes also mention a
  192.168.137.x link on `eth0` with no internet gateway, and syncing through
  Git bundles over SSH. Settle the addressing in `NETWORK_MAP.md`.

## 8. Rules for agents and engineers

1. Keep the six-output split head. The compiler cuts the network before the
   DFL decode, so box reconstruction must stay in `decode_split_heads`.
2. Keep both `decode_split_heads` and `decode_yolo_detections` exported.
   `inference/hailo_inference.py` and `inference/onnx_inference.py` each
   import one, so removing either breaks `edge_main.py` at startup; the tests
   and the R1 parity tools import both.
3. Every Hailo model script must contain
   `normalization([0,0,0],[255,255,255])`, and the optimize log must show
   it loaded. A HEF built without it is the faulty build.
4. Identify a HEF by its sha256 (section 3) before reasoning about it.
5. Never deploy a HEF, replace `models/detector.hef` or edit
   `models/manifest.json` without the owner's explicit approval. Never deploy
   one that has not passed the gates in section 5, unless the owner records
   an exception here, as for R1 (section 6).
6. Report Hailo performance only from measurements on the real Pi, with the
   command, HEF hash and frame source recorded.
7. Never print `.env`, camera credentials or keys.

## 9. Roadmap

- [ ] **R1 gates C and D on the Pi.** Parity with the ONNX and FPS, p50/p95
  latency, memory and temperature for `b999f784...c1f1`.
- [ ] **Live-feed check.** Review the snapshots behind live elephant
  detections; collect real CP Plus frames for empty-scene evaluation.
- [ ] **Telemetry.** The heartbeat already carries the SoC temperature
  (`health_main.py`). Hailo chip temperature, PCIe throughput and frame-drop
  rate are not reported yet.
- [ ] **Species voting.** `pipeline/stage2_verifier.py` already runs from
  `edge_main.py`: a voted confidence below 0.55 becomes `unknown_animal`.
  What remains is tuning it on real camera frames and deciding whether it
  should gate alerts before MQTT and cloud sync.
- [ ] **SAHI tiling.** `inference/sahi_slicer.py` exists and is off
  (`sahi.enabled: false`). Evaluate it on high-resolution streams for
  distant animals, together with the 8 px box filter in section 4.
- [ ] **Exposure profiles.** Day and night switching in `camera_control`
  based on the average brightness inside detection boxes.
- [ ] **Offline storage and sync.** Validate queuing under
  `/var/lib/zovive/storage` and upload through `transfer_main.py` when the
  uplink returns.
<!-- END ZOVIVE-EDGE WORKLOG -->
