<!-- ZOVIVE-EDGE WORKLOG & AI CONTEXT HANDOFF -->
# Worklog: Day 1 Hailo-8L deployment (28-29 Oct 2026)

Every claim below is marked:

- **verified**: checked in this repository or on the laptop; the evidence is named.
- **reported**: stated in the original Day 1 notes and not yet checked.
- **open**: known not to be done.

Hailo performance counts as verified only when it was measured on the real Pi.

## 1. Status

| Item | Status |
|---|---|
| Root cause of the bad detections (sloth_bear 1.00 on every frame) | **verified** on the laptop; see section 2 |
| Which HEF the Pi runs | **open**; identify it by sha256 (section 3) with the read-only Pi check |
| Split-head DFL decoder and per-class NMS (`73d8ea4`, `84652ff` on `develop`) | **verified** in the repository; see section 4 |
| `zovive-detect` running under systemd with live RTSP from the camera at 192.168.1.250 | **reported** |
| `/var/lib/zovive` and `/var/log/zovive` re-owned to `zovive1` to fix `PermissionError: [Errno 13]` | **reported**; this was a production change on the Pi |
| 85.37 ms end-to-end, 11.7 FPS on `ZOVIVE_FINAL_150E_BEST.hef` | **reported**; not reproduced; see section 5 |
| "Clean outputs, no background hallucination" | **open**; no empty-frame or per-class AP figures exist for any HEF yet |
| Isolated rebuild R1 | **in progress**; see section 6 |

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
| `2d460f582a0a64cf68bd3c9731f42f1725ef6dfcfc8a4f08e9cbf4374d6cc967` | `detector.hef`, 48,505,071 B, 28 Oct | no model script (no /255), optimization level 0, `calib_500` | **faulty**; must not run in production |
| `7aa106253c550b62edcc8aff3922bdf89f0d9ece71543909b15e0366fc3d4993` | `ZOVIVE_FINAL_150E_BEST.hef`, 54,175,048 B, 29 Oct 02:38 IST | normalization present, optimization level 0, `calib_500` (deer and gaur only) | **not validated** against the ONNX reference |
| (pending) | `hailo_builds/det150e_h8l_r1/` | normalization, optimization level 1, `calib_1024` (train split, every species plus empty frames) | **in progress** |

The faulty HEF was built from `ZOVIVE_FINAL_150E_BEST.onnx`
(sha256 `7fa08795...1224`); the overnight HEF carries the same network name.
That ONNX differs from this repository's `models/detector.onnx`
(sha256 `404de609...8652`).

## 4. Post-processing (verified in the repository)

`inference/yolo_postprocess.py` exports both decoders:

- `decode_split_heads`: Hailo. It reads six NHWC head outputs, gates
  candidates on the inverse-sigmoid logit threshold, rebuilds boxes from the
  DFL bins, then runs per-class NMS.
- `decode_yolo_detections`: ONNX, from the single `[1, 4 + classes, 8400]`
  output, with the same per-class NMS.

Known differences, not yet decided:

- `decode_split_heads` drops boxes 8 px or smaller in the 640 letterbox
  (`yolo_postprocess.py:193`); `decode_yolo_detections` does not. Parity
  checks that compare the two count small boxes the ONNX keeps as HEF
  misses, and the filter works against small, distant animals.
- Neither decoder drops boxes centred in the letterbox padding any more;
  the version in `6524fb9` did.

## 5. Performance on the Pi

The original notes report 85.37 ms end-to-end and 11.7 FPS for
`ZOVIVE_FINAL_150E_BEST.hef` on a Raspberry Pi 5 with the Hailo-8L. These
figures have not been reproduced. Neither the command nor the frame source
was recorded. Acceptance (decision D5) needs all of the following, measured
on the Pi with the candidate HEF:

- At least 5 FPS (10 FPS stretch), p50 and p95 latency, memory and
  temperature.
- Parity with the ONNX reference:
  - matched >= 0.90
  - same species >= 0.97
  - mean IoU >= 0.85
  - mean score difference <= 0.05
- AP50 drop of at most 3 points per class.
- Elephant-to-sloth-bear errors and empty-frame false positives no worse
  than the ONNX baseline. The ONNX baseline fires on 1 of 104 empty test
  frames: tiger, 0.49.

## 6. R1 rebuild

Folder: `zovive-ml/hailo_builds/det150e_h8l_r1/` (commands and gates in its
`BUILD.md`). It is isolated: nothing outside that folder is written and
nothing is deployed.

- The model script loads the normalization line, sets optimization level 1
  and compression level 0, and uses 256 calibration entries.
- The calibration set comes from the train split only: 128 images per
  species plus 128 empty frames, letterboxed with this repository's
  `letterbox()`.
- Gates:

  | Gate | Where | What |
  |---|---|---|
  | B0 | laptop | float emulator |
  | B1 | laptop | quantized emulator |
  | C | Pi | parity, using `pi/dump_detections.py` |
  | D | Pi | FPS and latency |

  Compiling the HEF, running gates C and D, and deploying each need separate
  approval.

## 7. Deployment layout (reported; confirm with the read-only Pi check)

- Service: `zovive-detect.service` (systemd).
- Code: `/opt/zovive`, said to be a symlink to `/home/zovive1/zovive-edge`.
  `docs/STORAGE_LAYOUT.md` describes `/opt/zovive` as the code location.
- Virtual environment: `/opt/zovive/.venv/`.
- Branch on the Pi: `develop`.
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
   `models/manifest.json` without the owner's explicit approval, and never
   deploy one that has not passed the gates in section 5.
6. Report Hailo performance only from measurements on the real Pi, with the
   command, HEF hash and frame source recorded.
7. Never print `.env`, camera credentials or keys.

## 9. Roadmap

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
