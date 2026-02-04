#!/usr/bin/env python3
"""
Audit the FinWhale training procedure and dataset prep.

Checks:
- Split integrity (duplicate files and time overlaps across splits)
- Crop size behavior (square, full frequency range when crop_size=None)
- Augmentation (positive call decentering stays in crop, based on energy peak)
- Normalization (range, clipping, NaN/Inf)

Writes a Markdown summary for review.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure repo root is on sys.path so `src` is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.mat_dataset import FinWhaleMatDataset, parse_crop_size, _power_to_db_norm
from src.training.mat_utils import list_mat_files, parse_mat_filename
from src.training.splits import build_entries, split_group_by_source, split_time_separated

# Hardcoded sample filenames to avoid slow directory scans when copying from mounted drives.
# These are basenames (with or without .mat). They will be resolved relative to pos_dir/neg_dir.
FIXED_POS_SAMPLE_NAMES: List[str] = [
    # 2018-07-05
    "ICLISTENHF1353_20180705T044128.948Z_57.0s_57.6s",
    "ICLISTENHF1353_20180705T044128.948Z_60.5s_60.9s",
    "ICLISTENHF1353_20180705T052128.948Z_285.3s_286.0s",
    "ICLISTENHF1353_20180705T052128.948Z_288.5s_289.0s",
    "ICLISTENHF1353_20180705T060128.948Z_8.0s_8.7s",
    "ICLISTENHF1353_20180705T050128.948Z_16.8s_17.3s",
    "ICLISTENHF1353_20180705T061128.947Z_56.5s_57.1s",
    "ICLISTENHF1353_20180705T061128.947Z_70.7s_71.1s",
    "ICLISTENHF1353_20180705T061128.947Z_170.4s_171.1s",
    "ICLISTENHF1353_20180705T062128.947Z_27.3s_27.9s",
    "ICLISTENHF1353_20180705T061128.947Z_295.6s_296.1s",
    "ICLISTENHF1353_20180705T062128.947Z_270.4s_271.1s",
    "ICLISTENHF1353_20180705T063128.947Z_133.2s_133.7s",
    "ICLISTENHF1353_20180705T064128.947Z_76.8s_77.3s",
    "ICLISTENHF1353_20180705T114128.946Z_184.5s_185.1s",
    "ICLISTENHF1353_20180705T093128.946Z_111.3s_112.0s",
    "ICLISTENHF1353_20180705T122128.945Z_193.8s_194.3s",
    "ICLISTENHF1353_20180705T123128.945Z_75.1s_75.7s",
    "ICLISTENHF1353_20180705T093128.946Z_119.6s_120.0s",
    "ICLISTENHF1353_20180705T122128.945Z_260.2s_260.6s",
    "ICLISTENHF1353_20180705T165018.944Z_24.6s_25.2s",
    "ICLISTENHF1353_20180705T165018.944Z_46.2s_46.7s",
    "ICLISTENHF1353_20180705T165018.944Z_56.1s_56.9s",
    "ICLISTENHF1353_20180705T165018.944Z_58.0s_58.6s",
    "ICLISTENHF1353_20180705T165018.944Z_59.7s_60.2s",
    "ICLISTENHF1353_20180705T165018.944Z_77.9s_78.7s",
    # 2018-07-15 (early hours)
    "ICLISTENHF1353_20180715T020431.118Z_264.8s_265.4s",
    "ICLISTENHF1353_20180715T021431.118Z_208.8s_209.3s",
    "ICLISTENHF1353_20180715T021431.118Z_245.8s_246.2s",
    "ICLISTENHF1353_20180715T023431.118Z_34.8s_35.3s",
    "ICLISTENHF1353_20180715T030431.117Z_31.0s_31.4s",
    "ICLISTENHF1353_20180715T032431.117Z_201.7s_202.3s",
    "ICLISTENHF1353_20180715T032431.117Z_208.1s_208.9s",
    "ICLISTENHF1353_20180715T032431.117Z_219.7s_220.1s",
    "ICLISTENHF1353_20180715T032431.117Z_234.3s_234.8s",
    "ICLISTENHF1353_20180715T032431.117Z_274.6s_275.2s",
    "ICLISTENHF1353_20180715T032431.117Z_276.8s_277.4s",
    "ICLISTENHF1353_20180715T032431.117Z_286.6s_287.2s",
    "ICLISTENHF1353_20180715T032431.117Z_297.3s_298.5s",
    "ICLISTENHF1353_20180715T041431.117Z_19.4s_19.9s",
    "ICLISTENHF1353_20180715T041431.117Z_24.1s_24.5s",
    "ICLISTENHF1353_20180715T041431.117Z_141.8s_142.3s",
    "ICLISTENHF1353_20180715T041431.117Z_179.7s_180.2s",
    "ICLISTENHF1353_20180715T041431.117Z_189.0s_189.6s",
    "ICLISTENHF1353_20180715T041431.117Z_256.1s_256.7s",
    "ICLISTENHF1353_20180715T041431.117Z_268.6s_269.1s",
    "ICLISTENHF1353_20180715T041431.117Z_274.7s_275.2s",
    "ICLISTENHF1353_20180715T041431.117Z_286.5s_287.2s",
    "ICLISTENHF1353_20180715T041431.117Z_291.8s_292.2s",
    "ICLISTENHF1353_20180715T042431.117Z_16.5s_17.2s",
    "ICLISTENHF1353_20180715T042431.117Z_24.6s_25.0s",
    "ICLISTENHF1353_20180715T042431.117Z_28.8s_29.2s",
    # 2018-07-15 (morning)
    "ICLISTENHF1353_20180715T060431.117Z_20.5s_20.9s",
    "ICLISTENHF1353_20180715T060431.117Z_25.3s_25.8s",
    "ICLISTENHF1353_20180715T060431.117Z_71.7s_72.3s",
    "ICLISTENHF1353_20180715T060431.117Z_95.8s_96.2s",
    "ICLISTENHF1353_20180715T061431.116Z_181.2s_181.8s",
    "ICLISTENHF1353_20180715T063431.116Z_18.5s_19.2s",
    "ICLISTENHF1353_20180715T063431.116Z_40.4s_40.9s",
    "ICLISTENHF1353_20180715T063431.116Z_56.3s_56.8s",
    "ICLISTENHF1353_20180715T063431.116Z_75.3s_75.8s",
    "ICLISTENHF1353_20180715T063431.116Z_105.0s_105.4s",
    "ICLISTENHF1353_20180715T071431.116Z_17.4s_17.9s",
    "ICLISTENHF1353_20180715T114431.115Z_82.9s_83.4s",
    "ICLISTENHF1353_20180715T114431.115Z_90.9s_91.6s",
    "ICLISTENHF1353_20180715T114431.115Z_102.8s_103.6s",
    "ICLISTENHF1353_20180715T114431.115Z_121.2s_121.6s",
    "ICLISTENHF1353_20180715T114431.115Z_130.0s_130.8s",
    "ICLISTENHF1353_20180715T114431.115Z_144.8s_145.4s",
    "ICLISTENHF1353_20180715T114431.115Z_158.6s_159.0s",
    "ICLISTENHF1353_20180715T114431.115Z_169.9s_170.5s",
    "ICLISTENHF1353_20180715T114431.115Z_186.4s_186.8s",
    "ICLISTENHF1353_20180715T114431.115Z_192.6s_193.0s",
    "ICLISTENHF1353_20180715T114431.115Z_210.3s_210.8s",
    "ICLISTENHF1353_20180715T114431.115Z_224.6s_224.9s",
    "ICLISTENHF1353_20180715T114431.115Z_234.0s_234.5s",
    "ICLISTENHF1353_20180715T114431.115Z_244.6s_245.2s",
    "ICLISTENHF1353_20180715T114431.115Z_256.5s_257.0s",
    # 2018-07-15 (afternoon/evening) + 2018-07-20
    "ICLISTENHF1353_20180715T174431.113Z_14.1s_14.5s",
    "ICLISTENHF1353_20180715T174431.113Z_205.3s_205.8s",
    "ICLISTENHF1353_20180715T175431.113Z_231.5s_231.8s",
    "ICLISTENHF1353_20180715T175431.113Z_243.8s_244.3s",
    "ICLISTENHF1353_20180715T181431.113Z_148.2s_148.7s",
    "ICLISTENHF1353_20180715T181431.113Z_161.4s_161.8s",
    "ICLISTENHF1353_20180715T181431.113Z_167.0s_167.4s",
    "ICLISTENHF1353_20180715T181431.113Z_180.8s_181.2s",
    "ICLISTENHF1353_20180715T181431.113Z_254.5s_255.0s",
    "ICLISTENHF1353_20180715T185431.112Z_205.7s_206.0s",
    "ICLISTENHF1353_20180715T192431.112Z_255.6s_256.4s",
    "ICLISTENHF1353_20180715T194431.112Z_83.1s_83.8s",
    "ICLISTENHF1353_20180715T194431.112Z_106.7s_107.3s",
    "ICLISTENHF1353_20180715T194431.112Z_240.6s_241.2s",
    "ICLISTENHF1353_20180715T194431.112Z_247.7s_248.4s",
    "ICLISTENHF1353_20180715T194431.112Z_254.3s_254.9s",
    "ICLISTENHF1353_20180715T194431.112Z_261.7s_262.4s",
    "ICLISTENHF1353_20180715T194431.112Z_296.0s_296.4s",
    "ICLISTENHF1353_20180715T200431.112Z_77.0s_77.7s",
    "ICLISTENHF1353_20180715T200431.112Z_126.3s_126.7s",
    "ICLISTENHF1353_20180715T200431.112Z_140.6s_141.0s",
    "ICLISTENHF1353_20180715T202431.112Z_158.5s_158.9s",
    "ICLISTENHF1353_20180720T000020.054Z_0.8s_1.5s",
    "ICLISTENHF1353_20180720T000020.054Z_19.2s_19.6s",
    "ICLISTENHF1353_20180720T000020.054Z_58.0s_58.5s",
    "ICLISTENHF1353_20180720T000020.054Z_207.0s_207.4s",
    # 2019-02-10
    "ICLISTENHF1353_20190210T215000.062Z_1.4s_2.0s",
    "ICLISTENHF1353_20190210T215000.062Z_1.7s_2.2s",
    "ICLISTENHF1353_20190210T215000.062Z_5.9s_6.6s",
    "ICLISTENHF1353_20190210T215000.062Z_6.9s_7.4s",
    "ICLISTENHF1353_20190210T215000.062Z_7.4s_8.0s",
    "ICLISTENHF1353_20190210T215000.062Z_7.9s_8.4s",
    "ICLISTENHF1353_20190210T215000.062Z_7.9s_8.6s",
    "ICLISTENHF1353_20190210T215000.062Z_8.6s_9.2s",
    "ICLISTENHF1353_20190210T215000.062Z_9.1s_9.6s",
    "ICLISTENHF1353_20190210T215000.062Z_9.6s_10.2s",
    "ICLISTENHF1353_20190210T215000.062Z_10.6s_11.0s",
    "ICLISTENHF1353_20190210T215000.062Z_11.1s_11.4s",
    "ICLISTENHF1353_20190210T215000.062Z_12.0s_12.4s",
    "ICLISTENHF1353_20190210T215000.062Z_13.3s_14.0s",
    "ICLISTENHF1353_20190210T215000.062Z_14.0s_14.8s",
    "ICLISTENHF1353_20190210T215000.062Z_15.6s_16.1s",
    "ICLISTENHF1353_20190210T215000.062Z_17.8s_18.3s",
    "ICLISTENHF1353_20190210T215000.062Z_18.8s_19.3s",
    "ICLISTENHF1353_20190210T215000.062Z_20.4s_20.9s",
    "ICLISTENHF1353_20190210T215000.062Z_22.0s_22.5s",
    "ICLISTENHF1353_20190210T215000.062Z_23.9s_24.4s",
    "ICLISTENHF1353_20190210T215000.062Z_25.2s_25.7s",
    "ICLISTENHF1353_20190210T215000.062Z_27.4s_27.9s",
    "ICLISTENHF1353_20190210T215000.062Z_28.0s_28.6s",
    "ICLISTENHF1353_20190210T215000.062Z_28.6s_29.2s",
    "ICLISTENHF1353_20190210T215000.062Z_29.2s_29.6s",
]

FIXED_NEG_SAMPLE_NAMES: List[str] = [
    # 2018-07-05
    "ICLISTENHF1353_20180705T044128.948Z.wav_neg_0",
    "ICLISTENHF1353_20180705T044128.948Z.wav_neg_1",
    "ICLISTENHF1353_20180705T050128.948Z.wav_neg_0",
    "ICLISTENHF1353_20180705T052128.948Z.wav_neg_0",
    "ICLISTENHF1353_20180705T052128.948Z.wav_neg_1",
    "ICLISTENHF1353_20180705T060128.948Z.wav_neg_0",
    "ICLISTENHF1353_20180705T061128.947Z.wav_neg_0",
    "ICLISTENHF1353_20180705T061128.947Z.wav_neg_1",
    "ICLISTENHF1353_20180705T061128.947Z.wav_neg_2",
    "ICLISTENHF1353_20180705T061128.947Z.wav_neg_3",
    "ICLISTENHF1353_20180705T062128.947Z.wav_neg_0",
    "ICLISTENHF1353_20180705T062128.947Z.wav_neg_1",
    "ICLISTENHF1353_20180705T063128.947Z.wav_neg_0",
    "ICLISTENHF1353_20180705T064128.947Z.wav_neg_0",
    "ICLISTENHF1353_20180705T093128.946Z.wav_neg_0",
    "ICLISTENHF1353_20180705T093128.946Z.wav_neg_1",
    "ICLISTENHF1353_20180705T114128.946Z.wav_neg_0",
    "ICLISTENHF1353_20180705T122128.945Z.wav_neg_0",
    "ICLISTENHF1353_20180705T122128.945Z.wav_neg_1",
    "ICLISTENHF1353_20180705T123128.945Z.wav_neg_0",
    "ICLISTENHF1353_20180705T162128.944Z.wav_neg_0",
    "ICLISTENHF1353_20180705T162128.944Z.wav_neg_1",
    "ICLISTENHF1353_20180705T164128.944Z.wav_neg_0",
    "ICLISTENHF1353_20180705T165018.944Z.wav_neg_0",
    "ICLISTENHF1353_20180705T165018.944Z.wav_neg_1",
    "ICLISTENHF1353_20180705T170018.944Z.wav_neg_0",
    "ICLISTENHF1353_20180705T170018.944Z.wav_neg_1",
    # 2018-07-20
    "ICLISTENHF1353_20180720T140020.049Z.wav_neg_2",
    "ICLISTENHF1353_20180720T140020.049Z.wav_neg_3",
    "ICLISTENHF1353_20180720T141020.049Z.wav_neg_0",
    "ICLISTENHF1353_20180720T141020.049Z.wav_neg_1",
    "ICLISTENHF1353_20180720T142020.049Z.wav_neg_0",
    "ICLISTENHF1353_20180720T143020.049Z.wav_neg_0",
    "ICLISTENHF1353_20180720T143020.049Z.wav_neg_1",
    "ICLISTENHF1353_20180720T143020.049Z.wav_neg_2",
    "ICLISTENHF1353_20180720T143020.049Z.wav_neg_3",
    "ICLISTENHF1353_20180720T150020.049Z.wav_neg_0",
    "ICLISTENHF1353_20180720T150020.049Z.wav_neg_1",
    "ICLISTENHF1353_20180720T151020.049Z.wav_neg_0",
    "ICLISTENHF1353_20180720T151020.049Z.wav_neg_1",
    "ICLISTENHF1353_20180720T151020.049Z.wav_neg_2",
    "ICLISTENHF1353_20180720T151020.049Z.wav_neg_3",
    "ICLISTENHF1353_20180720T151020.049Z.wav_neg_4",
    "ICLISTENHF1353_20180720T153020.048Z.wav_neg_0",
    "ICLISTENHF1353_20180720T153020.048Z.wav_neg_1",
    "ICLISTENHF1353_20180720T153020.048Z.wav_neg_2",
    "ICLISTENHF1353_20180720T155020.048Z.wav_neg_0",
    "ICLISTENHF1353_20180720T155020.048Z.wav_neg_1",
    "ICLISTENHF1353_20180720T155020.048Z.wav_neg_2",
    "ICLISTENHF1353_20180720T160020.048Z.wav_neg_0",
    "ICLISTENHF1353_20180720T160020.048Z.wav_neg_1",
    "ICLISTENHF1353_20180720T162020.048Z.wav_neg_0",
    "ICLISTENHF1353_20180720T162020.048Z.wav_neg_1",
    "ICLISTENHF1353_20180720T164020.048Z.wav_neg_0",
    # 2018-07-30
    "ICLISTENHF1353_20180730T040002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T040002.974Z.wav_neg_1",
    "ICLISTENHF1353_20180730T040002.974Z.wav_neg_2",
    "ICLISTENHF1353_20180730T041002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T051002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T051002.974Z.wav_neg_1",
    "ICLISTENHF1353_20180730T051002.974Z.wav_neg_2",
    "ICLISTENHF1353_20180730T053002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T053002.974Z.wav_neg_1",
    "ICLISTENHF1353_20180730T053002.974Z.wav_neg_2",
    "ICLISTENHF1353_20180730T054002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T054002.974Z.wav_neg_1",
    "ICLISTENHF1353_20180730T054002.974Z.wav_neg_2",
    "ICLISTENHF1353_20180730T055002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T055002.974Z.wav_neg_1",
    "ICLISTENHF1353_20180730T060002.974Z.wav_neg_0",
    "ICLISTENHF1353_20180730T060002.974Z.wav_neg_1",
    "ICLISTENHF1353_20180730T061002.973Z.wav_neg_0",
    "ICLISTENHF1353_20180730T062002.973Z.wav_neg_0",
    "ICLISTENHF1353_20180730T062002.973Z.wav_neg_1",
    "ICLISTENHF1353_20180730T064002.973Z.wav_neg_0",
    "ICLISTENHF1353_20180730T065002.973Z.wav_neg_0",
    "ICLISTENHF1353_20180730T065002.973Z.wav_neg_1",
    "ICLISTENHF1353_20180730T065002.973Z.wav_neg_2",
    "ICLISTENHF1353_20180730T065002.973Z.wav_neg_3",
    "ICLISTENHF1353_20180730T070002.973Z.wav_neg_0",
    "ICLISTENHF1353_20180730T070002.973Z.wav_neg_1",
    # 2018-11-20
    "ICLISTENHF1353_20181120T002000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T002000.086Z.wav_neg_1",
    "ICLISTENHF1353_20181120T003000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T003000.086Z.wav_neg_1",
    "ICLISTENHF1353_20181120T005000.335Z.wav_neg_0",
    "ICLISTENHF1353_20181120T005000.335Z.wav_neg_1",
    "ICLISTENHF1353_20181120T005000.335Z.wav_neg_2",
    "ICLISTENHF1353_20181120T005000.335Z.wav_neg_3",
    "ICLISTENHF1353_20181120T010000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T010000.086Z.wav_neg_1",
    "ICLISTENHF1353_20181120T010000.086Z.wav_neg_2",
    "ICLISTENHF1353_20181120T010000.086Z.wav_neg_3",
    "ICLISTENHF1353_20181120T010000.086Z.wav_neg_4",
    "ICLISTENHF1353_20181120T011000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T011000.086Z.wav_neg_1",
    "ICLISTENHF1353_20181120T011000.086Z.wav_neg_2",
    "ICLISTENHF1353_20181120T012000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T012000.086Z.wav_neg_1",
    "ICLISTENHF1353_20181120T012000.086Z.wav_neg_2",
    "ICLISTENHF1353_20181120T013000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T014000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T015000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T021000.086Z.wav_neg_0",
    "ICLISTENHF1353_20181120T030000.085Z.wav_neg_0",
    "ICLISTENHF1353_20181120T040000.085Z.wav_neg_0",
    "ICLISTENHF1353_20181120T041000.085Z.wav_neg_0",
    # 2019-03-05
    "ICLISTENHF1353_20190305T112000.156Z.wav_neg_2",
    "ICLISTENHF1353_20190305T115000.156Z.wav_neg_0",
    "ICLISTENHF1353_20190305T115000.156Z.wav_neg_1",
    "ICLISTENHF1353_20190305T123000.156Z.wav_neg_0",
    "ICLISTENHF1353_20190305T150000.155Z.wav_neg_0",
    "ICLISTENHF1353_20190305T150000.155Z.wav_neg_1",
    "ICLISTENHF1353_20190305T164000.155Z.wav_neg_0",
    "ICLISTENHF1353_20190305T164000.155Z.wav_neg_1",
    "ICLISTENHF1353_20190305T164000.155Z.wav_neg_2",
    "ICLISTENHF1353_20190305T164000.155Z.wav_neg_3",
    "ICLISTENHF1353_20190305T164000.155Z.wav_neg_4",
    "ICLISTENHF1353_20190305T164500.155Z.wav_neg_0",
    "ICLISTENHF1353_20190305T164500.155Z.wav_neg_1",
    "ICLISTENHF1353_20190305T164500.155Z.wav_neg_2",
    "ICLISTENHF1353_20190305T164500.155Z.wav_neg_3",
    "ICLISTENHF1353_20190305T164500.155Z.wav_neg_4",
    "ICLISTENHF1353_20190305T172000.155Z.wav_neg_0",
    "ICLISTENHF1353_20190305T172000.155Z.wav_neg_1",
    "ICLISTENHF1353_20190305T173000.155Z.wav_neg_0",
    "ICLISTENHF1353_20190305T173000.155Z.wav_neg_1",
    "ICLISTENHF1353_20190305T173000.155Z.wav_neg_2",
    "ICLISTENHF1353_20190305T173000.155Z.wav_neg_3",
    "ICLISTENHF1353_20190305T174000.155Z.wav_neg_0",
    "ICLISTENHF1353_20190305T174500.154Z.wav_neg_0",
    "ICLISTENHF1353_20190305T174500.154Z.wav_neg_1",
]


@dataclass
class AuditConfig:
    pos_dir: Path
    neg_dir: Path
    crop_size: Optional[object]
    min_db: float
    max_db: float
    train_ratio: float
    val_ratio: float
    seed: int
    split_strategy: str
    min_gap_seconds: float
    center_bias_sigma_frac: float
    sample_size: int
    n_augment: int
    out_path: Path
    splits_dir: Optional[Path]
    copy_sample: bool
    sample_out_dir: Optional[Path]
    max_overlap_count: int
    use_fixed_samples: bool
    fixed_samples_file: Optional[Path]
    fixed_neg_samples_file: Optional[Path]
    progress_interval: int
    scan_full: bool


def _load_args_pkl(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            args = pickle.load(f)
        if isinstance(args, dict):
            return args
        # argparse.Namespace
        return {k: getattr(args, k) for k in dir(args) if not k.startswith("_")}
    except Exception:
        return {}


def _parse_crop_arg(value: Optional[str]) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, (int, list, tuple)):
        return value
    s = str(value).strip()
    if not s:
        return None
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [int(parts[0]), int(parts[1])]
    try:
        return int(s)
    except Exception:
        return None


def _sample_paths(paths: Sequence[Path], n: int, seed: int) -> List[Path]:
    if n <= 0 or len(paths) <= n:
        return list(paths)
    rng = random.Random(seed)
    return rng.sample(list(paths), n)


def _load_name_list(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return []
    names: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def _resolve_names(root: Path, names: Sequence[str]) -> Tuple[List[Path], List[str]]:
    resolved: List[Path] = []
    missing: List[str] = []
    for name in names:
        cand = root / name
        if cand.exists():
            resolved.append(cand)
            continue
        if not name.lower().endswith(".mat"):
            cand2 = root / f"{name}.mat"
            if cand2.exists():
                resolved.append(cand2)
                continue
        missing.append(name)
    return resolved, missing


def _unique_ordered(items: Sequence[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_splits_dir(split_dir: Path) -> Optional[Dict[str, List[Tuple[Path, int]]]]:
    if not split_dir.exists():
        return None
    out: Dict[str, List[Tuple[Path, int]]] = {}
    for split in ("train", "val", "test"):
        p = split_dir / f"{split}.txt"
        if not p.exists():
            return None
        entries: List[Tuple[Path, int]] = []
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                entries.append((Path(parts[0]), int(parts[1])))
        out[split] = entries
    return out


def _internal_split(
    pos_files: Sequence[Path],
    neg_files: Sequence[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, int]]]:
    def _split(files: Sequence[Path]) -> Dict[str, List[Path]]:
        n = len(files)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = [files[i] for i in idx[:n_train]]
        val = [files[i] for i in idx[n_train:n_train + n_val]]
        test = [files[i] for i in idx[n_train + n_val:]]
        return {"train": train, "val": val, "test": test}

    pos_split = _split(pos_files)
    neg_split = _split(neg_files)
    out: Dict[str, List[Tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        out[split].extend([(p, 1) for p in pos_split[split]])
        out[split].extend([(n, 0) for n in neg_split[split]])
    return out


def _entries_from_split(split_files: Dict[str, List[Tuple[Path, int]]]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for split, items in split_files.items():
        for path, label in items:
            src, start, dur = parse_mat_filename(path.name)
            out[split].append({"path": path, "src": src, "start": start, "dur": dur, "label": label})
    return out


def _by_source_intervals(entries: Iterable[dict]) -> Dict[str, List[Tuple[float, float, Path]]]:
    out: Dict[str, List[Tuple[float, float, Path]]] = {}
    for e in entries:
        if e.get("label") != 1:
            continue
        start = e.get("start")
        dur = e.get("dur")
        if start is None or dur is None:
            continue
        end = float(start) + float(dur)
        out.setdefault(e["src"], []).append((float(start), end, e["path"]))
    for src in out:
        out[src].sort(key=lambda x: x[0])
    return out


def _count_overlaps(
    a: Dict[str, List[Tuple[float, float, Path]]],
    b: Dict[str, List[Tuple[float, float, Path]]],
    limit_examples: int = 5,
    max_count: int = 100000,
) -> Tuple[int, List[Tuple[Path, Path]]]:
    count = 0
    examples: List[Tuple[Path, Path]] = []
    for src, a_list in a.items():
        b_list = b.get(src)
        if not b_list:
            continue
        j = 0
        for a_start, a_end, a_path in a_list:
            while j < len(b_list) and b_list[j][1] <= a_start:
                j += 1
            k = j
            while k < len(b_list) and b_list[k][0] < a_end:
                count += 1
                if len(examples) < limit_examples:
                    examples.append((a_path, b_list[k][2]))
                if count >= max_count:
                    return count, examples
                k += 1
    return count, examples


def _compute_peak_time(spec: np.ndarray) -> Optional[int]:
    if spec.size == 0:
        return None
    energy = np.nanmean(spec, axis=0)
    if energy.size == 0 or np.all(np.isnan(energy)):
        return None
    return int(np.nanargmax(energy))


def _copy_sample(pos_samples: Sequence[Path], neg_samples: Sequence[Path], out_dir: Path) -> None:
    out_pos = out_dir / "pos"
    out_neg = out_dir / "neg"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pos_samples, 1):
        _log(f"[copy] pos {i}/{len(pos_samples)}: {p.name}")
        shutil.copy2(p, out_pos / p.name)
    for i, p in enumerate(neg_samples, 1):
        _log(f"[copy] neg {i}/{len(neg_samples)}: {p.name}")
        shutil.copy2(p, out_neg / p.name)


def _fmt(v: object) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def audit(config: AuditConfig) -> str:
    _log("Starting training audit...")
    _log(f"pos_dir: {config.pos_dir}")
    _log(f"neg_dir: {config.neg_dir}")
    _log(f"split_strategy: {config.split_strategy}")
    _log(f"use_fixed_samples: {config.use_fixed_samples}")
    _log(f"copy_sample: {config.copy_sample}")
    _log(f"sample_size per class: {config.sample_size}")

    # Load file lists (optional)
    pos_files: List[Path] = []
    neg_files: List[Path] = []
    if config.scan_full:
        _log("Listing .mat files in pos/neg directories (this can be slow on mounted drives)...")
        pos_files = list_mat_files(str(config.pos_dir))
        neg_files = list_mat_files(str(config.neg_dir))
        _log(f"Found pos_files={len(pos_files)} neg_files={len(neg_files)}")
    else:
        _log("Skipping full directory scan. Split checks will be limited to provided split files (if any).")

    # Resolve splits
    split_files = None
    split_source = "computed"
    if config.splits_dir:
        split_files = _load_splits_dir(config.splits_dir)
        if split_files:
            split_source = f"from {config.splits_dir}"
            _log(f"Loaded split files {split_source}")

    if split_files is None:
        if config.scan_full:
            _log("Computing splits...")
            if config.split_strategy == "internal":
                split_files = _internal_split(pos_files, neg_files, config.train_ratio, config.val_ratio, config.seed)
            else:
                entries = build_entries(str(config.pos_dir), str(config.neg_dir))
                if config.split_strategy == "group_by_source":
                    sp = split_group_by_source(entries, config.train_ratio, config.val_ratio, config.seed)
                else:
                    sp = split_time_separated(entries, config.train_ratio, config.val_ratio, config.seed, config.min_gap_seconds)
                split_files = {
                    "train": [(Path(e["path"]), int(e["label"])) for e in sp["train"]],
                    "val": [(Path(e["path"]), int(e["label"])) for e in sp["val"]],
                    "test": [(Path(e["path"]), int(e["label"])) for e in sp["test"]],
                }
            _log("Splits computed.")
        else:
            _log("No split files provided and full scan disabled; split checks will be skipped.")

    split_entries = _entries_from_split(split_files) if split_files else None
    split_counts = None
    missing_time = None
    if split_files:
        split_counts = {
            split: {
                "pos": sum(1 for _, lbl in split_files[split] if lbl == 1),
                "neg": sum(1 for _, lbl in split_files[split] if lbl == 0),
                "total": len(split_files[split]),
            }
            for split in ("train", "val", "test")
        }
        missing_time = {
            split: sum(1 for e in split_entries[split] if e.get("label") == 1 and (e.get("start") is None or e.get("dur") is None))
            for split in ("train", "val", "test")
        }

    # Duplicate files across splits
    dup_train_val = set()
    dup_train_test = set()
    dup_val_test = set()
    ov_train_val = ov_train_test = ov_val_test = 0
    ex_train_val = ex_train_test = ex_val_test = []
    if split_files and split_entries:
        _log("Checking duplicate files across splits...")
        split_sets = {k: set(p for p, _ in v) for k, v in split_files.items()}
        dup_train_val = split_sets["train"].intersection(split_sets["val"])
        dup_train_test = split_sets["train"].intersection(split_sets["test"])
        dup_val_test = split_sets["val"].intersection(split_sets["test"])

        # Overlap checks (positives, same source)
        _log("Checking positive overlaps across splits (same source)...")
        train_by_src = _by_source_intervals(split_entries["train"])
        val_by_src = _by_source_intervals(split_entries["val"])
        test_by_src = _by_source_intervals(split_entries["test"])
        ov_train_val, ex_train_val = _count_overlaps(train_by_src, val_by_src, max_count=config.max_overlap_count)
        ov_train_test, ex_train_test = _count_overlaps(train_by_src, test_by_src, max_count=config.max_overlap_count)
        ov_val_test, ex_val_test = _count_overlaps(val_by_src, test_by_src, max_count=config.max_overlap_count)

    # Sample for deep checks
    _log("Resolving sample files...")
    fixed_pos_missing: List[str] = []
    fixed_neg_missing: List[str] = []
    fixed_pos_fallback = False
    fixed_neg_fallback = False
    pos_samples: List[Path] = []
    neg_samples: List[Path] = []

    fixed_pos_names = list(FIXED_POS_SAMPLE_NAMES)
    fixed_pos_names += _load_name_list(config.fixed_samples_file)
    fixed_pos_names = [str(n).strip() for n in fixed_pos_names if str(n).strip()]
    fixed_pos_names = [str(n) for n in _unique_ordered(fixed_pos_names)]

    fixed_neg_names = list(FIXED_NEG_SAMPLE_NAMES)
    fixed_neg_names += _load_name_list(config.fixed_neg_samples_file)
    fixed_neg_names = [str(n).strip() for n in fixed_neg_names if str(n).strip()]
    fixed_neg_names = [str(n) for n in _unique_ordered(fixed_neg_names)]

    if config.use_fixed_samples and fixed_pos_names:
        pos_samples, fixed_pos_missing = _resolve_names(config.pos_dir, fixed_pos_names)
        if config.sample_size > 0 and len(pos_samples) > config.sample_size:
            pos_samples = pos_samples[: config.sample_size]
        if not pos_samples:
            fixed_pos_fallback = True
            if split_files:
                pos_samples = _sample_paths([p for p, _ in split_files["train"] if _ == 1], config.sample_size, config.seed)
            else:
                raise SystemExit("No fixed pos samples resolved and full scan disabled; provide fixed samples or enable --scan-full.")
        _log(f"Fixed pos samples resolved: {len(pos_samples)} (missing: {len(fixed_pos_missing)})")
    else:
        if split_files:
            pos_samples = _sample_paths([p for p, _ in split_files["train"] if _ == 1], config.sample_size, config.seed)
            _log(f"Random pos samples selected: {len(pos_samples)}")
        else:
            raise SystemExit("Random sampling requires split files or full scan. Use --use-fixed-samples or --scan-full.")

    if config.use_fixed_samples and fixed_neg_names:
        neg_samples, fixed_neg_missing = _resolve_names(config.neg_dir, fixed_neg_names)
        if config.sample_size > 0 and len(neg_samples) > config.sample_size:
            neg_samples = neg_samples[: config.sample_size]
        if not neg_samples:
            fixed_neg_fallback = True
            if split_files:
                neg_samples = _sample_paths([p for p, _ in split_files["train"] if _ == 0], config.sample_size, config.seed)
            else:
                raise SystemExit("No fixed neg samples resolved and full scan disabled; provide fixed samples or enable --scan-full.")
        _log(f"Fixed neg samples resolved: {len(neg_samples)} (missing: {len(fixed_neg_missing)})")
    else:
        if split_files:
            neg_samples = _sample_paths([p for p, _ in split_files["train"] if _ == 0], config.sample_size, config.seed)
            _log(f"Random neg samples selected: {len(neg_samples)}")
        else:
            raise SystemExit("Random sampling requires split files or full scan. Use --use-fixed-samples or --scan-full.")

    if config.copy_sample and config.sample_out_dir:
        _log(f"Copying samples to {config.sample_out_dir} ...")
        _copy_sample(pos_samples, neg_samples, config.sample_out_dir)
        _log("Sample copy complete.")

    # Dataset for normalization and cropping checks
    _log("Initializing dataset for normalization/crop checks...")
    freq_crop, time_crop = parse_crop_size(config.crop_size)
    sample_list = [(p, 1) for p in pos_samples] + [(n, 0) for n in neg_samples]
    ds = FinWhaleMatDataset(
        str(config.pos_dir),
        str(config.neg_dir),
        split="train",
        crop_size=config.crop_size,
        min_db=config.min_db,
        max_db=config.max_db,
        center_bias_sigma_frac=config.center_bias_sigma_frac,
        seed=config.seed,
        file_list=sample_list,
        return_meta=True,
    )

    # Collect stats
    raw_shapes: List[Tuple[int, int]] = []
    crop_shapes: List[Tuple[int, int]] = []
    freq_crop_counts = {"padded": 0, "cropped": 0, "unchanged": 0}
    time_crop_counts = {"padded": 0, "cropped": 0, "unchanged": 0}
    spec_kind_counts = {"power": 0, "db": 0}
    norm_min = math.inf
    norm_max = -math.inf
    norm_nan = 0
    norm_inf = 0
    norm_mean_acc = 0.0
    norm_var_acc = 0.0
    norm_n = 0
    clip_below = 0
    clip_above = 0
    clip_total = 0

    # Augmentation stats
    peak_outside = 0
    peak_total = 0
    peak_offset_abs = []
    crop_center_offset_frac = []

    for idx, (path, label) in enumerate(sample_list, 1):
        if idx == 1 or idx % max(1, config.progress_interval) == 0:
            _log(f"[sample] {idx}/{len(sample_list)}: {path.name}")
        try:
            spec, spec_kind = ds._load_spectrogram_raw(path)
        except Exception:
            continue
        F, T = spec.shape
        raw_shapes.append((F, T))

        if spec_kind == "power":
            spec_kind_counts["power"] += 1
            spec_db = _power_to_db_norm(spec)
        else:
            spec_kind_counts["db"] += 1
            spec_db = spec.astype(np.float32)

        clip_total += spec_db.size
        clip_below += int(np.sum(spec_db < config.min_db))
        clip_above += int(np.sum(spec_db > config.max_db))

        target_f = freq_crop if freq_crop is not None else F
        target_t = time_crop if time_crop is not None else target_f

        if F < target_f:
            freq_crop_counts["padded"] += 1
        elif F > target_f:
            freq_crop_counts["cropped"] += 1
        else:
            freq_crop_counts["unchanged"] += 1

        if T < target_t:
            time_crop_counts["padded"] += 1
        elif T > target_t:
            time_crop_counts["cropped"] += 1
        else:
            time_crop_counts["unchanged"] += 1

        # Augmentation: only positives
        if label == 1:
            peak = _compute_peak_time(spec_db)
            if peak is not None:
                center = T // 2
                peak_offset_abs.append(abs(peak - center) / max(1, center))
                for _ in range(config.n_augment):
                    _, start = ds._crop(spec, is_positive=True)
                    inside = (peak >= start) and (peak < start + target_t)
                    peak_total += 1
                    if not inside:
                        peak_outside += 1

        # Normalized stats: use dataset transform
        try:
            x, _, meta = ds[idx - 1]  # type: ignore[misc]
        except Exception:
            continue
        if isinstance(meta, dict):
            if "dist_from_center_frac" in meta:
                crop_center_offset_frac.append(float(meta["dist_from_center_frac"]))
        arr = x.numpy()
        if np.isnan(arr).any():
            norm_nan += int(np.isnan(arr).sum())
        if np.isinf(arr).any():
            norm_inf += int(np.isinf(arr).sum())
        norm_min = min(norm_min, float(np.nanmin(arr)))
        norm_max = max(norm_max, float(np.nanmax(arr)))
        # running mean/var
        n = arr.size
        if n > 0:
            norm_n += n
            norm_mean_acc += float(np.nanmean(arr)) * n
            norm_var_acc += float(np.nanvar(arr)) * n
        crop_shapes.append((int(arr.shape[-2]), int(arr.shape[-1])))

    norm_mean = norm_mean_acc / norm_n if norm_n > 0 else None
    norm_std = math.sqrt(norm_var_acc / norm_n) if norm_n > 0 else None

    # Build markdown
    lines: List[str] = []
    lines.append("# Training Procedure Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- pos_dir: `{config.pos_dir}`")
    lines.append(f"- neg_dir: `{config.neg_dir}`")
    lines.append(f"- split_strategy: `{config.split_strategy}` ({split_source})")
    lines.append(f"- train_ratio: {config.train_ratio}")
    lines.append(f"- val_ratio: {config.val_ratio}")
    lines.append(f"- seed: {config.seed}")
    lines.append(f"- min_gap_seconds: {config.min_gap_seconds}")
    lines.append(f"- crop_size: {_fmt(config.crop_size)}")
    lines.append(f"- min_db/max_db: {config.min_db} / {config.max_db}")
    lines.append(f"- center_bias_sigma_frac: {config.center_bias_sigma_frac}")
    lines.append(f"- sample_size per class: {config.sample_size}")
    lines.append(f"- use_fixed_samples: {config.use_fixed_samples}")
    if config.use_fixed_samples and fixed_pos_names:
        extra = " (fallback to random)" if fixed_pos_fallback else ""
        lines.append(f"- fixed_pos_names: {len(fixed_pos_names)} (resolved: {len(pos_samples)}, missing: {len(fixed_pos_missing)}){extra}")
    if config.use_fixed_samples and fixed_neg_names:
        extra = " (fallback to random)" if fixed_neg_fallback else ""
        lines.append(f"- fixed_neg_names: {len(fixed_neg_names)} (resolved: {len(neg_samples)}, missing: {len(fixed_neg_missing)}){extra}")
    lines.append(f"- n_augment per positive: {config.n_augment}")
    if config.copy_sample and config.sample_out_dir:
        lines.append(f"- sample copied to: `{config.sample_out_dir}`")
    lines.append("")

    if config.use_fixed_samples and (fixed_pos_missing or fixed_neg_missing):
        lines.append("## Fixed Sample Resolution")
        if fixed_pos_missing:
            preview = ", ".join(fixed_pos_missing[:10])
            suffix = " ..." if len(fixed_pos_missing) > 10 else ""
            lines.append(f"- missing pos samples (showing up to 10): {preview}{suffix}")
        if fixed_neg_missing:
            preview = ", ".join(fixed_neg_missing[:10])
            suffix = " ..." if len(fixed_neg_missing) > 10 else ""
            lines.append(f"- missing neg samples (showing up to 10): {preview}{suffix}")
        lines.append("")

    lines.append("## Split Summary")
    if split_counts and missing_time:
        for split in ("train", "val", "test"):
            c = split_counts[split]
            lines.append(f"- {split}: pos={c['pos']} neg={c['neg']} total={c['total']}")
        lines.append("")
        lines.append("### Duplicate Files Across Splits")
        lines.append(f"- train ∩ val: {len(dup_train_val)}")
        lines.append(f"- train ∩ test: {len(dup_train_test)}")
        lines.append(f"- val ∩ test: {len(dup_val_test)}")
        lines.append("")
        lines.append("### Positive Window Overlap (Same Source)")
        lines.append(f"- train ↔ val overlaps: {ov_train_val}")
        lines.append(f"- train missing time info (pos): {missing_time['train']}")
        if ex_train_val:
            lines.append(f"- train ↔ val examples: {ex_train_val[0][0].name} | {ex_train_val[0][1].name}")
        lines.append(f"- train ↔ test overlaps: {ov_train_test}")
        lines.append(f"- val missing time info (pos): {missing_time['val']}")
        if ex_train_test:
            lines.append(f"- train ↔ test examples: {ex_train_test[0][0].name} | {ex_train_test[0][1].name}")
        lines.append(f"- val ↔ test overlaps: {ov_val_test}")
        lines.append(f"- test missing time info (pos): {missing_time['test']}")
        if ex_val_test:
            lines.append(f"- val ↔ test examples: {ex_val_test[0][0].name} | {ex_val_test[0][1].name}")
        lines.append("")
    else:
        lines.append("- Split checks skipped (no full scan and no split files provided).")
        lines.append("")

    lines.append("## Crop Size Behavior")
    if raw_shapes:
        f_vals = [s[0] for s in raw_shapes]
        t_vals = [s[1] for s in raw_shapes]
        lines.append(f"- raw F bins: min={min(f_vals)} max={max(f_vals)}")
        lines.append(f"- raw T bins: min={min(t_vals)} max={max(t_vals)}")
    if crop_shapes:
        uniq = sorted(set(crop_shapes))
        lines.append(f"- cropped shapes (F,T) unique: {uniq[:6]}{' ...' if len(uniq) > 6 else ''}")
    lines.append(f"- freq crop: {freq_crop_counts}")
    lines.append(f"- time crop: {time_crop_counts}")
    lines.append("")

    lines.append("## Augmentation Checks (Positives)")
    if peak_total > 0:
        outside_pct = 100.0 * peak_outside / peak_total
        lines.append(f"- peak inside crop: {100.0 - outside_pct:.3f}% ({peak_total - peak_outside}/{peak_total})")
    else:
        lines.append("- peak inside crop: N/A (no valid peaks)")
    if peak_offset_abs:
        lines.append(f"- peak offset from center (fraction of half-range): min={min(peak_offset_abs):.4f} mean={np.mean(peak_offset_abs):.4f} max={max(peak_offset_abs):.4f}")
    if crop_center_offset_frac:
        lines.append(f"- crop center offset from spec center (fraction): min={min(crop_center_offset_frac):.4f} mean={np.mean(crop_center_offset_frac):.4f} max={max(crop_center_offset_frac):.4f}")
    lines.append("")

    lines.append("## Normalization Checks")
    lines.append(f"- spec kinds: {spec_kind_counts}")
    if clip_total > 0:
        lines.append(f"- raw dB clipped below min_db: {100.0 * clip_below / clip_total:.3f}%")
        lines.append(f"- raw dB clipped above max_db: {100.0 * clip_above / clip_total:.3f}%")
    if norm_n > 0:
        lines.append(f"- normalized range: min={norm_min:.4f} max={norm_max:.4f}")
        lines.append(f"- normalized mean/std: mean={_fmt(norm_mean)} std={_fmt(norm_std)}")
    lines.append(f"- NaN count: {norm_nan} | Inf count: {norm_inf}")
    lines.append("")

    lines.append("## Findings")
    findings = []
    if dup_train_val or dup_train_test or dup_val_test:
        findings.append("Duplicate files exist across splits.")
    if ov_train_val or ov_train_test or ov_val_test:
        findings.append("Positive windows overlap across splits; risk of leakage.")
    if norm_min < -1e-3 or norm_max > 1.001:
        findings.append("Normalized values outside [0,1] range.")
    if norm_nan > 0 or norm_inf > 0:
        findings.append("NaN/Inf present after normalization.")
    if not findings:
        findings.append("No immediate red flags detected in sampled checks.")
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FinWhale training procedure and dataset prep")
    parser.add_argument("--pos-dir", type=str, default=None, help="Positive MAT directory")
    parser.add_argument("--neg-dir", type=str, default=None, help="Negative MAT directory")
    parser.add_argument("--exp-dir", type=str, default=None, help="Experiment dir with args.pkl and optional splits/")
    parser.add_argument("--splits-dir", type=str, default=None, help="Directory with train/val/test.txt split files")
    parser.add_argument("--crop-size", type=str, default=None, help="Crop size (int or 'freq,time')")
    parser.add_argument("--min-db", type=float, default=None, help="Min dB for normalization")
    parser.add_argument("--max-db", type=float, default=None, help="Max dB for normalization")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-strategy", type=str, default=None, choices=["internal", "group_by_source", "time_separated"])
    parser.add_argument("--min-gap-seconds", type=float, default=None)
    parser.add_argument("--center-bias-sigma-frac", type=float, default=0.25)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--n-augment", type=int, default=5)
    parser.add_argument("--max-overlap-count", type=int, default=100000)
    parser.add_argument("--out", type=str, default="output/training_audit.md")
    parser.add_argument("--copy-sample", action="store_true", help="Copy sampled MATs to sample-out-dir")
    parser.add_argument("--sample-out-dir", type=str, default=None, help="Output dir for copied sample")
    parser.add_argument("--use-fixed-samples", action="store_true",
                        help="Use hardcoded sample filename list (pos) and optional fixed-samples-file")
    parser.add_argument("--random-sample", action="store_true",
                        help="Force random sampling even if fixed list exists")
    parser.add_argument("--fixed-samples-file", type=str, default=None,
                        help="Optional file with additional sample basenames (one per line)")
    parser.add_argument("--fixed-neg-samples-file", type=str, default=None,
                        help="Optional file with negative sample basenames (one per line)")
    parser.add_argument("--progress-interval", type=int, default=25,
                        help="Progress print interval for sample processing")
    parser.add_argument("--scan-full", action="store_true",
                        help="Scan full directories to compute splits and random samples (slow on mounted drives)")
    args = parser.parse_args()

    # Load defaults from args.pkl if present
    args_pkl = {}
    exp_dir = Path(args.exp_dir) if args.exp_dir else None
    if exp_dir and (exp_dir / "args.pkl").exists():
        args_pkl = _load_args_pkl(exp_dir / "args.pkl")

    pos_dir_raw = args.pos_dir or args_pkl.get("pos_dir")
    neg_dir_raw = args.neg_dir or args_pkl.get("neg_dir")
    if not pos_dir_raw or not neg_dir_raw:
        raise SystemExit("pos-dir and neg-dir are required (or must exist in args.pkl)")
    pos_dir = Path(str(pos_dir_raw))
    neg_dir = Path(str(neg_dir_raw))
    if not pos_dir.exists() or not neg_dir.exists():
        raise SystemExit(f"pos-dir or neg-dir does not exist: {pos_dir} | {neg_dir}")

    crop_size = _parse_crop_arg(args.crop_size) if args.crop_size is not None else _parse_crop_arg(args_pkl.get("crop_size"))
    min_db = args.min_db if args.min_db is not None else float(args_pkl.get("min_db", -80.0))
    max_db = args.max_db if args.max_db is not None else float(args_pkl.get("max_db", 0.0))
    train_ratio = args.train_ratio if args.train_ratio is not None else float(args_pkl.get("train_ratio", 0.8))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(args_pkl.get("val_ratio", 0.1))
    seed = args.seed if args.seed is not None else int(args_pkl.get("seed", 42))
    split_strategy = args.split_strategy or str(args_pkl.get("split_strategy", "internal"))
    min_gap_seconds = args.min_gap_seconds if args.min_gap_seconds is not None else float(args_pkl.get("min_gap_seconds", 120.0))

    splits_dir = Path(args.splits_dir) if args.splits_dir else (exp_dir / "splits" if exp_dir else None)

    sample_out_dir = Path(args.sample_out_dir) if args.sample_out_dir else None
    if args.copy_sample and sample_out_dir is None:
        sample_out_dir = Path("output/audit_sample")

    use_fixed_samples = bool(args.use_fixed_samples)
    if args.copy_sample and not args.random_sample:
        # Default to fixed samples when copying, unless explicitly overridden.
        use_fixed_samples = True

    scan_full = bool(args.scan_full)
    if not scan_full and not use_fixed_samples and not args.splits_dir and not (exp_dir and (exp_dir / "splits").exists()):
        _log("No full scan, no fixed samples, and no split files. Enable --scan-full or --use-fixed-samples.")
        raise SystemExit(2)

    config = AuditConfig(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        crop_size=crop_size,
        min_db=min_db,
        max_db=max_db,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        split_strategy=split_strategy,
        min_gap_seconds=min_gap_seconds,
        center_bias_sigma_frac=args.center_bias_sigma_frac,
        sample_size=args.sample_size,
        n_augment=args.n_augment,
        out_path=Path(args.out),
        splits_dir=splits_dir if splits_dir and splits_dir.exists() else None,
        copy_sample=bool(args.copy_sample),
        sample_out_dir=sample_out_dir,
        max_overlap_count=args.max_overlap_count,
        use_fixed_samples=use_fixed_samples,
        fixed_samples_file=Path(args.fixed_samples_file) if args.fixed_samples_file else None,
        fixed_neg_samples_file=Path(args.fixed_neg_samples_file) if args.fixed_neg_samples_file else None,
        progress_interval=int(args.progress_interval),
        scan_full=scan_full,
    )

    md = audit(config)
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text(md)
    print(f"Wrote audit summary to {config.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
