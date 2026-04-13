from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
import torch
import torch.nn.functional as F

from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus, HiResCAM, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.fin_models import create_model
from src.training.mat_dataset import (
    DB_KEYS,
    FREQ_KEYS,
    POWER_KEYS,
    SPECTRO_KEYS,
    TIME_KEYS,
    _find_key,
    _infer_time_bin_seconds,
    _normalize_db_to_unit,
    _power_to_db_norm,
    parse_crop_size,
)
from src.utils.model_utils import extract_model_info


CAM_METHODS = {
    "gradcampp": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}

FIN_BUCKET_FREQ_PRIORS = {
    "20Hz": (12.0, 30.0),
    "40Hz": (30.0, 55.0),
    "other_fin": (10.0, 60.0),
}


@dataclass(frozen=True)
class AnnotationBox:
    annotation_id: str
    filename: str
    species: str
    call_type_bucket: str
    call_type_raw: str
    begin_time_s: float
    end_time_s: float
    low_freq_hz: Optional[float]
    high_freq_hz: Optional[float]
    peak_freq_hz: Optional[float]
    context_tags: Tuple[str, ...]
    comments: str = ""


@dataclass(frozen=True)
class CropSpec:
    annotation_id: str
    filename: str
    mat_path: Path
    label: int
    crop_name: str
    time_start_idx: int
    time_end_idx: int
    freq_start_idx: int
    freq_end_idx: int
    time_start_s: float
    time_end_s: float
    freq_low_hz: float
    freq_high_hz: float
    annotation_box_bins: Optional[Tuple[int, int, int, int]]
    annotation_box_seconds_hz: Optional[Tuple[float, float, float, float]]
    has_valid_box: bool
    call_type_bucket: str
    context_tags: Tuple[str, ...]
    source: str


@dataclass(frozen=True)
class AttentionArtifacts:
    heatmap: np.ndarray
    mask: np.ndarray
    top_box: Optional[Tuple[int, int, int, int]]
    peak_bin: Tuple[int, int]
    score: float
    score_after_mask: Optional[float]


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _split_tags(raw: str) -> Tuple[str, ...]:
    tags = [token.strip() for token in str(raw or "").split("|") if token.strip()]
    return tuple(sorted(dict.fromkeys(tags)))


def _relative_box_to_absolute(
    box: Tuple[int, int, int, int],
    crop: CropSpec,
) -> Tuple[float, float, float, float]:
    t0, t1, f0, f1 = box
    return (
        _bin_to_time_seconds(t0 + crop.time_start_idx, crop.time_start_s, crop.time_end_s, crop.time_end_idx - crop.time_start_idx),
        _bin_to_time_seconds(t1 + crop.time_start_idx, crop.time_start_s, crop.time_end_s, crop.time_end_idx - crop.time_start_idx),
        _bin_to_freq_hz(f0 + crop.freq_start_idx, crop.freq_low_hz, crop.freq_high_hz, crop.freq_end_idx - crop.freq_start_idx),
        _bin_to_freq_hz(f1 + crop.freq_start_idx, crop.freq_low_hz, crop.freq_high_hz, crop.freq_end_idx - crop.freq_start_idx),
    )


def _bin_to_time_seconds(bin_idx: int, start_s: float, end_s: float, total_bins: int) -> float:
    span = max(end_s - start_s, 1e-9)
    denom = max(total_bins - 1, 1)
    return float(start_s + (span * float(bin_idx) / float(denom)))


def _bin_to_freq_hz(bin_idx: int, low_hz: float, high_hz: float, total_bins: int) -> float:
    span = max(high_hz - low_hz, 1e-9)
    denom = max(total_bins - 1, 1)
    return float(low_hz + (span * float(bin_idx) / float(denom)))


def _time_to_bin(time_s: float, start_s: float, end_s: float, total_bins: int) -> int:
    if total_bins <= 1 or end_s <= start_s:
        return 0
    frac = (float(time_s) - float(start_s)) / max(float(end_s - start_s), 1e-9)
    return int(np.clip(round(frac * (total_bins - 1)), 0, total_bins - 1))


def _freq_to_bin(freq_hz: float, low_hz: float, high_hz: float, total_bins: int) -> int:
    if total_bins <= 1 or high_hz <= low_hz:
        return 0
    frac = (float(freq_hz) - float(low_hz)) / max(float(high_hz - low_hz), 1e-9)
    return int(np.clip(round(frac * (total_bins - 1)), 0, total_bins - 1))


def load_localized_annotations(path: Path | str) -> List[AnnotationBox]:
    out: List[AnnotationBox] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            begin = _safe_float(row.get("begin_time_s"))
            end = _safe_float(row.get("end_time_s"))
            if begin is None or end is None or end <= begin:
                continue
            out.append(
                AnnotationBox(
                    annotation_id=str(row.get("annotation_id") or f"ann_{idx:06d}"),
                    filename=filename,
                    species=str(row.get("species", "")).strip(),
                    call_type_bucket=str(row.get("call_type_bucket", "")).strip(),
                    call_type_raw=str(row.get("call_type_raw", "")).strip(),
                    begin_time_s=float(begin),
                    end_time_s=float(end),
                    low_freq_hz=_safe_float(row.get("low_freq_hz")),
                    high_freq_hz=_safe_float(row.get("high_freq_hz")),
                    peak_freq_hz=_safe_float(row.get("peak_freq_hz")),
                    context_tags=_split_tags(row.get("context_tags", "")),
                    comments=str(row.get("comments", "")).strip(),
                )
            )
    return out


def build_mat_lookup(mat_dir: Path | str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(Path(mat_dir).glob("*.mat")):
        name = path.name
        out.setdefault(name, path.resolve())
        for suffix in (".wav", ".flac"):
            marker = f"{suffix}_"
            if marker in name:
                source_name = name[: name.index(marker) + len(suffix)]
                out.setdefault(source_name, path.resolve())
    return out


def load_model_checkpoint(checkpoint_path: Path | str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model_info = extract_model_info(checkpoint)
    architecture = str(model_info["architecture"])
    model = create_model(architecture, num_classes=2, in_ch=1).to(device)
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    meta = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "architecture": architecture,
        "training_args": checkpoint.get("training_args", {}) if isinstance(checkpoint, dict) else {},
        "val_metrics": checkpoint.get("val_metrics", {}) if isinstance(checkpoint, dict) else {},
    }
    return model, meta


def resolve_target_layers(model: torch.nn.Module, layer_preset: str = "last") -> List[torch.nn.Module]:
    net = getattr(model, "net", model)
    if not hasattr(net, "layer4"):
        raise ValueError("attention experiment currently expects a ResNet-style model with layer4")
    if layer_preset == "last":
        return [net.layer4[-1]]
    if layer_preset == "late":
        return [net.layer3[-1], net.layer4[-1]]
    if layer_preset == "hierarchical":
        return [net.layer2[-1], net.layer3[-1], net.layer4[-1]]
    raise ValueError(f"unknown layer preset: {layer_preset}")


def _load_mat_spectrogram(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = sio.loadmat(str(path), simplify_cells=True)
    power_key = _find_key(data, POWER_KEYS)
    spec_kind = "power"
    key = power_key
    if key is None:
        key = _find_key(data, DB_KEYS) or _find_key(data, SPECTRO_KEYS)
        spec_kind = "db"
    if key is None:
        raise KeyError(f"No spectrogram key found in {path}")
    spec = np.asarray(data[key])
    if spec.ndim != 2:
        raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {path}")
    freq_key = _find_key(data, FREQ_KEYS)
    time_key = _find_key(data, TIME_KEYS)
    if freq_key is None or time_key is None:
        raise KeyError(f"Missing frequency/time vectors in {path}")
    freqs = np.asarray(data[freq_key]).squeeze().astype(np.float32)
    times = np.asarray(data[time_key]).squeeze().astype(np.float32)
    if spec.shape == (times.shape[0], freqs.shape[0]):
        spec = spec.T
    if spec.shape != (freqs.shape[0], times.shape[0]):
        raise ValueError(f"Spectrogram shape {spec.shape} does not match axes in {path}")
    if spec_kind == "power":
        spec = _power_to_db_norm(spec)
    spec = _normalize_db_to_unit(spec, min_db=-80.0, max_db=0.0)
    return spec.astype(np.float32), freqs.astype(np.float32), times.astype(np.float32)


def infer_crop_geometry(
    *,
    spec: np.ndarray,
    times: np.ndarray,
    training_args: Mapping[str, Any],
) -> Tuple[int, int]:
    crop_size = training_args.get("crop_size") if isinstance(training_args, Mapping) else None
    if isinstance(crop_size, np.ndarray):
        if crop_size.ndim == 0:
            crop_size = crop_size.item()
        else:
            crop_size = crop_size.tolist()
    if isinstance(crop_size, str):
        text = crop_size.strip()
        if not text:
            crop_size = None
        elif "," in text:
            crop_size = [int(token.strip()) for token in text.split(",") if token.strip()]
        else:
            crop_size = int(text)
    elif isinstance(crop_size, np.generic):
        crop_size = crop_size.item()
    freq_crop, time_crop = parse_crop_size(crop_size)
    freq_bins = int(freq_crop) if freq_crop is not None else int(spec.shape[0])
    crop_time_seconds = _safe_float(training_args.get("crop_time_seconds")) if isinstance(training_args, Mapping) else None
    if crop_time_seconds is not None:
        dt = _infer_time_bin_seconds(times)
        if dt is not None and dt > 0:
            time_bins = max(1, int(round(crop_time_seconds / dt)))
        elif time_crop is not None:
            time_bins = int(time_crop)
        else:
            time_bins = int(freq_bins)
    else:
        time_bins = int(time_crop) if time_crop is not None else int(freq_bins)
    return freq_bins, time_bins


def derive_annotation_frequency_bounds(annotation: AnnotationBox) -> Tuple[Optional[float], Optional[float], str]:
    if annotation.low_freq_hz is not None and annotation.high_freq_hz is not None and annotation.high_freq_hz > annotation.low_freq_hz:
        return annotation.low_freq_hz, annotation.high_freq_hz, "annotation"
    prior = FIN_BUCKET_FREQ_PRIORS.get(annotation.call_type_bucket)
    if prior is not None:
        return prior[0], prior[1], "bucket_prior"
    return None, None, "missing"


def build_annotation_crop(
    annotation: AnnotationBox,
    *,
    mat_path: Path,
    training_args: Mapping[str, Any],
    crop_pad_time_fraction: float = 0.25,
) -> Tuple[torch.Tensor, CropSpec, np.ndarray]:
    spec, freqs, times = _load_mat_spectrogram(mat_path)
    freq_bins, time_bins = infer_crop_geometry(spec=spec, times=times, training_args=training_args)
    full_f, full_t = spec.shape

    time_center = 0.5 * (annotation.begin_time_s + annotation.end_time_s)
    center_idx = _time_to_bin(time_center, float(times[0]), float(times[-1]), full_t)
    time_start = int(np.clip(center_idx - (time_bins // 2), 0, max(full_t - time_bins, 0)))
    time_end = min(full_t, time_start + time_bins)
    if time_end - time_start < time_bins:
        time_start = max(0, time_end - time_bins)

    if full_f <= freq_bins:
        freq_start = 0
        freq_end = full_f
    else:
        low_hz, high_hz, _ = derive_annotation_frequency_bounds(annotation)
        if low_hz is None or high_hz is None:
            peak_hz = annotation.peak_freq_hz if annotation.peak_freq_hz is not None else float(freqs[len(freqs) // 2])
            low_hz = peak_hz
            high_hz = peak_hz
        freq_center = 0.5 * (float(low_hz) + float(high_hz))
        freq_center_idx = _freq_to_bin(freq_center, float(freqs[0]), float(freqs[-1]), full_f)
        freq_start = int(np.clip(freq_center_idx - (freq_bins // 2), 0, max(full_f - freq_bins, 0)))
        freq_end = min(full_f, freq_start + freq_bins)
        if freq_end - freq_start < freq_bins:
            freq_start = max(0, freq_end - freq_bins)

    crop = spec[freq_start:freq_end, time_start:time_end]
    if crop.shape[0] < freq_bins:
        crop = np.pad(crop, ((0, freq_bins - crop.shape[0]), (0, 0)), mode="edge")
    if crop.shape[1] < time_bins:
        crop = np.pad(crop, ((0, 0), (0, time_bins - crop.shape[1])), mode="edge")

    gt_box_bins: Optional[Tuple[int, int, int, int]] = None
    gt_box_absolute: Optional[Tuple[float, float, float, float]] = None
    low_hz, high_hz, freq_source = derive_annotation_frequency_bounds(annotation)
    if low_hz is not None and high_hz is not None:
        time0 = _time_to_bin(annotation.begin_time_s, float(times[time_start]), float(times[min(time_end - 1, len(times) - 1)]), crop.shape[1])
        time1 = _time_to_bin(annotation.end_time_s, float(times[time_start]), float(times[min(time_end - 1, len(times) - 1)]), crop.shape[1])
        freq0 = _freq_to_bin(low_hz, float(freqs[freq_start]), float(freqs[min(freq_end - 1, len(freqs) - 1)]), crop.shape[0])
        freq1 = _freq_to_bin(high_hz, float(freqs[freq_start]), float(freqs[min(freq_end - 1, len(freqs) - 1)]), crop.shape[0])
        t0, t1 = sorted((time0, time1))
        f0, f1 = sorted((freq0, freq1))
        gt_box_bins = (
            int(np.clip(t0, 0, crop.shape[1] - 1)),
            int(np.clip(max(t0 + 1, t1), 1, crop.shape[1])),
            int(np.clip(f0, 0, crop.shape[0] - 1)),
            int(np.clip(max(f0 + 1, f1), 1, crop.shape[0])),
        )
        gt_box_absolute = (annotation.begin_time_s, annotation.end_time_s, float(low_hz), float(high_hz))

    crop_spec = CropSpec(
        annotation_id=annotation.annotation_id,
        filename=annotation.filename,
        mat_path=mat_path.resolve(),
        label=1 if annotation.species == "Bp" else 0,
        crop_name=f"{annotation.annotation_id}_{annotation.filename}",
        time_start_idx=int(time_start),
        time_end_idx=int(time_start + crop.shape[1]),
        freq_start_idx=int(freq_start),
        freq_end_idx=int(freq_start + crop.shape[0]),
        time_start_s=float(times[time_start]),
        time_end_s=float(times[min(time_end - 1, len(times) - 1)]),
        freq_low_hz=float(freqs[freq_start]),
        freq_high_hz=float(freqs[min(freq_end - 1, len(freqs) - 1)]),
        annotation_box_bins=gt_box_bins,
        annotation_box_seconds_hz=gt_box_absolute,
        has_valid_box=gt_box_bins is not None,
        call_type_bucket=annotation.call_type_bucket,
        context_tags=annotation.context_tags,
        source=freq_source,
    )
    return torch.from_numpy(crop[None, None, :, :]).float(), crop_spec, crop


def build_negative_crop_from_annotation(
    annotation: AnnotationBox,
    *,
    mat_path: Path,
    training_args: Mapping[str, Any],
) -> Tuple[torch.Tensor, CropSpec, np.ndarray]:
    tensor, crop_spec, crop = build_annotation_crop(annotation, mat_path=mat_path, training_args=training_args)
    crop_spec = CropSpec(
        **{
            **asdict(crop_spec),
            "label": 0,
            "annotation_box_bins": None,
            "annotation_box_seconds_hz": None,
            "has_valid_box": False,
        }
    )
    return tensor, crop_spec, crop


def _normalize_attention_map(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    arr = np.maximum(arr, 0.0)
    max_val = float(np.max(arr)) if arr.size else 0.0
    if max_val <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    arr = arr / max_val
    return np.clip(arr, 0.0, 1.0)


def _threshold_mask(heatmap: np.ndarray, threshold_rel: float = 0.6, min_pixels: int = 4) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        return np.zeros_like(heatmap, dtype=bool)
    threshold = float(np.max(heatmap)) * float(threshold_rel)
    mask = heatmap >= threshold
    if int(mask.sum()) >= int(min_pixels):
        return mask
    if heatmap.size >= int(min_pixels):
        flat = heatmap.ravel()
        idx = np.argpartition(flat, -int(min_pixels))[-int(min_pixels) :]
        new_mask = np.zeros_like(flat, dtype=bool)
        new_mask[idx] = True
        return new_mask.reshape(heatmap.shape)
    return mask


def _largest_component_box(mask: np.ndarray, heatmap: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if not np.any(mask):
        return None
    labels, count = ndimage.label(mask.astype(np.uint8))
    if count <= 0:
        return None
    best_label = None
    best_score = -1.0
    for label in range(1, count + 1):
        comp = labels == label
        score = float(np.sum(np.asarray(heatmap, dtype=np.float32)[comp]))
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        return None
    ys, xs = np.where(labels == best_label)
    return (
        int(xs.min()),
        int(xs.max()) + 1,
        int(ys.min()),
        int(ys.max()) + 1,
    )


def compute_top_region_confidence_drop(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    *,
    target_class: int,
    baseline_value: float = 0.0,
    threshold_rel: float = 0.6,
) -> float:
    mask = _threshold_mask(heatmap, threshold_rel=threshold_rel)
    masked = input_tensor.clone()
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(masked.device)[None, None, :, :]
    masked = masked * (1.0 - mask_tensor) + (mask_tensor * float(baseline_value))
    with torch.no_grad():
        base_score = F.softmax(model(input_tensor), dim=1)[0, target_class].item()
        masked_score = F.softmax(model(masked), dim=1)[0, target_class].item()
    return float(base_score - masked_score)


def _run_cam_method(
    *,
    method_name: str,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layers: Sequence[torch.nn.Module],
    target_class: int,
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
) -> np.ndarray:
    cam_cls = CAM_METHODS[method_name]
    targets = [ClassifierOutputTarget(target_class)]
    uses_grad = method_name != "eigencam"
    with cam_cls(model=model, target_layers=list(target_layers)) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets if uses_grad else None,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )
    return np.asarray(grayscale_cam[0], dtype=np.float32)


def _run_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    target_class: int,
    steps: int = 32,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    baseline = torch.zeros_like(input_tensor)
    total_grad = torch.zeros_like(input_tensor)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=input_tensor.device)[1:]
    for alpha in alphas:
        sample = baseline + alpha * (input_tensor - baseline)
        sample.requires_grad_(True)
        logits = model(sample)
        score = logits[:, target_class].sum()
        grads = torch.autograd.grad(score, sample, retain_graph=False, create_graph=False)[0]
        total_grad += grads.detach()
    attributions = (input_tensor - baseline) * total_grad / float(max(len(alphas), 1))
    heatmap = attributions.abs().sum(dim=1).detach().cpu().numpy()[0]
    return heatmap.astype(np.float32)


def _run_occlusion(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    target_class: int,
    window: Tuple[int, int] = (8, 8),
    stride: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    with torch.no_grad():
        base_score = F.softmax(model(input_tensor), dim=1)[0, target_class].item()
    _, _, height, width = input_tensor.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    win_h, win_w = window
    step_h, step_w = stride
    for top in range(0, max(1, height - win_h + 1), step_h):
        for left in range(0, max(1, width - win_w + 1), step_w):
            bottom = min(height, top + win_h)
            right = min(width, left + win_w)
            masked = input_tensor.clone()
            masked[:, :, top:bottom, left:right] = 0.0
            with torch.no_grad():
                masked_score = F.softmax(model(masked), dim=1)[0, target_class].item()
            delta = float(base_score - masked_score)
            heatmap[top:bottom, left:right] += delta
            counts[top:bottom, left:right] += 1.0
    counts = np.maximum(counts, 1.0)
    return heatmap / counts


def generate_attention_artifacts(
    *,
    method_name: str,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layers: Sequence[torch.nn.Module],
    target_class: int = 1,
    threshold_rel: float = 0.6,
    integrated_gradients_steps: int = 32,
    occlusion_window: Tuple[int, int] = (8, 8),
    occlusion_stride: Tuple[int, int] = (4, 4),
    confidence_drop: bool = True,
) -> AttentionArtifacts:
    input_tensor = input_tensor.detach()
    with torch.no_grad():
        score = F.softmax(model(input_tensor), dim=1)[0, target_class].item()
    if method_name in CAM_METHODS:
        raw = _run_cam_method(
            method_name=method_name,
            model=model,
            input_tensor=input_tensor,
            target_layers=target_layers,
            target_class=target_class,
        )
    elif method_name == "integrated_gradients":
        raw = _run_integrated_gradients(model, input_tensor, target_class=target_class, steps=integrated_gradients_steps)
    elif method_name == "occlusion":
        raw = _run_occlusion(
            model,
            input_tensor,
            target_class=target_class,
            window=occlusion_window,
            stride=occlusion_stride,
        )
    else:
        raise ValueError(f"unknown attention method: {method_name}")

    heatmap = _normalize_attention_map(raw)
    mask = _threshold_mask(heatmap, threshold_rel=threshold_rel)
    top_box = _largest_component_box(mask, heatmap)
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    score_after_mask = None
    if confidence_drop:
        score_after_mask = score - compute_top_region_confidence_drop(
            model,
            input_tensor,
            heatmap,
            target_class=target_class,
            baseline_value=0.0,
            threshold_rel=threshold_rel,
        )
    return AttentionArtifacts(
        heatmap=heatmap,
        mask=mask,
        top_box=top_box,
        peak_bin=(int(peak_idx[1]), int(peak_idx[0])),
        score=float(score),
        score_after_mask=score_after_mask,
    )


def box_iou(box_a: Optional[Tuple[int, int, int, int]], box_b: Optional[Tuple[int, int, int, int]]) -> Optional[float]:
    if box_a is None or box_b is None:
        return None
    ax0, ax1, ay0, ay1 = box_a
    bx0, bx1, by0, by1 = box_b
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = max(area_a + area_b - inter, 1)
    return float(inter / denom)


def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return float(inter / max(union, 1))


def mask_overlap_metrics(mask: np.ndarray, gt_box: Optional[Tuple[int, int, int, int]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if gt_box is None:
        return None, None, None
    mask = np.asarray(mask, dtype=bool)
    gt_mask = np.zeros_like(mask, dtype=bool)
    x0, x1, y0, y1 = gt_box
    gt_mask[y0:y1, x0:x1] = True
    inter = float(np.logical_and(mask, gt_mask).sum())
    pred = float(mask.sum())
    truth = float(gt_mask.sum())
    union = float(np.logical_or(mask, gt_mask).sum())
    coverage = inter / truth if truth > 0 else None
    precision = inter / pred if pred > 0 else None
    iou = inter / union if union > 0 else None
    return coverage, precision, iou


def pointing_hit(peak_bin: Tuple[int, int], gt_box: Optional[Tuple[int, int, int, int]]) -> Optional[float]:
    if gt_box is None:
        return None
    x, y = peak_bin
    x0, x1, y0, y1 = gt_box
    return 1.0 if (x0 <= x < x1 and y0 <= y < y1) else 0.0


def summarize_localization(
    crop: CropSpec,
    artifacts: AttentionArtifacts,
    *,
    method_name: str,
    model_label: str,
) -> Dict[str, Any]:
    gt_box = crop.annotation_box_bins
    pred_box = artifacts.top_box
    coverage, mask_precision, mask_iou = mask_overlap_metrics(artifacts.mask, gt_box)
    temporal = None
    frequency = None
    if pred_box is not None and gt_box is not None:
        temporal = interval_iou(pred_box[0], pred_box[1], gt_box[0], gt_box[1])
        frequency = interval_iou(pred_box[2], pred_box[3], gt_box[2], gt_box[3])
    top_box_abs = _relative_box_to_absolute(pred_box, crop) if pred_box is not None else None
    return {
        "model_label": model_label,
        "method": method_name,
        "annotation_id": crop.annotation_id,
        "filename": crop.filename,
        "call_type_bucket": crop.call_type_bucket,
        "context_tags": "|".join(crop.context_tags),
        "score": artifacts.score,
        "score_after_mask": artifacts.score_after_mask,
        "time_start_s": crop.time_start_s,
        "time_end_s": crop.time_end_s,
        "freq_low_hz": crop.freq_low_hz,
        "freq_high_hz": crop.freq_high_hz,
        "has_valid_box": int(crop.has_valid_box),
        "box_iou": box_iou(pred_box, gt_box),
        "temporal_iou": temporal,
        "frequency_iou": frequency,
        "pointing_hit": pointing_hit(artifacts.peak_bin, gt_box),
        "mask_coverage": coverage,
        "mask_precision": mask_precision,
        "mask_iou": mask_iou,
        "peak_time_bin": artifacts.peak_bin[0],
        "peak_freq_bin": artifacts.peak_bin[1],
        "pred_box_t0": pred_box[0] if pred_box is not None else None,
        "pred_box_t1": pred_box[1] if pred_box is not None else None,
        "pred_box_f0": pred_box[2] if pred_box is not None else None,
        "pred_box_f1": pred_box[3] if pred_box is not None else None,
        "pred_box_time_start_s": top_box_abs[0] if top_box_abs is not None else None,
        "pred_box_time_end_s": top_box_abs[1] if top_box_abs is not None else None,
        "pred_box_freq_low_hz": top_box_abs[2] if top_box_abs is not None else None,
        "pred_box_freq_high_hz": top_box_abs[3] if top_box_abs is not None else None,
        "gt_box_source": crop.source,
    }


def aggregate_metric_rows(rows: Sequence[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_key, "")).strip() or "all"
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    metric_names = [
        "box_iou",
        "temporal_iou",
        "frequency_iou",
        "pointing_hit",
        "mask_coverage",
        "mask_precision",
        "mask_iou",
        "score",
    ]
    for key, group_rows in sorted(grouped.items()):
        summary = {group_key: key, "count": len(group_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in (None, "")]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_median"] = float(np.median(values)) if values else None
        out.append(summary)
    return out


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def render_attention_panel(
    *,
    crop_image: np.ndarray,
    artifacts: AttentionArtifacts,
    crop_spec: CropSpec,
    method_name: str,
    model_label: str,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = artifacts.heatmap
    mask = artifacts.mask
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    axes[0].imshow(crop_image, origin="lower", aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0].set_title("Spectrogram")
    axes[1].imshow(crop_image, origin="lower", aspect="auto", cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(heatmap, origin="lower", aspect="auto", cmap="turbo", alpha=0.65, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{method_name} heatmap")
    axes[2].imshow(crop_image, origin="lower", aspect="auto", cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(mask.astype(np.float32), origin="lower", aspect="auto", cmap="viridis", alpha=0.5, vmin=0.0, vmax=1.0)
    axes[2].set_title("Thresholded mask")

    gt_box = crop_spec.annotation_box_bins
    pred_box = artifacts.top_box
    for ax in axes:
        if gt_box is not None:
            x0, x1, y0, y1 = gt_box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="#66ff99", linewidth=2))
        if pred_box is not None:
            x0, x1, y0, y1 = pred_box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="#ffcc33", linewidth=2))
        ax.scatter([artifacts.peak_bin[0]], [artifacts.peak_bin[1]], c="white", s=18)
        ax.set_xlabel("time bins")
        ax.set_ylabel("freq bins")

    fig.suptitle(
        f"{model_label} | {method_name} | {crop_spec.annotation_id} | "
        f"{crop_spec.call_type_bucket} | score={artifacts.score:.3f}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_attention_arrays(
    *,
    artifacts: AttentionArtifacts,
    output_root: Path | str,
    stem: str,
) -> Dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_root / f"{stem}_heatmap.npy"
    mask_path = output_root / f"{stem}_mask.npy"
    np.save(heatmap_path, artifacts.heatmap)
    np.save(mask_path, artifacts.mask.astype(np.uint8))
    return {"heatmap_npy": str(heatmap_path), "mask_npy": str(mask_path)}


def stitch_sliding_window_heatmaps(
    *,
    model: torch.nn.Module,
    method_name: str,
    spec: np.ndarray,
    target_layers: Sequence[torch.nn.Module],
    target_class: int = 1,
    time_bins: int = 96,
    step_bins: int = 24,
    device: torch.device,
) -> np.ndarray:
    full_f, full_t = spec.shape
    if full_t <= time_bins:
        tensor = torch.from_numpy(spec[None, None, :, :]).float().to(device)
        return generate_attention_artifacts(
            method_name=method_name,
            model=model,
            input_tensor=tensor,
            target_layers=target_layers,
            target_class=target_class,
        ).heatmap
    accum = np.zeros((full_f, full_t), dtype=np.float32)
    counts = np.zeros((full_f, full_t), dtype=np.float32)
    for start in range(0, max(1, full_t - time_bins + 1), step_bins):
        stop = min(full_t, start + time_bins)
        window = spec[:, start:stop]
        if window.shape[1] < time_bins:
            window = np.pad(window, ((0, 0), (0, time_bins - window.shape[1])), mode="edge")
        tensor = torch.from_numpy(window[None, None, :, :]).float().to(device)
        artifacts = generate_attention_artifacts(
            method_name=method_name,
            model=model,
            input_tensor=tensor,
            target_layers=target_layers,
            target_class=target_class,
            confidence_drop=False,
        )
        width = stop - start
        accum[:, start:stop] += artifacts.heatmap[:, :width]
        counts[:, start:stop] += 1.0
    counts = np.maximum(counts, 1.0)
    return accum / counts
