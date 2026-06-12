#!/usr/bin/env python3
"""Train SSAMBA on an E123/E126 H5 dataset.

This replaces the missing legacy ``src/run_amba_spectrogram.py`` entrypoint in
the current selfsupervision_anomalies_onc checkout. It intentionally keeps the
same CLI surface expected by ``scripts/run_amba_spectrogram.sh`` while delegating
the actual model, dataset, and training loops to the SSL package.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    parser.add_argument(name, type=parse_bool, default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use_wandb", "--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb_entity", "--wandb-entity", dest="wandb_entity", default=None)
    parser.add_argument("--wandb_project", "--wandb-project", dest="wandb_project", default="multispecies_e123_ssl")
    parser.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="E123_ssl_multispecies")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--data-train", "--data_train", dest="data_train", required=True)
    parser.add_argument("--exp-dir", "--exp_dir", dest="exp_dir", required=True)
    parser.add_argument("--dataset_mean", type=float, default=51.506817)
    parser.add_argument("--dataset_std", type=float, default=13.638703)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=10.0)
    parser.add_argument("--n-epochs", "--n_epochs", dest="n_epochs", type=int, default=200)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--save_model", type=parse_bool, default=True)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--freqm", type=int, default=0)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--bal", default="none")
    parser.add_argument("--tstride", type=int, default=16)
    parser.add_argument("--fstride", type=int, default=16)
    parser.add_argument("--fshape", type=int, default=16)
    parser.add_argument("--tshape", type=int, default=16)
    parser.add_argument("--target_length", type=int, default=512)
    parser.add_argument("--num_mel_bins", type=int, default=512)
    parser.add_argument("--model_size", default="base")
    parser.add_argument("--mask_patch", type=int, default=300)
    parser.add_argument("--n-print-steps", "--n_print_steps", dest="n_print_steps", type=int, default=100)
    parser.add_argument("--task", default="pretrain_joint")
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--epoch_iter", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=24)
    add_bool_arg(parser, "--rms_norm", False)
    add_bool_arg(parser, "--residual_in_fp32", False)
    add_bool_arg(parser, "--fused_add_norm", False)
    add_bool_arg(parser, "--if_rope", False)
    add_bool_arg(parser, "--if_rope_residual", False)
    parser.add_argument("--bimamba_type", default="v2")
    add_bool_arg(parser, "--use_middle_cls_token", False)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--norm_epsilon", type=float, default=1e-5)
    add_bool_arg(parser, "--if_bidirectional", True)
    parser.add_argument("--final_pool_type", default="none")
    add_bool_arg(parser, "--if_abs_pos_embed", True)
    add_bool_arg(parser, "--if_bimamba", False)
    add_bool_arg(parser, "--if_cls_token", True)
    add_bool_arg(parser, "--if_divide_out", True)
    add_bool_arg(parser, "--use_double_cls_token", False)
    parser.add_argument("--main_metric", default=None)
    parser.add_argument("--pretrained_path", "--pretrained-path", dest="pretrained_path", default=None)
    parser.add_argument("--exclude_labels", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--multiclass", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    is_pretrain = "pretrain" in args.task
    args.data_train = str(Path(args.data_train))
    args.exp_dir = str(Path(args.exp_dir))
    args.loss = "BCE"
    args.warmup = True
    args.optim = "adam"
    args.adaptschedule = False
    args.use_tqdm = True
    args.amount = 1.0
    args.ood = -1
    args.subsample_test = False
    if args.main_metric is None:
        args.main_metric = "acc" if is_pretrain else "auc"
    if is_pretrain:
        args.n_class = 2
        args.num_classes = 2
    elif args.multiclass:
        args.num_classes = int(args.num_classes or 4)
        args.n_class = args.num_classes
    else:
        args.n_class = 2
        args.num_classes = 1
    if args.exclude_labels == []:
        args.exclude_labels = None
    return args


def write_run_config(args: argparse.Namespace, unknown_args: Sequence[str]) -> None:
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)
    with (exp_dir / "args.pkl").open("wb") as handle:
        pickle.dump(args, handle)
    payload = vars(args).copy()
    payload["unknown_args"] = list(unknown_args)
    (exp_dir / "e128_runner_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def label_tokens(value: Any) -> List[str]:
    labels = [clean(part) for part in re.split(r"[;,|]", clean(value)) if clean(part)]
    return labels or ["normal"]


def is_normal_label(labels: Sequence[str]) -> bool:
    return all((not clean(label)) or clean(label).lower() == "normal" for label in labels)


def normalize_spectrogram(data: Any, *, dataset_mean: Optional[float], dataset_std: Optional[float], amount: float):
    import numpy as np  # type: ignore

    arr = np.asarray(data, dtype=np.float32)
    if dataset_mean is not None and dataset_std is not None:
        arr = (arr - float(dataset_mean)) / (float(dataset_std) * 2.0)
    else:
        lo, hi = np.percentile(arr, [float(amount), 100.0 - float(amount)])
        arr = np.clip(arr, lo, hi)
        arr = np.log(np.maximum(arr, 1e-12))
        denom = max(float(np.max(arr) - np.min(arr)), 1e-12)
        arr = (arr - float(np.min(arr))) / denom
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class H5SplitSpectrogramDataset:
    """H5 dataset that honors stored split labels instead of re-randomizing rows."""

    def __init__(
        self,
        *,
        h5_path: Path,
        split: str,
        supervised: bool,
        multiclass: bool,
        num_classes: int,
        dataset_mean: Optional[float],
        dataset_std: Optional[float],
        amount: float,
        mixup: float,
        balance: bool,
        seed: int,
    ) -> None:
        import h5py  # type: ignore
        import numpy as np  # type: ignore

        self.h5_path = Path(h5_path)
        self.split = split
        self.supervised = bool(supervised)
        self.multiclass = bool(multiclass)
        self.num_classes = int(num_classes)
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.amount = float(amount)
        self.mixup = float(mixup)
        self.balance = bool(balance)
        self.seed = int(seed)
        with h5py.File(self.h5_path, "r") as h5:
            n_rows = int(h5["spectrograms"].shape[0])
            splits = [clean(value) for value in h5["splits"][:]] if "splits" in h5 else ["unknown"] * n_rows
            labels = [label_tokens(value) for value in h5["label_strings"][:]]
            sources = [clean(value) for value in h5["sources"][:]] if "sources" in h5 else [str(idx) for idx in range(n_rows)]
            h5_label_names = [clean(value) for value in h5["anomaly_label_names"][:]] if "anomaly_label_names" in h5 else []

        selected = [
            {
                "index": idx,
                "labels": labels[idx],
                "source": sources[idx],
                "is_anomalous": not is_normal_label(labels[idx]),
            }
            for idx in range(n_rows)
            if splits[idx] == split and (self.supervised or is_normal_label(labels[idx]))
        ]
        if self.supervised and self.balance:
            normal = [sample for sample in selected if not sample["is_anomalous"]]
            anomalous = [sample for sample in selected if sample["is_anomalous"]]
            limit = min(len(normal), len(anomalous))
            if limit > 0:
                rng = np.random.default_rng(self.seed + (0 if split == "train" else 1009))
                normal_idx = np.sort(rng.choice(len(normal), size=limit, replace=False))
                anomalous_idx = np.sort(rng.choice(len(anomalous), size=limit, replace=False))
                selected = [normal[int(i)] for i in normal_idx] + [anomalous[int(i)] for i in anomalous_idx]
                selected.sort(key=lambda sample: int(sample["index"]))
        if not selected:
            mode = "supervised" if self.supervised else "normal-only SSL"
            raise ValueError(f"No {mode} rows found for H5 split {split!r} in {self.h5_path}")
        self.sample_info = selected
        label_names = ["normal"] + [label for label in h5_label_names if label and label != "normal"]
        if not h5_label_names:
            seen = sorted({label for sample in selected for label in sample["labels"] if label and label != "normal"})
            label_names = ["normal"] + seen
        self.label_to_index = {label: idx for idx, label in enumerate(label_names[: self.num_classes])}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self) -> int:
        return len(self.sample_info)

    def _label_for_sample(self, sample: dict[str, Any]):
        import torch  # type: ignore

        if self.multiclass:
            for label in sample["labels"]:
                if label in self.label_to_index:
                    return torch.tensor(self.label_to_index[label], dtype=torch.long)
            return torch.tensor(0, dtype=torch.long)
        return torch.tensor(float(sample["is_anomalous"]), dtype=torch.float32)

    def _load_tensor_and_label(self, sample_idx: int):
        import h5py  # type: ignore
        import torch  # type: ignore

        sample = self.sample_info[sample_idx]
        with h5py.File(self.h5_path, "r") as h5:
            data = h5["spectrograms"][int(sample["index"])]
        data = normalize_spectrogram(data, dataset_mean=self.dataset_mean, dataset_std=self.dataset_std, amount=self.amount)
        tensor = torch.from_numpy(data).permute(2, 0, 1)
        return tensor, self._label_for_sample(sample), clean(sample["source"])

    def __getitem__(self, idx: int):
        import numpy as np  # type: ignore

        tensor, label, source = self._load_tensor_and_label(idx)
        if self.supervised and self.split == "train" and self.mixup > 0 and np.random.random() < self.mixup:
            mix_idx = int(np.random.randint(0, len(self.sample_info)))
            mix_tensor, mix_label, _ = self._load_tensor_and_label(mix_idx)
            lam = float(np.random.beta(0.4, 0.4))
            tensor = lam * tensor + (1.0 - lam) * mix_tensor
            if not self.multiclass:
                label = lam * label + (1.0 - lam) * mix_label
        return tensor, label, source


def dataset_lengths(datasets: Iterable[Any]) -> dict[str, int]:
    names = ["ssl_train", "ssl_val", "train", "val"]
    return {name: len(dataset) for name, dataset in zip(names, datasets) if dataset is not None}


def run_training(args: argparse.Namespace) -> None:
    import torch  # type: ignore

    is_pretrain = "pretrain" in args.task
    ssl_train = H5SplitSpectrogramDataset(
        h5_path=Path(args.data_train),
        split="train",
        supervised=False,
        multiclass=False,
        num_classes=2,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        amount=args.amount,
        mixup=0.0,
        balance=False,
        seed=args.split_seed,
    )
    ssl_val = H5SplitSpectrogramDataset(
        h5_path=Path(args.data_train),
        split="val",
        supervised=False,
        multiclass=False,
        num_classes=2,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        amount=args.amount,
        mixup=0.0,
        balance=False,
        seed=args.split_seed,
    )
    train = H5SplitSpectrogramDataset(
        h5_path=Path(args.data_train),
        split="train",
        supervised=True,
        multiclass=args.multiclass,
        num_classes=args.num_classes if args.multiclass else 2,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        amount=args.amount,
        mixup=args.mixup,
        balance=True,
        seed=args.split_seed,
    )
    val = H5SplitSpectrogramDataset(
        h5_path=Path(args.data_train),
        split="val",
        supervised=True,
        multiclass=args.multiclass,
        num_classes=args.num_classes if args.multiclass else 2,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        amount=args.amount,
        mixup=0.0,
        balance=True,
        seed=args.split_seed,
    )
    if is_pretrain:
        train_dataset = ssl_train
        val_dataset = ssl_val
    else:
        train_dataset = train
        val_dataset = val

    pin_memory = bool(torch.cuda.is_available())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=is_pretrain,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print(json.dumps({"dataset_lengths": dataset_lengths([ssl_train, ssl_val, train, val])}, sort_keys=True))
    if is_pretrain:
        from onc_ssamba.traintest_mask import trainmask  # type: ignore

        trainmask(None, train_loader, val_loader, args)
    else:
        from onc_ssamba.traintest import train  # type: ignore

        train(None, train_loader, val_loader, args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[WARN] Ignoring unknown args: {unknown}", file=sys.stderr)
    args = normalize_args(args)
    write_run_config(args, unknown)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
