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
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


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


def dataset_lengths(datasets: Iterable[Any]) -> dict[str, int]:
    names = ["ssl_train", "ssl_val", "test", "train", "val", "excluded_test"]
    return {name: len(dataset) for name, dataset in zip(names, datasets) if dataset is not None}


def run_training(args: argparse.Namespace) -> None:
    import torch  # type: ignore
    from onc_ssamba.dataset import get_onc_spectrogram_data  # type: ignore

    datasets = get_onc_spectrogram_data(
        data_path=args.data_train,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_length=args.target_length,
        num_mel_bins=args.num_mel_bins,
        freqm=args.freqm,
        timem=args.timem,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        mixup=args.mixup,
        ood=args.ood,
        amount=args.amount,
        subsample_test=args.subsample_test,
        exclude_labels=args.exclude_labels,
        multiclass=args.multiclass,
        num_classes=args.num_classes,
    )
    if len(datasets) == 6:
        ssl_train, ssl_val, _test, train, val, _excluded = datasets
    else:
        ssl_train, ssl_val, _test, train, val = datasets

    is_pretrain = "pretrain" in args.task
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
    print(json.dumps({"dataset_lengths": dataset_lengths(datasets)}, sort_keys=True))
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
