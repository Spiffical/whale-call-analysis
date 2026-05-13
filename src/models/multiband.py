"""Multi-band acoustic fusion models."""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tvm
except Exception:
    tvm = None


class ResNetEncoder(nn.Module):
    def __init__(self, arch: Literal["resnet18", "resnet34", "resnet50"] = "resnet18", in_ch: int = 1):
        super().__init__()
        if tvm is None:
            raise RuntimeError("torchvision is required for ResNet multiband models")
        if arch == "resnet18":
            net = tvm.resnet18(weights=None)
        elif arch == "resnet34":
            net = tvm.resnet34(weights=None)
        elif arch == "resnet50":
            net = tvm.resnet50(weights=None)
        else:
            raise ValueError(f"Unknown ResNet arch: {arch}")
        if in_ch != 3:
            old = net.conv1.weight
            net.conv1 = nn.Conv2d(
                in_ch,
                net.conv1.out_channels,
                kernel_size=net.conv1.kernel_size,
                stride=net.conv1.stride,
                padding=net.conv1.padding,
                bias=False,
            )
            if old.shape[1] == 3 and in_ch == 1:
                net.conv1.weight.data = old.data.mean(dim=1, keepdim=True)
        self.embedding_dim = int(net.fc.in_features)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepCNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 64, depth: int = 8, dropout: float = 0.3):
        super().__init__()
        channels = [in_ch] + [width] * int(depth)
        blocks = []
        for idx in range(int(depth)):
            blocks.extend(
                [
                    nn.Conv2d(channels[idx], channels[idx + 1], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[idx + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
            if (idx % 2) == 1:
                blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = int(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        return self.dropout(x)


def _make_encoder(name: str, in_ch: int = 1) -> nn.Module:
    name = name.lower()
    if name in {"resnet18", "resnet34", "resnet50"}:
        return ResNetEncoder(arch=name, in_ch=in_ch)
    if name.startswith("deepcnn"):
        width = 64
        depth = 8
        if ":" in name:
            for part in name.split(":")[1:]:
                if part.startswith("w"):
                    width = int(part[1:])
                elif part.startswith("d"):
                    depth = int(part[1:])
        return DeepCNNEncoder(in_ch=in_ch, width=width, depth=depth)
    raise ValueError(f"Unknown multiband encoder: {name}")


class MultiBandFusionModel(nn.Module):
    """Independent band encoders with late fusion.

    ``fusion='gated'`` learns label-wise band weights, which is a close fit for
    mixed-source whale data where some species live almost entirely in one band.
    """

    def __init__(
        self,
        *,
        bands: Sequence[str],
        num_classes: int,
        encoder: str = "resnet18",
        fusion: str = "gated",
        dropout: float = 0.3,
        in_ch: int = 1,
    ) -> None:
        super().__init__()
        self.bands = tuple(str(band) for band in bands)
        if not self.bands:
            raise ValueError("At least one band is required")
        self.num_classes = int(num_classes)
        self.fusion = str(fusion).lower()
        self.branches = nn.ModuleDict({band: _make_encoder(encoder, in_ch=in_ch) for band in self.bands})
        dims = [int(getattr(self.branches[band], "embedding_dim")) for band in self.bands]
        if len(set(dims)) != 1:
            raise ValueError(f"All branch encoders must expose the same embedding_dim, got {dims}")
        self.embedding_dim = dims[0]
        self.dropout = nn.Dropout(dropout)
        self.branch_heads = nn.ModuleDict(
            {band: nn.Linear(self.embedding_dim, self.num_classes) for band in self.bands}
        )
        if self.fusion == "gated":
            self.gate = nn.Linear(self.embedding_dim * len(self.bands), len(self.bands) * self.num_classes)
        elif self.fusion == "concat":
            self.head = nn.Linear(self.embedding_dim * len(self.bands), self.num_classes)
        elif self.fusion in {"mean", "mean_logits"}:
            pass
        else:
            raise ValueError("fusion must be one of: gated, concat, mean_logits")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        branch_logits = []
        for band in self.bands:
            if band not in inputs:
                raise KeyError(f"Missing input band {band!r}")
            emb = self.dropout(self.branches[band](inputs[band]))
            embeddings.append(emb)
            branch_logits.append(self.branch_heads[band](emb))
        stacked_logits = torch.stack(branch_logits, dim=1)
        if self.fusion in {"mean", "mean_logits"}:
            return stacked_logits.mean(dim=1)
        concat = torch.cat(embeddings, dim=1)
        if self.fusion == "concat":
            return self.head(concat)
        gate_logits = self.gate(concat).view(concat.shape[0], len(self.bands), self.num_classes)
        weights = torch.softmax(gate_logits, dim=1)
        return (weights * stacked_logits).sum(dim=1)


def create_multiband_model(
    *,
    encoder: str,
    num_classes: int,
    bands: Sequence[str] = ("low", "mid", "high"),
    fusion: str = "gated",
    dropout: float = 0.3,
    in_ch: int = 1,
) -> MultiBandFusionModel:
    return MultiBandFusionModel(
        bands=bands,
        num_classes=num_classes,
        encoder=encoder,
        fusion=fusion,
        dropout=dropout,
        in_ch=in_ch,
    )


def load_resnet_encoder_checkpoint(
    model: nn.Module,
    checkpoint: Dict[str, torch.Tensor],
    *,
    bands: Iterable[str],
) -> Dict[str, int]:
    """Load matching single-band ResNet tensors into selected branch encoders."""

    clean_state = {
        str(key).removeprefix("module.").removeprefix("net."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor) and not str(key).endswith("fc.weight") and not str(key).endswith("fc.bias")
    }
    out: Dict[str, int] = {}
    branches = getattr(model, "branches", {})
    for band in bands:
        branch = branches[band] if band in branches else None
        if branch is None or not isinstance(branch, ResNetEncoder):
            out[str(band)] = 0
            continue
        current = branch.net.state_dict()
        matched = {
            key: tensor
            for key, tensor in clean_state.items()
            if key in current and tuple(current[key].shape) == tuple(tensor.shape)
        }
        current.update(matched)
        branch.net.load_state_dict(current)
        out[str(band)] = len(matched)
    return out
