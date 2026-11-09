"""Проверить checkpoint практики и при необходимости повторить валидационный инференс."""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

from kidney_vessels.data.checkpoint_archive import validate_checkpoint_archive

# Архитектуры относятся к трём сохранённым экспериментам практики.
MODELS = {
    "best_model.pth": ("Unet", "mit_b2", "scse"),
    "best_model_effb2_unetplusplus_0.5308.pth": ("UnetPlusPlus", "efficientnet-b2", "scse"),
    "best_model_resnet34_0.5566.pth": ("Unet", "resnet34", None),
}


@dataclass
class PixelCounts:
    """Накопленные счётчики и сумма Dice по тайлам для одного порога."""

    intersection: int = 0
    predicted: int = 0
    reference: int = 0
    dice_sum: float = 0.0

    def update(self, intersection: int, predicted: int, reference: int) -> None:
        self.intersection += intersection
        self.predicted += predicted
        self.reference += reference
        self.dice_sum += (2 * intersection + 1e-6) / (predicted + reference + 1e-6)

    def metrics(self, tiles: int) -> dict[str, float]:
        return {
            "dice_macro": self.dice_sum / tiles,
            "dice_micro": 2 * self.intersection / max(1, self.predicted + self.reference),
            "iou_micro": self.intersection
            / max(1, self.predicted + self.reference - self.intersection),
            "precision_micro": self.intersection / max(1, self.predicted),
            "recall_micro": self.intersection / max(1, self.reference),
        }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _load_weights(path: Path, torch: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    """Загрузить на CPU только после проверки контейнера и статического анализа globals."""
    validate_checkpoint_archive(path)
    release = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if release is None or (int(release[1]), int(release[2])) < (2, 6):
        raise ValueError("Для проверки checkpoint требуется PyTorch 2.6 или новее")
    # Изолированный процесс аудита использует только штатный список разрешений PyTorch.
    torch.serialization.clear_safe_globals()
    if unsafe := torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise ValueError(f"Checkpoint содержит неподдерживаемые globals: {unsafe}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Ожидается state_dict или словарь с model_state")
    state = payload.get("model_state", payload)
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(key, str)
            or type(value) is not torch.Tensor
            or value.layout != torch.strided
            or not torch.isfinite(value).all().item()
            for key, value in state.items()
        )
    ):
        raise ValueError("state_dict должен содержать только конечные плотные тензоры")
    metadata = {
        key: payload[key] for key in ("config", "epoch", "val_dice", "threshold") if key in payload
    }
    json.dumps(metadata, allow_nan=False)
    return state, metadata


def _validation_ids(data: Path, wsi: set[str]) -> list[str]:
    """Восстановить порядок тайлов по tile_meta и наличию записи JSONL."""
    with (data / "polygons.jsonl").open(encoding="utf-8-sig") as stream:
        annotated = {json.loads(line)["id"] for line in stream if line.strip()}
    with (data / "tile_meta.csv").open(encoding="utf-8-sig", newline="") as stream:
        ids = [
            row["id"]
            for row in csv.DictReader(stream)
            if row["id"] in annotated and row["source_wsi"] in wsi
        ]
    if not ids or len(ids) != len(set(ids)) or any(not image_id.isalnum() for image_id in ids):
        raise ValueError("Нужен непустой список уникальных идентификаторов валидации")
    return ids


def _accumulate(
    probabilities: Any,
    target: Any,
    thresholds: Any,
    counts: list[PixelCounts],
    reference: int,
) -> None:
    predicted = probabilities > thresholds
    positive = predicted.sum(dim=(1, 2, 3)).tolist()
    overlap = (predicted & target).sum(dim=(1, 2, 3)).tolist()
    for counter, intersection, total in zip(counts, overlap, positive, strict=True):
        counter.update(intersection, total, reference)


def _evaluate(
    data: Path, model: Any, torch: ModuleType, device: str, wsi: set[str], saved: float
) -> dict[str, object]:
    """Воспроизвести инференс на WSI валидации и два протокола выбора порога."""
    image = import_module("PIL.Image")
    np = import_module("numpy")
    metadata_hashes = {name: _sha256(data / name) for name in ("polygons.jsonl", "tile_meta.csv")}
    ids = _validation_ids(data, wsi)
    if len(ids) > 1000:
        raise ValueError("Сценарий предназначен для ограниченной выборки до 1000 тайлов")
    raw_grid = [float(value) for value in np.arange(0.30, 0.81, 0.025)]
    tta_grid = [float(value) for value in np.arange(0.30, 0.86, 0.05)]
    raw_counters = [PixelCounts() for _ in range(len(raw_grid) + 2)]
    tta_counters = [PixelCounts() for _ in range(len(tta_grid) + 1)]
    raw_thresholds = torch.tensor([*raw_grid, saved, 0.5], device=device).view(-1, 1, 1, 1)
    tta_thresholds = torch.tensor([*tta_grid, saved], device=device).view(-1, 1, 1, 1)
    cache_hashes: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    with torch.inference_mode():
        for index, image_id in enumerate(ids, 1):
            image_path = data / "train" / f"{image_id}.tif"
            mask_path = data / "mask_cache" / f"{image_id}.png"
            image_hashes[image_id] = _sha256(image_path)
            cache_hashes[image_id] = _sha256(mask_path)
            with image.open(image_path) as tile, image.open(mask_path) as mask:
                if tile.size != (512, 512) or mask.size != tile.size or mask.mode != "L":
                    raise ValueError("Ожидается тайл 512×512 и одноканальная маска того же размера")
                rgb = np.array(tile.convert("RGB"), dtype=np.float32) / 255.0
                pixels = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                target = torch.from_numpy(np.array(mask) > 127).view(1, 1, 512, 512).to(device)
            raw = model(pixels).sigmoid()
            flips = [(3,), (2,), (2, 3)]
            tta = torch.stack(
                [raw, *[model(pixels.flip(dims)).sigmoid().flip(dims) for dims in flips]]
            ).mean(dim=0)
            if not (torch.isfinite(raw).all() & torch.isfinite(tta).all()).item():
                raise ValueError("Инференс вернул нечисловые вероятности")
            reference = int(target.sum().item())
            _accumulate(raw, target, raw_thresholds, raw_counters, reference)
            _accumulate(tta, target, tta_thresholds, tta_counters, reference)
            if index % 50 == 0:
                print(f"Проверено тайлов: {index}/{len(ids)}", flush=True)

    def row(threshold: float, counts: PixelCounts) -> dict[str, object]:
        return {"threshold": threshold, **counts.metrics(len(ids)), "pixel_counts": asdict(counts)}

    raw_best = max(range(len(raw_grid)), key=lambda index: raw_counters[index].dice_sum)
    tta_best = max(range(len(tta_grid)), key=lambda index: tta_counters[index].dice_sum)
    return {
        "tiles": len(ids),
        "validation_wsi": sorted(wsi),
        "validation_ids_sha256": sha256("\n".join(ids).encode()).hexdigest(),
        "image_sha256": image_hashes,
        "mask_cache_sha256": cache_hashes,
        "source_metadata_sha256": metadata_hashes,
        "raw_saved_threshold": row(saved, raw_counters[-2]),
        "raw_threshold_0_5": row(0.5, raw_counters[-1]),
        "raw_best_macro_threshold": row(raw_grid[raw_best], raw_counters[raw_best]),
        "tta_saved_threshold": row(saved, tta_counters[-1]),
        "tta_best_macro_threshold": row(tta_grid[tta_best], tta_counters[tta_best]),
        "raw_grid": [row(t, c) for t, c in zip(raw_grid, raw_counters[:-2], strict=True)],
        "tta_grid": [row(t, c) for t, c in zip(tta_grid, tta_counters[:-1], strict=True)],
    }


def main() -> int:
    """Запустить проверку в отдельном окружении с PyTorch и SMP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=MODELS, default="best_model.pth")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    practice = args.practice.resolve()
    output = args.output.resolve()
    if output.is_relative_to(practice) or output.exists():
        parser.error("Отчёт должен быть новым файлом вне каталога практики")
    path = practice / "checkpoints" / args.checkpoint
    architecture, encoder, attention = MODELS[args.checkpoint]
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    torch = import_module("torch")
    smp = import_module("segmentation_models_pytorch")
    torch.set_num_threads(2)
    state, metadata = _load_weights(path, torch)
    config = metadata.get("config", {})
    if not isinstance(config, dict) or config.get("encoder_name", encoder) != encoder:
        raise ValueError("Конфигурация checkpoint не совпадает с историческим энкодером")
    model = getattr(smp, architecture)(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type=attention,
    )
    model.load_state_dict(state, strict=True)
    del state
    result: dict[str, object] = {
        "checkpoint": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "checkpoint_metadata": metadata,
        "architecture": architecture,
        "encoder": encoder,
        "decoder_attention": attention,
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_entries": len(model.state_dict()),
        "strict_load": True,
        "python": sys.version.split()[0],
        "device": args.device,
        "libraries": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "segmentation-models-pytorch",
                "timm",
                "Pillow",
                "numpy",
            )
        },
    }
    if args.evaluate:
        device = args.device
        if device == "cuda":
            result["gpu"] = torch.cuda.get_device_name(0)
            torch.backends.cudnn.benchmark = False
        model.to(device).eval()
        saved = float(metadata.get("threshold", 0.5))
        result["evaluation"] = _evaluate(
            practice / "data",
            model,
            torch,
            device,
            {str(value) for value in config.get("val_wsi", [4])},
            saved,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Проверен {path.name}, отчёт {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
