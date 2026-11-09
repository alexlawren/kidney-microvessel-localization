"""Проверить разбиение и маски проекта практики по готовому аудиту HuBMAP."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import TypedDict

from PIL import Image, ImageDraw


class SourceAnnotation(TypedDict):
    """Поля уже проверенной записи JSONL, которые использовал старый растеризатор."""

    type: str
    coordinates: list[list[list[float]]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _coverage(
    images: list[dict[str, str]], annotations: list[dict[str, str]]
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in images:
        groups[(row["dataset"], row["source_wsi"])].append(row)
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)
    rows: list[dict[str, object]] = []
    for (dataset, wsi), tiles in sorted(groups.items()):
        objects = [a for row in tiles for a in by_image[row["image_id"]]]
        areas = [
            float(row["area_px2"])
            for row in objects
            if row["category"] == "blood_vessel" and row["measurable"] == "True"
        ]
        rows.append(
            {
                "dataset": dataset,
                "source_wsi": wsi,
                "tiles": len(tiles),
                "presence": dict(Counter(row["annotation_presence"] for row in tiles)),
                "categories": dict(Counter(row["category"] for row in objects)),
                "invalid_geometry": sum(row["geometry_valid"] != "True" for row in objects),
                "measurable_vessels": len(areas),
                "vessel_area_median_px2": median(areas) if areas else None,
            }
        )
    return rows


def _mask_check(
    annotations: list[SourceAnnotation], dimensions: tuple[int, int], cache: Path
) -> tuple[str, bool]:
    """Повторить прежнюю бинарную растеризацию в памяти и сравнить пиксели с PNG."""
    with Image.new("L", dimensions) as rendered:
        draw = ImageDraw.Draw(rendered)
        for annotation in annotations:
            if annotation["type"] == "blood_vessel":
                for part in annotation["coordinates"]:
                    draw.polygon([(point[0], point[1]) for point in part], fill=255)
        empty = rendered.getbbox() is None
        if not cache.is_file():
            return "missing", empty
        with Image.open(cache) as mask:
            match = (
                mask.mode == "L"
                and mask.size == rendered.size
                and mask.tobytes() == rendered.tobytes()
            )
        return ("match" if match else "mismatch"), empty


def _split_report(
    data: Path,
    rows: list[dict[str, str]],
    raw: dict[str, list[SourceAnnotation]],
    fingerprints: dict[str, str],
) -> tuple[dict[str, object], Counter[str], Counter[str]]:
    cache: Counter[str] = Counter()
    byte_hashes: Counter[str] = Counter()
    rgb_hashes: Counter[str] = Counter()
    masks_without_vessels = 0
    footprints: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    for row in rows:
        image_id = row["image_id"]
        metadata = json.loads(row["tile_metadata"])
        relative = f"train/{image_id}.tif"
        path = data / relative
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
            if digest != fingerprints.get(relative):
                raise ValueError(f"Файл не совпадает с manifest: {relative}")
            stream.seek(0)
            with Image.open(stream) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                if width != height:
                    raise ValueError("Сравнение смещений ожидает квадратные тайлы практики")
                digest_rgb = sha256(rgb.tobytes()).hexdigest()
        byte_hashes[digest] += 1
        rgb_hashes[digest_rgb] += 1
        status, empty = _mask_check(
            raw[image_id], (width, height), data / "mask_cache" / f"{image_id}.png"
        )
        cache[status] += 1
        masks_without_vessels += empty
        footprints[row["source_wsi"]].append(
            (int(metadata["i"]), int(metadata["j"]), width, row["dataset"])
        )
    overlaps = 0
    across_datasets = 0
    for tiles in footprints.values():
        for (i, j, side, dataset), (i2, j2, side2, dataset2) in combinations(tiles, 2):
            if i < i2 + side2 and i2 < i + side and j < j2 + side2 and j2 < j + side:
                overlaps += 1
                across_datasets += dataset != dataset2
    return (
        {
            "tiles": len(rows),
            "by_wsi": dict(Counter(row["source_wsi"] for row in rows)),
            "by_dataset": dict(Counter(row["dataset"] for row in rows)),
            "cache": dict(cache),
            "masks_without_vessel_pixels": masks_without_vessels,
            "patient_id_missing": sum(not row["patient_id"] for row in rows),
            "within_split_overlap_pairs": overlaps,
            "within_split_overlap_pairs_across_datasets": across_datasets,
            "image_ids_sha256": sha256(
                "\n".join(row["image_id"] for row in rows).encode()
            ).hexdigest(),
        },
        byte_hashes,
        rgb_hashes,
    )


def audit_split(
    data: Path, audit: Path, train_wsi: set[str], val_wsi: set[str]
) -> dict[str, object]:
    """Восстановить выборки практики и проверить идентичность изображений и масок."""
    manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    if Path(manifest["config"]["input"]).resolve() != data.resolve():
        raise ValueError("Аудит выполнен для другого каталога данных")
    fingerprints: dict[str, str] = {row["path"]: row["sha256"] for row in manifest["input_files"]}
    for name in ("polygons.jsonl", "tile_meta.csv", "wsi_meta.csv"):
        if _sha256(data / name) != fingerprints.get(name):
            raise ValueError(f"Метаданные изменились после аудита: {name}")
    raw: dict[str, list[SourceAnnotation]] = {}
    with (data / "polygons.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["id"]
            if not isinstance(image_id, str) or not image_id.isalnum() or image_id in raw:
                raise ValueError("Нужны однозначные идентификаторы записей JSONL")
            raw[image_id] = record["annotations"]
    images = _read_csv(audit / "images.csv")
    index = {row["image_id"]: row for row in images}
    if len(index) != len(images) or not raw.keys() <= index.keys():
        raise ValueError("Записи JSONL не имеют однозначной связи с train-тайлами")
    annotated = []
    for metadata in _read_csv(data / "tile_meta.csv"):
        if metadata["id"] not in raw:
            continue
        row = index[metadata["id"]]
        if (
            row["source_wsi"] != metadata["source_wsi"].strip()
            or row["dataset"] != metadata["dataset"].strip()
            or json.loads(row["tile_metadata"]) != metadata
        ):
            raise ValueError("Строка images.csv не соответствует исходной tile_meta.csv")
        annotated.append(row)
    if len(annotated) != len(raw) or {row["image_id"] for row in annotated} != raw.keys():
        raise ValueError("Для каждой записи JSONL нужен ровно один тайл из tile_meta.csv")
    train = [row for row in annotated if row["source_wsi"] in train_wsi]
    val = [row for row in annotated if row["source_wsi"] in val_wsi]
    if not train or not val:
        raise ValueError("Каждая выборка должна содержать тайлы с записями JSONL")
    train_report, train_bytes, train_rgb = _split_report(data, train, raw, fingerprints)
    val_report, val_bytes, val_rgb = _split_report(data, val, raw, fingerprints)
    train_patients = {row["patient_id"] for row in train}
    val_patients = {row["patient_id"] for row in val}
    return {
        "source_manifest_sha256": _sha256(audit / "manifest.json"),
        "audit_csv_sha256": {
            name: _sha256(audit / name) for name in ("images.csv", "annotations.csv")
        },
        "coverage": _coverage(images, _read_csv(audit / "annotations.csv")),
        "train": train_report,
        "validation": val_report,
        "cross_split": {
            "tiles": len({r["image_id"] for r in train} & {r["image_id"] for r in val}),
            "wsi": sorted({r["source_wsi"] for r in train} & {r["source_wsi"] for r in val}),
            "patients": (
                None
                if "" in train_patients | val_patients
                else sorted(train_patients & val_patients)
            ),
            "identical_file_pairs": sum(
                count * val_bytes[key] for key, count in train_bytes.items()
            ),
            "identical_rgb_pairs": sum(count * val_rgb[key] for key, count in train_rgb.items()),
        },
    }


def main() -> int:
    """Сохранить отдельный отчёт о разбиении без изменения исходного аудита."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-wsi", nargs="+", required=True)
    parser.add_argument("--val-wsi", nargs="+", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(args.input.resolve()) or output.exists():
        parser.error("Отчёт должен быть новым файлом вне каталога данных")
    result = audit_split(args.input, args.audit, set(args.train_wsi), set(args.val_wsi))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
